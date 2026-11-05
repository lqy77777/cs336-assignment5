from pathlib import Path
import json
import random
import wandb
import gc
from statistics import mean, variance, stdev
import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from cs336_alignment.training_step import grpo_train_step, evaluate_policy

grpo = {
    "name":"grpo",
    "baseline": "mean",
    "advantage_normalizer": "std",
    "loss_normalization":"sequence"
}
grpo_constant = {
    "name":"grpo_constant",
    "baseline": "mean",
    "advantage_normalizer": "std",
    "loss_normalization":"constant"
}
dr_grpo = {
    "name":"dr_grpo",
    "baseline": "mean",
    "advantage_normalizer": "none",
    "loss_normalization":"constant"
}
rft = {
    "name":"rft",
    "baseline": "none",
    "advantage_normalizer": "none",
    "loss_normalization":"constant"
}
maxrl = {
    "name":'maxrl',
    "baseline": "mean",
    "advantage_normalizer": "mean",
    "loss_normalization":"constant"
}
varients = [grpo,grpo_constant,dr_grpo,rft, maxrl]
def main():
    learning_rate = 1e-5
    betas = (0.9,0.95)
    weight_decay = 0.0

    B = 32  #question_batch_size
    G = 8   #responses_per_question
    N = B * G  #overall_batch_size
    gradient_accumulation_steps = 32   #一层更新分成几个microbatch
    m = N // gradient_accumulation_steps   #每个microbatch的大小
    max_grad_norm = 1.0
    rollout_steps = 200
    n_eval = 1024   #验证集大小
    n_train = B * rollout_steps

    seeds = [0,1,2]

    training_file_path = Path("data/gsm8k/train.jsonl")
    validation_file_path = Path("data/gsm8k/test.jsonl")

    model_id = 'allenai/OLMo-2-0425-1B'
    model_device = 'cuda:0'
    server_device = 1
    prompt_dir = Path(__file__).resolve().parent / "prompts"
    zero_shot = (prompt_dir / "r1_zero.prompt").read_text(encoding="utf-8")

    #从training file和test file中读取数据
    training_data = []     #list[str]
    with training_file_path.open("r", encoding = 'utf-8') as f:
        for line in f:
            example = json.loads(line)
            question = example['question']
            answer = example['answer']
            ground_truth = answer.split('####', 1)[1].strip()
            training_data.append({'question':question,'ground_truth':ground_truth})
    validation_data = []   #list[str]
    with validation_file_path.open("r", encoding = 'utf-8') as f:
        for line in f:
            example = json.loads(line)
            question = example['question']
            answer = example['answer']
            ground_truth = answer.split('####', 1)[1].strip()
            validation_data.append({'question':question,'ground_truth':ground_truth})

    results = {varient["name"]:{} for varient in varients}

    for varient in varients:
        for seed in seeds:
            final_reward = 0
            model, tokenizer = get_model_and_tokenizer(model_id, model_device)
            server = VLLMServer(model_id, gpu = server_device)
            optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    betas=betas,
                    weight_decay=weight_decay,
                )
            # 同一个 seed 下，让各学习率使用相同的问题抽样序列
            rng = random.Random(seed)
            torch.manual_seed(seed)
            sampling_params = {
                "temperature": 1.0, 
                "max_tokens": 512, 
                "n": G,     #每个问题生成G个回答
                "seed":seed,
                "stop":"</answer>",
                "include_stop_str_in_output": True
            }
            with wandb.init(
                project = 'cs336-grpo-tuning-grpo-varients',
                name = f'{varient["name"]}-seed-{seed}',
                config = {
                    "varient":varient["name"],
                    "seed": seed,
                    "learning_rate": learning_rate,
                    "n_eval": n_eval,
                    "rollout_steps": rollout_steps,
                    "microbatch_size": m,
                }
            ) as run:
                # step：已经完成的训练更新次数，用作曲线横轴。
                def log_validation(step):
                    metrics, sample_rows = evaluate_policy(
                        model=model,
                        server=server,
                        tokenizer=tokenizer,
                        validation_data=validation_data,
                        prompt_template=zero_shot,
                        sampling_params=sampling_params,
                        reward_fn=r1_zero_reward_fn,
                        n_eval=n_eval,
                    )

                    run.log(
                        {
                            **metrics,
                            "validation_samples": wandb.Table(
                                columns=[
                                    "prompt",
                                    "response",
                                    "ground_truth",
                                    "reward",
                                ],
                                data=sample_rows,
                            ),
                        },
                        step=step,
                    )

                    print(f"varient={varient["name"]},seed={seed},step={step}, validation={metrics}", flush=True)
                    return metrics['validation_reward']
                try:
                    model.train()
                    server.start()
                    server.init_weight_sync(policy_device=model_device)
                    # 训练前验证，得到初始表现。
                    log_validation(step=0)
                    for k in range(rollout_steps):
                        print(f"正在进行第{k+1}次step")
                        server.sync_policy_weights(model)    #同步权重
                        #1.采样 B 个questions
                        batch = rng.sample(training_data, B)
                        #2.每个question采样 G 个responses，生成3个输入list[str]
                        repeated_prompts = []
                        rollout_responses = []
                        repeated_ground_truths = []
                        for dictionary in batch:
                            question = dictionary['question']
                            prompt = zero_shot.format(question = question)
                            ground_truth = dictionary['ground_truth']
                            repeated_prompts += [prompt] * G
                            repeated_ground_truths += [ground_truth] * G
                            completions = server.generate_completions([prompt],sampling_params)
                            for response in completions:
                                rollout_responses.append(response.text)
                        batch_loss, metadata = grpo_train_step(
                            model,
                            tokenizer,
                            optimizer,
                            gradient_accumulation_steps,
                            max_grad_norm,
                            r1_zero_reward_fn,
                            repeated_prompts,
                            rollout_responses,
                            repeated_ground_truths,
                            G,
                            advantage_eps= 1e-6,
                            baseline = varient["baseline"],
                            advantage_normalizer=varient["advantage_normalizer"],
                            loss_normalization = varient["loss_normalization"],
                            normalization_constant= N * sampling_params["max_tokens"] if varient["loss_normalization"] == "constant" else None
                        )    
                        # 采样问题 → 生成回答 → 调用 grpo_train_step()
                        step = k + 1
                        if step % 40 == 0:
                            run.log(
                                {
                                    "train/loss": batch_loss.item(),
                                    "train/grad_norm": metadata['grad_norm'].item(),
                                    "train/mean_entropy": metadata['mean_entropy'],
                                    "train/reward": metadata['mean_reward'],
                                    "train/format_reward": metadata['mean_format_reward'],
                                    'train/std_reward':metadata['std_reward']
                                },
                                step=step,
                            )
                        # 每 10 次更新验证一次，最后一次更新后也验证。
                        if step % 10 == 0 or step == rollout_steps:
                            final_reward = log_validation(step)
                        if step == rollout_steps:
                            run.summary["final_validation_reward"] = final_reward
                            results[varient["name"]][seed] = final_reward
                finally:
                    try:
                        server.stop()
                    finally:
                        del optimizer
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()

    summary_rows = []
    seed_rows = []

    for variant_name, seed_rewards in results.items():
        # 避免某个 seed 没有结果时，使用不完整的数据汇总
        if set(seed_rewards) != set(seeds):
            raise ValueError(
                f"{variant_name} 的 seed 结果不完整："
                f"预期 {seeds}，实际 {list(seed_rewards)}"
            )

        rewards = [
            float(seed_rewards[seed])
            for seed in seeds
        ]

        mean_reward = mean(rewards)
        reward_variance = variance(rewards)
        reward_std = stdev(rewards)

        summary_rows.append([
            variant_name,
            len(rewards),
            mean_reward,
            reward_variance,
            reward_std,
        ])

        for seed in seeds:
            seed_rows.append([
                variant_name,
                seed,
                float(seed_rewards[seed]),
            ])

        print(
            f"{variant_name}: "
            f"mean={mean_reward:.6f}, "
            f"variance={reward_variance:.8f}, "
            f"std={reward_std:.6f}",
            flush=True,
        )

    with wandb.init(
        project="cs336-grpo-tuning-grpo-varients",
        name="grpo-variants-summary",
        job_type="summary",
        config={
            "variants": list(results),
            "seeds": seeds,
            "learning_rate": learning_rate,
            "rollout_steps": rollout_steps,
            "n_eval": n_eval,
            "variance_definition": "sample variance, denominator n-1",
        },
    ) as summary_run:
        summary_table = wandb.Table(
            columns=[
                "variant",
                "num_seeds",
                "mean_final_reward",
                "variance_final_reward",
                "std_final_reward",
            ],
            data=summary_rows,
        )

        seed_table = wandb.Table(
            columns=[
                "variant",
                "seed",
                "final_reward",
            ],
            data=seed_rows,
        )

        summary_run.log({
            "variant_summary": summary_table,
            "per_seed_results": seed_table,

            "mean_final_reward_by_variant": wandb.plot.bar(
                summary_table,
                "variant",
                "mean_final_reward",
                title="Mean final reward across seeds",
            ),

            "variance_final_reward_by_variant": wandb.plot.bar(
                summary_table,
                "variant",
                "variance_final_reward",
                title="Sample variance of final reward across seeds",
            ),
        })

        for row in summary_rows:
            summary_run.summary[row[0]] = {
                "num_seeds": row[1],
                "mean_final_reward": row[2],
                "variance_final_reward": row[3],
                "std_final_reward": row[4],
            }
    
if __name__ == '__main__':
    main()