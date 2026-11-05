from pathlib import Path
import json
import random
import wandb
import gc
from statistics import mean, variance, stdev
import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

from cs336_alignment.training_step import grpo_train_step, evaluate_policy

def main():
    prompt_dir = Path(__file__).resolve().parent / "prompts"
    question_only = (prompt_dir / "question_only.prompt").read_text(encoding="utf-8")    
    question_only = question_only.replace(r"\\boxed", r"\boxed")
    zero_shot = (prompt_dir / "r1_zero.prompt").read_text(encoding="utf-8")
    three_shot = (prompt_dir / "r1_zero_three_shot_gsm8k.prompt").read_text(encoding="utf-8")
    templates = ["question_only", "zero_shot","three_shot"]
    prompts = {"question_only": question_only,"zero_shot":zero_shot,
                        "three_shot":three_shot}
    reward_functions = {"question_only": question_only_reward_fn,"zero_shot":r1_zero_reward_fn,
                        "three_shot":r1_zero_reward_fn}

    seeds = [0,1,2]

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

    training_file_path = Path("data/gsm8k/train.jsonl")
    validation_file_path = Path("data/gsm8k/test.jsonl")

    model_id = 'allenai/OLMo-2-0425-1B'
    model_device = 'cuda:0'
    server_device = 1

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

    results = {template: {} for template in templates}

    for template in templates:
        for seed in seeds:
            final_reward = 0
            raw_prompt = prompts[template]
            reward_function = reward_functions[template]

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
            }
            if template != "question_only":
                sampling_params["stop"] = ["</answer>"]
                sampling_params["include_stop_str_in_output"] = True
            with wandb.init(
                project = 'cs336-grpo-tuning-prompt-ablation',
                name = f'{template}-seed-{seed}',
                config = {
                    "prompt_name":template,
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
                        prompt_template=raw_prompt,
                        sampling_params=sampling_params,
                        reward_fn=reward_function,
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

                    print(f"step={step}, validation={metrics}", flush=True)
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
                            prompt = raw_prompt.format(question = question)
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
                            reward_function,
                            repeated_prompts,
                            rollout_responses,
                            repeated_ground_truths,
                            G,
                            advantage_eps= 1e-6
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
                            results[template][seed] = final_reward
                finally:
                    try:
                        server.stop()
                    finally:
                        del optimizer
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
    #汇总
    summary_stats = {}

    for template in templates:
        seed_results = results[template]

        if set(seed_results) != set(seeds):
            raise ValueError(
                f"{template} 的 seed 不完整，实际得到 {list(seed_results)}"
            )

        rewards = [seed_results[seed] for seed in seeds]

        summary_stats[template] = {
            "mean": mean(rewards),
            "variance": variance(rewards),
            "std": stdev(rewards),
        }

        print(template, summary_stats[template])

    rows = [
        [template, stats["mean"], stats["variance"], stats["std"]]
        for template, stats in summary_stats.items()
    ]

    table = wandb.Table(
        columns=["prompt_name", "mean_reward", "variance", "std"],
        data=rows,
    )

    with wandb.init(
        project="cs336-grpo-tuning-prompt-ablation",
        name="prompt-ablation-summary",
        job_type="summary",
    ) as summary_run:
        summary_run.log({
            "prompt_statistics": table,

            "mean_reward_comparison": wandb.plot.bar(
                table,
                "prompt_name",
                "mean_reward",
                title="Mean final validation reward",
            ),

            "variance_comparison": wandb.plot.bar(
                table,
                "prompt_name",
                "variance",
                title="Variance across seeds",
            ),
        })
if __name__ == '__main__':
    main()