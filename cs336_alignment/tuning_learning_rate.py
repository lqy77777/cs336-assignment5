from pathlib import Path
import json
import random
import wandb
import gc
import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from cs336_alignment.training_step import grpo_train_step, evaluate_policy

def main():
    learning_rates = [1e-4,3e-5,1e-5,3e-6,1e-6]
    final_rewards = []

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

    seed = 0
    
    sampling_params = {
            "temperature": 1.0, 
            "max_tokens": 512, 
            "n": G,     #每个问题生成G个回答
            "seed":seed,
            "stop": '</answer>',
            "include_stop_str_in_output": True
        }

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

    for learning_rate in learning_rates:
        model, tokenizer = get_model_and_tokenizer(model_id, model_device)
        server = VLLMServer(model_id, gpu = server_device)

        # 同一个 seed 下，让各学习率使用相同的问题抽样序列
        rng = random.Random(seed)
        torch.manual_seed(seed)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )
        final_reward = 0
        with wandb.init(
            project = 'cs336-grpo-tuning-learning-rate',
            name = f'lr-{learning_rate:.0e}-seed-{seed}',
            config = {
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
                        final_rewards.append(final_reward)
            finally:
                try:
                    server.stop()
                finally:
                    del optimizer
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    
    # 五个学习率实验全部结束后，汇总最终奖励。
    if len(final_rewards) != len(learning_rates):
        raise ValueError("学习率数量与最终奖励数量不一致，无法汇总")

    rows = sorted(
        [
            [lr, float(reward)]
            for lr, reward in zip(learning_rates, final_rewards)
        ],
        key=lambda row: row[0],
    )

    table = wandb.Table(
        columns=["learning_rate", "final_validation_reward"],
        data=rows,
    )

    best_lr, best_reward = max(rows, key=lambda row: row[1])

    with wandb.init(
        project="cs336-grpo-tuning-learning-rate",
        name=f"learning-rate-summary-seed-{seed}",
        job_type="summary",
        config={
            "seed": seed,
            "learning_rates": learning_rates,
            "rollout_steps": rollout_steps,
            "n_eval": n_eval,
        },
    ) as summary_run:
        summary_run.log({
            "learning_rate_results": table,

            "final_reward_vs_learning_rate": wandb.plot.line(
                table,
                "learning_rate",
                "final_validation_reward",
                title="Final validation reward vs learning rate",
            ),
        })

        summary_run.summary["best_learning_rate"] = best_lr
        summary_run.summary["best_final_validation_reward"] = best_reward
    return
if __name__ == '__main__':
    main()