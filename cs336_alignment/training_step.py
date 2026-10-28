from typing import Callable, Literal
from pathlib import Path
import json
import random
import wandb

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

from cs336_alignment.tool import (
    aggregate_loss_across_microbatch_sequence,
    compute_group_normalized_rewards_grpo,
    compute_policy_gradient_loss_on_policy,
    compute_rollout_rewards,
    get_response_log_probs,
    tokenize_prompt_and_output,
)

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal['mean','none'] = 'mean',
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal['std','none','mean'] = 'std',
    importance_reweighting_method: Literal['none','noclip','grpo','gspo'] = 'none',
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal['sequence','constant'] = 'sequence',
    normalization_constant: int | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    device = next(model.parameters()).device
    batch_loss = torch.zeros((), device=device)
    metadata = {}
    N = len(repeated_prompts)     # N = BG
    G = group_size
    B = N // G
    input_dict = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = input_dict['input_ids'].to(device)         # N * S
    labels = input_dict['labels'].to(device)               # N * S
    response_mask = input_dict['response_mask'].to(device) # N * S
    raw_rewards, reward_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)  #长度为N
    metadata['mean_reward'] = float(reward_metadata['mean_reward'])
    metadata['mean_format_reward'] = float(reward_metadata['mean_format_reward'])
    normed_advantages = compute_group_normalized_rewards_grpo(raw_rewards,G,baseline,advantage_eps
                                                              ,advantage_normalizer)[0]  #长度为N
    normed_advantages = normed_advantages.to(device)
    m = N //gradient_accumulation_steps   #microbatch_size
    entropy_sum = torch.zeros((), device=device)
    
    for i in range(0,N, m):
        response_outputs = get_response_log_probs(
            model, input_ids[i:i+m], labels[i:i+m], return_token_entropy=True,
        )
        log_probs = response_outputs['log_probs']   # m * S
        # 按整轮回答 token 数加权，排除 prompt 和 padding。
        entropy_sum += (response_outputs['token_entropy'] * response_mask[i:i+m]).sum()
        per_loss = compute_policy_gradient_loss_on_policy(normed_advantages[i:i+m],log_probs,importance_reweighting_method,
                                                old_log_probs,cliprange,response_mask[i:i+m])[0]  # m * (T-1)
        loss = aggregate_loss_across_microbatch_sequence(per_loss, response_mask[i:i+m], loss_normalization,
                                                         normalization_constant)
        batch_loss += (loss/gradient_accumulation_steps).detach()
        (loss/gradient_accumulation_steps).backward()
    metadata['token_entropy'] = (entropy_sum / response_mask.sum()).item()
    #梯度裁剪
    clip_limit = max_grad_norm if max_grad_norm is not None else float('inf')
    grad_norm = clip_grad_norm_(model.parameters(), max_norm = clip_limit)
    metadata['grad_norm'] = grad_norm
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss, metadata

def evaluate_policy(
    model,
    server,
    tokenizer,
    validation_data: list[dict[str,str]],
    prompt_template,
    sampling_params,
    reward_fn,
    n_eval=1024,
):
    # 每次验证使用相同的数据，方便比较训练前后的变化。
    validations = validation_data[:n_eval]

    prompts = [prompt_template.format(question=validation["question"]) for validation in validations]
    ground_truths = [validation["ground_truth"] for validation in validations]

    # 验证当前训练模型，而不是 vLLM 中残留的旧权重。
    server.sync_policy_weights(model)
    #验证的推理参数需要和训练的推理参数不同, n=1每个问题生成一个回答
    eval_params = dict(sampling_params)
    eval_params.update(temperature=0.0, n=1)

    completions = server.generate_completions(
        prompts,
        eval_params,
        batch_size=32,   #每次向 vLLM 服务提交多少个 prompt，不重要
    )
    responses = [completion.text for completion in completions]

    rewards, reward_metadata = compute_rollout_rewards(
        reward_fn,
        responses,
        ground_truths,
    )

    # 优先使用生成服务返回的 token ID；缺失时重新 tokenize。用来计算response的长度
    lengths = [
        len(completion.token_ids)
        if completion.token_ids
        else len(tokenizer.encode(
            completion.text,
            add_special_tokens=False,
        ))
        for completion in completions
    ]

    metrics = {
        "validation_reward": float(reward_metadata["mean_reward"]),
        "validation_format_reward": float(
            reward_metadata["mean_format_reward"]
        ),
        "validation_response_length": sum(lengths) / len(lengths),
    }

    # 保存固定的几条回答，便于比较训练前后。
    sample_rows = [
        [prompt, response, truth, float(reward)]
        for prompt, response, truth, reward in zip(
            prompts[:4],
            responses[:4],
            ground_truths[:4],
            rewards[:4],
        )
    ]

    return metrics, sample_rows
def main():
    #0.初始化超参数、模型等等
    learning_rate = 1e-5
    betas = (0.9,0.95)
    weight_decay = 0.0

    B = 32  #question_batch_size
    G = 8   #responses_per_question
    N = B * G  #overall_batch_size
    gradient_accumulation_steps = 64   #一层更新分成几个microbatch
    m = N // gradient_accumulation_steps   #每个microbatch的大小
    max_grad_norm = 1.0
    rollout_steps = 50
    n_eval = 1024   #验证集大小

    seed = 0
    rng = random.Random(seed)   #由种子初始化的随机数生成器

    training_file_path = Path("data/gsm8k/train.jsonl")
    validation_file_path = Path("data/gsm8k/test.jsonl")

    model_id = 'allenai/OLMo-2-0425-1B'
    model_device = 'cuda:0'
    server_device = 1
    model, tokenizer = get_model_and_tokenizer(model_id, model_device)

    prompt_dir = Path(__file__).resolve().parent / "prompts"
    zero_shot = (prompt_dir / "r1_zero.prompt").read_text(encoding="utf-8")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )
    server = VLLMServer(model_id, gpu = server_device, gpu_memory_utilization=0.75)
    sampling_params = {
        "temperature": 1.0, 
        "max_tokens": 512, 
        "n": G,     #每个问题生成G个回答
        "seed":seed,
        "stop": '</answer>',
        "include_stop_str_in_output": True
    }
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


    # 以下放在 main() 中，模型、数据、优化器等初始化完成之后。
    with wandb.init(
        project="cs336-grpo",
        name=f"standard-on-policy-seed-{seed}-experiment-{1}",
        config={
            "seed": seed,
            "n_eval": n_eval,
            "rollout_steps": rollout_steps,
            "microbatch_size": 4,
        },
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

        model.train()
        server.start()
        server.init_weight_sync(policy_device=model_device)
        # 训练前验证，得到初始表现。
        log_validation(step=0)
        for k in range(rollout_steps):
            print(f"正在进行第{k}次step")
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
            run.log(
                {
                    "train/loss": batch_loss.item(),
                    "train/grad_norm": metadata['grad_norm'].item(),
                    "train/token_entropy": metadata['token_entropy'],
                    "train/reward": metadata['mean_reward'],
                    "train/format_reward": metadata['mean_format_reward'],
                },
                step=step,
            )
            # 每 10 次更新验证一次，最后一次更新后也验证。
            if step % 10 == 0 or step == rollout_steps:
                log_validation(step)

    server.stop()
    return
if __name__ == '__main__':
    main()