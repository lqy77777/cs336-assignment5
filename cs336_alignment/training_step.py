from typing import Callable, Literal
from pathlib import Path
import json
import random
import wandb

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

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
    device = next(model.parameters()).device       #确定设备
    batch_loss = torch.zeros((), device=device)
    metadata = {}
    N = len(repeated_prompts)     # N = B * G
    G = group_size

    input_dict = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = input_dict['input_ids'].to(device)         # N * S
    labels = input_dict['labels'].to(device)               # N * S
    response_mask = input_dict['response_mask'].to(device) # N * S

    raw_rewards, reward_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)  #长度为N
    metadata['mean_reward'] = float(reward_metadata['mean_reward'])  #将Tensor标量转化为浮点数
    metadata['mean_format_reward'] = float(reward_metadata['mean_format_reward'])
    metadata['std_reward'] = float(reward_metadata['std_reward'])
    normed_advantages = compute_group_normalized_rewards_grpo(raw_rewards,G,baseline,advantage_eps
                                                              ,advantage_normalizer)[0].to(device)  #长度为N
    m = N //gradient_accumulation_steps   #microbatch_size
    entropy_sum = torch.zeros((), device=device)
    #由于改进，需要计算有效token数
    entropy_token_count =torch.zeros((),device = device)
    
    for i in range(0,N, m):
        micro_advantages = normed_advantages[i:i+m]
        original_size = micro_advantages.numel()
        keep = micro_advantages != 0
        # 当前 microbatch 没有有效贡献，省掉模型计算
        if not keep.any():
            continue
        micro_input_ids = input_ids[i:i+m][keep]
        micro_labels = labels[i:i+m][keep]
        micro_mask = response_mask[i:i+m][keep]
        micro_advantages = micro_advantages[keep]
        response_outputs = get_response_log_probs(
            model, micro_input_ids, micro_labels, return_token_entropy=True,
        )
        log_probs = response_outputs['log_probs']   # m * S
        # 按整轮回答 token 数加权，排除 prompt 和 padding。
        entropy_sum += (response_outputs['token_entropy'] * micro_mask).sum()
        entropy_token_count += micro_mask.sum()
        per_loss = compute_policy_gradient_loss_on_policy(micro_advantages,log_probs,importance_reweighting_method,
                                                old_log_probs,cliprange,response_mask=micro_mask)[0]  # m * (T-1)
        loss = aggregate_loss_across_microbatch_sequence(per_loss, micro_mask, loss_normalization,
                                                         normalization_constant)
        #接下来这个if-else非常重要
        if loss_normalization == 'sequence':
            loss = loss * (micro_advantages.numel() / original_size)
            loss = loss / gradient_accumulation_steps
        batch_loss += loss.detach()   #记录用的，不参与反向传播
        loss.backward()
     #所有有效回答token的平均熵
    metadata['mean_entropy'] = (entropy_sum / entropy_token_count).item() if entropy_token_count.item() > 0 else 0.0
    #防止所有normed_advantaged都等于0的特殊情况
    if not (normed_advantages != 0).any():
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue

            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            else:
                parameter.grad.zero_()
    #梯度裁剪
    clip_limit = max_grad_norm if max_grad_norm is not None else float('inf')
    grad_norm = clip_grad_norm_(model.parameters(), max_norm = clip_limit)
    metadata['grad_norm'] = grad_norm
    metadata['batch_loss'] = batch_loss
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss, metadata

def evaluate_policy(
    model: PreTrainedModel,
    server: VLLMServer,
    tokenizer: PreTrainedTokenizerBase,
    validation_data: list[dict[str, str]],
    prompt_template: str,                 #提问模板
    sampling_params: dict[str, float | int | str | bool],
    reward_fn: Callable[[str, str], dict[str, float]],
    n_eval: int = 1024,
):
    """用来评估“当前模型答题表现如何”：让模型回答固定的验证集中的问题，给回答打分，
    统计指标，并保存几条回答供你查看。它不会对训练模型执行梯度更新。"""
    # 每次验证使用相同的数据，方便比较训练前后的变化。
    validations = validation_data[:n_eval]

    prompts = [prompt_template.format(question=validation["question"]) for validation in validations]
    ground_truths = [validation["ground_truth"] for validation in validations]

    # 同步当前模型权重
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
    #整理指标
    metrics = {
        "validation_reward": float(reward_metadata["mean_reward"]),
        "validation_format_reward": float(
            reward_metadata["mean_format_reward"]
        ),
        "validation_response_length": sum(lengths) / len(lengths),
        'validation_std_reward': float(reward_metadata['std_reward'])
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
    gradient_accumulation_steps = 32   #一层更新分成几个microbatch
    m = N // gradient_accumulation_steps   #每个microbatch的大小
    max_grad_norm = 1.0
    rollout_steps = 200
    n_eval = 1024   #验证集大小
    n_train = B * rollout_steps

    seed = 0
    rng = random.Random(seed)   #由种子初始化的随机数生成器
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
    model, tokenizer = get_model_and_tokenizer(model_id, model_device)

    prompt_dir = Path(__file__).resolve().parent / "prompts"
    zero_shot = (prompt_dir / "r1_zero.prompt").read_text(encoding="utf-8")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )

    server = VLLMServer(model_id, gpu = server_device)

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

    final_reward = 0
    # 以下放在 main() 中，模型、数据、优化器等初始化完成之后。
    with wandb.init(
        project="cs336-grpo",
        name=f"standard-on-policy-seed-{seed}-experiment-{1}",
        config={
            "seed": seed,
            "n_eval": n_eval,
            "rollout_steps": rollout_steps,
            "microbatch_size": m,
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
                    print(f'final validation reward is {final_reward}')
        finally:
            server.stop()
    return
if __name__ == '__main__':
    main()