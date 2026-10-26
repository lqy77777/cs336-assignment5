from typing import Callable, Literal
from pathlib import Path
import json
import random

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
    batch_loss = torch.zeros((), device=next(model.parameters()).device)
    metadata = {}
    N = len(repeated_prompts)     # N = BG
    G = group_size
    B = N // G
    input_dict = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = input_dict['input_ids']         # N * S
    labels = input_dict['labels']               # N * S
    response_mask = input_dict['response_mask'] # N * S
    raw_rewards = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)[0]  #长度为N
    normed_advantages = compute_group_normalized_rewards_grpo(raw_rewards,G,baseline,advantage_eps
                                                              ,advantage_normalizer)[0]  #长度为N
    m = N //gradient_accumulation_steps   #microbatch_size
    
    for i in range(0,N, m):
        log_probs = get_response_log_probs(model, input_ids[i:i+m], labels[i:i+m])['log_probs']   # m * S
        per_loss = compute_policy_gradient_loss_on_policy(normed_advantages[i:i+m],log_probs,importance_reweighting_method,
                                                old_log_probs,cliprange,response_mask[i:i+m])[0]  # m * (T-1)
        loss = aggregate_loss_across_microbatch_sequence(per_loss, response_mask[i:i+m], loss_normalization,
                                                         normalization_constant)
        batch_loss += (loss/gradient_accumulation_steps).detach()
        (loss/gradient_accumulation_steps).backward()
    #梯度裁剪
    clip_limit = max_grad_norm if max_grad_norm is not None else float('inf')
    grad_norm = clip_grad_norm_(model.parameters(), max_norm = clip_limit)
    metadata['grad_norm'] = grad_norm
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss, metadata


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
    rollout_steps = 50

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
    server = VLLMServer(model_id, gpu = server_device)
    sampling_params = {
        "temperature": 1.0, 
        "max_tokens": 512, 
        "n": G,     #每个问题生成G个回答
        "seed":seed,
        "stop": '</answer>',
        "include_stop_str_in_output": True
    }
    training_data = []
    with training_file_path.open("r", encoding = 'utf-8') as f:
        for line in f:
            example = json.loads(line)
            question = example['question']
            answer = example['answer']
            ground_truth = answer.split('####', 1)[1].strip()
            training_data.append({'question':question,'ground_truth':ground_truth})
    validation_data = []
    with validation_file_path.open("r", encoding = 'utf-8') as f:
        for line in f:
            example = json.loads(line)
            question = example['question']
            answer = example['answer']
            ground_truth = answer.split('####', 1)[1].strip()
            validation_data.append({'question':question,'ground_truth':ground_truth})

    model.train()
    server.start()
    server.init_weight_sync(policy_device=model_device)
    for k in range(rollout_steps):
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
    return
if __name__ == '__main__':
    main()