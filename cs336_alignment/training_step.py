from typing import Callable, Literal

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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
    N = len(repeated_prompts)     # N = BG
    G = group_size
    B = N // G
    input_dict = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = input_dict['input_ids']         # N * S
    labels = input_dict['labels']               # N * S
    response_mask = input_dict['response_mask'] # N * S
    log_probs = get_response_log_probs(model, input_ids, labels)['log_probs']   # N * S
    raw_rewards = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)[0]  #长度为N
    normed_advantages = compute_group_normalized_rewards_grpo(raw_rewards,G,baseline,advantage_eps
                                                              ,advantage_normalizer)[0]  #长度为N
    per_loss = compute_policy_gradient_loss_on_policy(normed_advantages,log_probs,importance_reweighting_method,
                                                      old_log_probs,cliprange,response_mask)[0]  # N * (T-1)
    loss = aggregate_loss_across_microbatch_sequence(per_loss,response_mask,loss_normalization,
                                                     normalization_constant)
    microbatch_size = N //gradient_accumulation_steps
    for i in range(0,N, microbatch_size):
        loss = aggregate_loss_across_microbatch_sequence(
            per_loss[i:i+microbatch_size],
            response_mask[i:i+microbatch_size],
            loss_normalization,
            normalization_constant
        )
        loss.backward()
    optimizer.step()
    optimizer.zero_grad()



