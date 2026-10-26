import torch
from transformers import PreTrainedTokenizerBase,PreTrainedModel
from typing import Callable,Literal
from einops import einsum, rearrange

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """将输入文本转化为token id序列"""
    #batch_size = B*G   因为这里面prompt_strs已经包含重复的问题
    batch_size = len(prompt_strs)
    max_length = 0
    original_ids = []
    labels = []
    response_mask = []   #回答是1，问题和padding是0
    for i in range(batch_size):
        prompt_ids = tokenizer.encode(prompt_strs[i],add_special_tokens = False)
        output_ids = tokenizer.encode(output_strs[i], add_special_tokens= False)
        ids = prompt_ids + output_ids
        original_ids.append(ids)

        length = len(ids)
        prompt_length = len(prompt_ids)
        mask = [0] * (prompt_length - 1) + [1] * (length - prompt_length)
        response_mask.append(mask)
        if length > max_length:
            max_length = length
    for i in range(batch_size):
        length = max_length - len(original_ids[i])
        if length > 0:
            original_ids[i] += [tokenizer.pad_token_id] * length
            response_mask[i] += [0] * length
    original_ids = torch.tensor(original_ids,dtype = torch.long)
    input_ids = original_ids[:,:-1]
    labels = original_ids[:,1:]
    response_mask = torch.tensor(response_mask, dtype = torch.long)
    return {"input_ids":input_ids,
            "labels": labels,
            "response_mask": response_mask}
    #这三个东西的大小都是  BG*(L-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,    #BG*(L-1)
    labels: torch.Tensor,       #BG*(L-1)
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    result = {}
    logits = model(input_ids).logits
    log_probs_all = torch.log_softmax(logits, dim = -1)   #维度是b*s*v
    log_probs = log_probs_all.gather(dim=-1,index=labels.unsqueeze(-1),).squeeze(-1)
    # log_probs_all        :BG * (T-1) * V
    # labels:               BG * (T-1)
    # labels.unsqueeze(-1) :BG * (T-1) * 1
    # gather后             :BG * (T-1) * 1
    # squeeze(-1)后        :BG * (T-1)
    result['log_probs'] = log_probs
    if return_token_entropy:
        probs = torch.softmax(logits, dim = -1)
        token_entropy = -(probs * log_probs_all).sum(dim = -1)
        result['token_entropy'] = token_entropy
    return result

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_response: list[str],
    repeated_ground_truths:list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """模型已经生成了一批回答，现在要给每条回答打分，把这些分数整理成后续训练能使用的形式，同时记录这一批回答的整体表现。"""
    rewards = []
    format_rewards = []
    for response, ground_truth in zip(rollout_response, repeated_ground_truths):
        scores = reward_fn(response, ground_truth)
        rewards.append(scores['reward'])
        format_rewards.append(scores['format_reward'])
    rewards = torch.tensor(rewards)
    format_rewards = torch.tensor(format_rewards)
    metadata = {'mean_reward': rewards.mean(),
                'mean_format_reward': format_rewards.mean()}
    return rewards, metadata

def compute_group_normalized_rewards_grpo(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal['mean','none'] = 'mean',
    advantage_eps: float=1e-6,
    advantage_normalizer: Literal['std','none','mean'] = 'std',
):
    """求advantage"""
    n = len(raw_rewards) // group_size
    mu = []
    norm = []
    for i in range(n):
        reward = raw_rewards[(i*n):(i*n+group_size)]
        if baseline == 'mean':
            mu += [reward.mean().item()] * group_size
        else:
            pass
        if advantage_normalizer == 'std':
            norm += [reward.std().item() + advantage_eps] * group_size
        else:
            pass
    mu = torch.tensor(mu)
    norm = torch.tensor(norm)
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }
    rewards = (raw_rewards - mu) / norm
    return rewards, metadata

def compute_policy_gradient_loss_on_policy(
    raw_rewards_or_advantages: torch.Tensor,   # N或者 N * 1
    policy_log_probs: torch.Tensor,    #N * (T-1)
    importance_reweighting_method: Literal['none','noclip','grpo','gspo'] = 'none',
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """把每条回答的 reward 或 advantage，乘到这条回答每个 token 的 log probability 上，得到逐 token 的策略梯度 loss。"""
    metadata = {}
    advantages = rearrange(raw_rewards_or_advantages,'... -> (...)')    #转化为长度为N的向量
    result = - einsum(advantages, policy_log_probs,'b,b t -> b t')  #负号不要漏了
    return result, metadata
    #result大小  N*(T-1)

def aggregate_loss_across_microbatch_sequence(
    per_token_policy_gradient_loss: torch.Tensor,  #N * （T-1）
    mask: torch.Tensor,
    loss_normalization: Literal['sequence', 'constant'] = 'sequence',
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """接收上一题算出的逐 token loss，排除题目和 padding 位置，按规则平均，最终得到一个可以调用 backward() 的标量 loss。"""
    # 排除 prompt 和 padding 位置
    masked_loss = per_token_policy_gradient_loss * mask   # N * (T-1)

    # 每条回答的有效 token loss 总和，形状 (batch_size,)
    loss_sum = masked_loss.sum(dim=-1)                    #N

    # 每条回答的有效 token 数量，形状 (batch_size,)
    token_count = mask.sum(dim=-1)   #即为len              #N

    # 每条回答内部取平均
    sequence_loss = loss_sum / token_count                #N

    # 对本批所有回答取平均，返回标量 Tensor
    return sequence_loss.mean()