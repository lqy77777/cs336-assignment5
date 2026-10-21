import torch
from transformers import PreTrainedTokenizerBase,PreTrainedModel

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    batch_size = len(prompt_strs)
    max_length = 0
    original_ids = []
    labels = []
    response_mask = []
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

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    result = {}
    batch_size, seq_len = input_ids.shape
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim = -1)
    probs = torch.softmax(logits, dim = -1)
    token_entropy = -(probs * log_probs).sum(dim = -1)
    result['log_probs'] = log_probs
    if return_token_entropy:
        result['token_entropy'] = token_entropy
    return result
