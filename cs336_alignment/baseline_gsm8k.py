import re
from typing import Any
from pathlib import Path
import json
from cs336_alignment.vllm_utils import VLLMServer
import time    #统计吞吐量
from decimal import Decimal

def parse_gsm8k_response(
    model_output: str
) -> str | None:
    #找出模型输出中的所有数字，取最后一个；如果没有数字，就返回 None
    pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
    matches = re.findall(pattern, model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")

def loading_gsm8k(
    dir_path: str
) -> list[dict[str,Any]]:
    test_dir = Path(dir_path)
    data_path = test_dir / "test.jsonl"
    examples = []
    with data_path.open("r", encoding = 'utf-8') as f:
        for line in f:
            example = json.loads(line)
            question = example['question']
            answer = example['answer']

            ground_truth = answer.split('####', 1)[1].strip().replace(",", "")
            examples.append({'question':question,'answer':answer,'ground_truth':ground_truth})

    return examples

def format_prompts_gsm8k(
    examples: list[dict[str,Any]],
) -> list[str]:
    prompts = []
    prompt_dir = Path("cs336_alignment/prompts_safety")
    zero_shot_system_prompt = (prompt_dir / "zero_shot_system_prompt.prompt").read_text(encoding = "utf-8")
    gsm8k_zero_shot_prompt = (prompt_dir/"gsm8k_zero_shot.prompt").read_text(encoding= "utf-8")
    for example in examples:
        instruction = gsm8k_zero_shot_prompt.format(question = example["question"])
        prompt = zero_shot_system_prompt.format(instruction = instruction)
        prompts.append(prompt)
        example["prompt"] = prompt
    return prompts

server = VLLMServer(
    model_id = "meta-llama/Llama-3.1-8B",
    gpu = 0,
)

sampling_params = {
    "temperature": 0.0,
    "max_tokens": 512,
    "top_p": 1.0,
    "n": 1,
    "seed":0,
    "stop": ["# Query:"],
    "include_stop_str_in_output": False
}

def main():
    output_path = Path("results/gsm8k_baseline.json")
    examples = loading_gsm8k("data/gsm8k")
    prompts = format_prompts_gsm8k(examples)
    try:
        server.start()
        start_time = time.perf_counter()
        metrics = {}
        results = []
        completions = server.generate_completions(
            prompts=prompts,
            sampling_params = sampling_params,
            batch_size = 128
        )
        generation_seconds = time.perf_counter() - start_time

        for example, completion in zip(examples, completions):
            model_output = completion.text
            prediction = parse_gsm8k_response(
                model_output=model_output
            )
            parse_success = prediction is not None
            score = int(
                prediction is not None
                and Decimal(prediction) == Decimal(example["ground_truth"])
            )
            results.append({
                **example,
                "model_output": model_output,
                "prediction": prediction,
                "parse_success": parse_success,
                "score": score,
                "finish_reason": completion.finish_reason,
            })
        num_examples = len(results)
        num_correct = sum(result["score"] for result in results)
        num_parse_failed = sum(
            not result["parse_success"] for result in results
        )
        metrics = {
            "generation_seconds": generation_seconds,
            "examples_per_second": num_examples / generation_seconds,
            "num_examples": num_examples,
            "num_correct": num_correct,
            "accuracy": num_correct / num_examples,
            "num_parse_failed": num_parse_failed,
            "parse_failure_rate": num_parse_failed / num_examples,

        }
        payload = {
            "model_id": server.model_id,
            "data_split": "test",
            "sampling_params": sampling_params,
            "metrics": metrics,
           "results": results,
        }
        with output_path.open("w",encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent =2)
    finally:
        server.stop()
if __name__ == '__main__':
    main()