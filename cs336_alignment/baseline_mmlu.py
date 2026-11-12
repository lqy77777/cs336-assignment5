import re
from typing import Any
from pathlib import Path
import csv
import json
from vllm_utils import VLLMServer
import time    #统计吞吐量
def parse_mmlu_response(
    mmlu_example: dict[str,Any],
    model_output: str
) -> str| None:
    pattern = r"\bThe\s+correct\s+answer\s+is\s+([ABCD])\b"
    match = re.search(pattern, model_output, flags=re.IGNORECASE)
    if match is not None:
        return (match.group(1)).upper()
    else:
        return None

def loading_mmlu(
    dir_path: str
) -> list[dict[str,Any]]:
    test_dir = Path(dir_path)
    examples = []
    for csv_path in sorted(test_dir.glob("*.csv")):  #排序是为了是为了让遍历顺序固定，方便检查和复现实验
        subject = csv_path.stem.removesuffix("_test").replace("_"," ")   #从文件名中获取学科名
        with csv_path.open(encoding="utf-8",newline = "") as f:
            reader = csv.reader(f)
            for row in reader:
                example = {
                    "subject" : subject,
                    "question" : row[0],
                    "options" : row[1:5],
                    "answer": row[5]
                }
                examples.append(example)
    return examples

def format_prompts_mmlu(
    examples: list[dict[str,Any]],
) -> list[str]:
    prompts = []
    prompt_dir = Path("cs336_alignment/prompts_safety")
    zero_shot_system_prompt = (prompt_dir / "zero_shot_system_prompt.prompt").read_text(encoding = "utf-8")
    mmlu_zero_shot_prompt = (prompt_dir/"mmlu_zero_shot.prompt").read_text(encoding= "utf-8")
    for example in examples:
        instruction = mmlu_zero_shot_prompt.format(subject = example["subject"],question = example["question"],
                                              options = example["options"])
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
    output_path = Path("results/mmlu_baseline.json")
    examples = loading_mmlu("data/mmlu/test")
    prompts = format_prompts_mmlu(examples)
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
            prediction = parse_mmlu_response(
                mmlu_example=example,
                model_output=model_output
            )
            parse_success = prediction is not None
            score = int(prediction == example["answer"])
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