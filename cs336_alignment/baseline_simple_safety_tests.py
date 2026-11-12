import re
from typing import Any
from pathlib import Path
import json
from cs336_alignment.vllm_utils import VLLMServer
import time    #统计吞吐量
from decimal import Decimal
import csv

def loading_simple_safe_tytests(
    dir_path: str
) -> list[dict[str,Any]]:
    test_dir = Path(dir_path)
    data_path = test_dir / "simple_safety_tests.csv"
    examples = []
    with data_path.open("r", encoding = 'utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)   #跳过第一行表头
        for row in reader:
            example = {
                "harm_area" : row[1],
                "category" : row[3],
                "prompts_final" : row[4],            
            }
            examples.append(example)
    return examples

def format_prompts_simple_safe_tytests(
    examples: list[dict[str,Any]],
) -> list[str]:
    prompts = []
    prompt_dir = Path("cs336_alignment/prompts_safety")
    zero_shot_system_prompt = (prompt_dir / "zero_shot_system_prompt.prompt").read_text(encoding = "utf-8")
    simple_safe_tytests_zero_shot_prompt = (prompt_dir/"simple_safety_tests_zero_shot.prompt").read_text(encoding= "utf-8")
    for example in examples:
        instruction = simple_safe_tytests_zero_shot_prompt.format(instruction = example["prompts_final"])
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
    output_path = Path("results/simple_safe_tytests_output.jsonl")
    examples = loading_simple_safe_tytests("data/simple_safety_tests")
    prompts = format_prompts_simple_safe_tytests(examples)
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
            
            results.append({
                "prompts_final": example["prompts_final"],
                "output": model_output,
            })
        metrics = {
            "generation_seconds": generation_seconds,
            "examples_per_second": len(completions) / generation_seconds,
        }
        print(metrics)
        with output_path.open("w",encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        server.stop()
if __name__ == '__main__':
    main()