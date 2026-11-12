import re
from typing import Any
from pathlib import Path
import json
from cs336_alignment.vllm_utils import VLLMServer
import time    #统计吞吐量
from decimal import Decimal


def loading_alpacaeval(
    dir_path: str
) -> list[dict[str,Any]]:
    test_dir = Path(dir_path)
    data_path = test_dir / "alpaca_eval_gpt4_turbo.json"
    with data_path.open("r", encoding = 'utf-8') as f:
        examples = json.load(f)

    return examples

def format_prompts_alpacaeval(
    examples: list[dict[str,Any]],
) -> list[str]:
    prompts = []
    baseline_outputs = []
    prompt_dir = Path("cs336_alignment/prompts_safety")
    zero_shot_system_prompt = (prompt_dir / "zero_shot_system_prompt.prompt").read_text(encoding = "utf-8")
    alpacaeval_zero_shot_prompt = (prompt_dir/"alpaca_eval_zero_shot.prompt").read_text(encoding= "utf-8")
    for example in examples:
        instruction = alpacaeval_zero_shot_prompt.format(instruction = example["instruction"])
        prompt = zero_shot_system_prompt.format(instruction = instruction)
        prompts.append(prompt)
        example["prompt"] = prompt
        baseline_outputs.append(example["output"])
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
    output_path = Path("results/alpacaeval_output.json")
    examples = loading_alpacaeval("data/alpaca_eval")
    prompts = format_prompts_alpacaeval(examples)
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
                "instruction": example["instruction"],
                "output": model_output,
                "generator":"llama-3.1-8b-base",
                "dataset": example["dataset"],
            })
        metrics = {
            "generation_seconds": generation_seconds,
            "examples_per_second": len(completions) / generation_seconds,
        }
        print(metrics)
        with output_path.open("w",encoding="utf-8") as f:
            json.dump(results,f)
    finally:
        server.stop()
if __name__ == '__main__':
    main()