"""Measure prompt batching with the existing vLLM helpers, without training.

The original source files are hashed before/after. All cases use the same 32
prompts, n=8, temperature=1, seed=0, and max_tokens=512. Prefix caches are reset
before every case. Each batch size gets a warmup and three measured repeats.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import statistics
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZES = (1, 4, 8, 16, 32)


def metric_sum(text: str, name: str):
    expression = re.compile(r"^" + re.escape(name) + r"(?:\{.*\})?\s+([0-9.eE+\-]+)(?:\s+\d+)?$")
    values = [float(m.group(1)) for line in text.splitlines() if (m := expression.match(line))]
    return sum(values) if values else None


def describe_outputs(completions, expected=256):
    if len(completions) != expected:
        raise ValueError(f"Expected {expected} completions, received {len(completions)}")
    if any(not c.token_ids for c in completions):
        raise ValueError("Missing/empty token IDs; do not silently undercount throughput")
    lengths = sorted(len(c.token_ids) for c in completions)
    finishes = Counter(c.finish_reason for c in completions)
    token_digest = hashlib.sha256(json.dumps([c.token_ids for c in completions]).encode()).hexdigest()
    return {
        "responses": len(completions), "output_tokens": sum(lengths),
        "mean_tokens_per_response": statistics.mean(lengths),
        "p50_tokens": statistics.median(lengths),
        "p90_tokens": lengths[math.ceil(0.9 * len(lengths)) - 1],
        "max_tokens_observed": max(lengths),
        "length_finished_responses": finishes.get("length", 0),
        "stop_finished_responses": finishes.get("stop", 0),
        "other_finish_reasons": json.dumps({str(k): v for k, v in finishes.items() if k not in {"stop", "length"}}),
        "token_ids_sha256": token_digest,
    }


def write_tables(output: Path, trials: list[dict]):
    if not trials:
        return []
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
    groups = []
    for size in BATCH_SIZES:
        rows = [r for r in trials if r["batch_size"] == size and not r["warmup"]]
        if not rows:
            continue
        seconds = [r["seconds"] for r in rows]
        counts = [r["preemptions_delta"] for r in rows]
        groups.append({
            "batch_size": size, "candidates_per_http_batch": size * 8,
            "http_requests_per_trial": 32 // size, "repeats": len(rows),
            "mean_seconds": statistics.mean(seconds), "median_seconds": statistics.median(seconds),
            "min_seconds": min(seconds), "max_seconds": max(seconds),
            "stdev_seconds": statistics.stdev(seconds) if len(seconds) > 1 else None,
            "mean_output_tokens": statistics.mean(r["output_tokens"] for r in rows),
            "aggregate_output_tokens_per_second": sum(r["output_tokens"] for r in rows) / sum(seconds),
            "length_finished_responses": sum(r["length_finished_responses"] for r in rows),
            "preemptions_total": sum(counts) if all(v is not None for v in counts) else None,
        })
    baseline = next((g["median_seconds"] for g in groups if g["batch_size"] == 1), None)
    for group in groups:
        group["median_speedup_vs_batch_1"] = baseline / group["median_seconds"] if baseline else None
    if groups:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(groups[0]))
            writer.writeheader()
            writer.writerows(groups)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if Path.cwd().resolve() != ROOT:
        parser.error(f"Run from {ROOT}")
    bypass = [s.strip() for key in ("NO_PROXY", "no_proxy")
              for s in os.environ.get(key, "").split(",") if s.strip()]
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(dict.fromkeys(
        [*bypass, "127.0.0.1", "localhost", "::1"]))
    sys.path.insert(0, str(ROOT))
    from cs336_alignment.vllm_utils import VLLMServer, start_server, wait_for_server, _http_json

    output = args.output_dir or ROOT / "results" / "generation_batch_benchmark" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    protected = sorted((ROOT / "cs336_alignment").rglob("*.py"))
    protected += sorted((ROOT / "cs336_alignment/prompts").glob("*.prompt"))
    def hashes():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    before = hashes()
    data = [json.loads(line) for line in (ROOT / "data/gsm8k/train.jsonl").read_text().splitlines() if line.strip()]
    indices = random.Random(0).sample(range(len(data)), 32)
    template = (ROOT / "cs336_alignment/prompts/r1_zero.prompt").read_text()
    prompts = [template.format(question=data[i]["question"]) for i in indices]
    (output / "prompts.json").write_text(json.dumps([
        {"prompt_index": j, "dataset_row": i, "prompt": prompts[j]}
        for j, i in enumerate(indices)], ensure_ascii=False, indent=2), encoding="utf-8")
    parameters = {"temperature": 1.0, "max_tokens": 512, "n": 8, "seed": 0,
                  "stop": "</answer>", "include_stop_str_in_output": True}
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    server = VLLMServer("allenai/OLMo-2-0425-1B", port=port, gpu=1)
    order_rng = random.Random(20261106)
    schedules = []
    for _ in range(args.repeats):
        order = list(BATCH_SIZES)
        order_rng.shuffle(order)
        schedules.append(order)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "model": server.model_id,
        "gpu": 1, "port": port, "gpu_memory_utilization": 0.9,
        "sampling_parameters": parameters, "num_prompts": 32,
        "batch_sizes": BATCH_SIZES, "repeats": args.repeats,
        "measured_case_order": schedules, "prefix_cache_reset_before_each_case": True,
        "source_sha256_before": before,
        "versions": {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")},
        "limitations": ["Base model generation only; no training or weight synchronization.",
                        "Cases may generate different tokens despite a fixed seed; actual output lengths are recorded.",
                        "Throughput is end-to-end output tokens/s, not a pure kernel benchmark."],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    trials = []
    raw = (output / "responses.jsonl").open("w", encoding="utf-8")
    metric_names = ("vllm:num_preemptions_total", "vllm:generation_tokens_total")
    def metrics():
        with urllib.request.urlopen(server.base_url + "/metrics", timeout=10) as response:
            text = response.read().decode()
        return {name: metric_sum(text, name) for name in metric_names}

    def run_case(size, repeat, warmup):
        reset = _http_json("POST", server.base_url + "/reset_prefix_cache", timeout=30)
        if reset is False:
            raise RuntimeError("Prefix-cache reset failed")
        counters_before = metrics()
        started = time.perf_counter()
        completions = server.generate_completions(prompts, parameters, batch_size=size)
        seconds = time.perf_counter() - started
        counters_after = metrics()
        left = counters_before[metric_names[0]]
        right = counters_after[metric_names[0]]
        row = {
            "trial": len(trials) + 1, "warmup": warmup, "repeat": repeat,
            "batch_size": size, "seconds": seconds, **describe_outputs(completions),
            "preemptions_delta": right - left if left is not None and right is not None else None,
            "metrics_before": json.dumps(counters_before), "metrics_after": json.dumps(counters_after),
        }
        row["output_tokens_per_second"] = row["output_tokens"] / seconds
        trials.append(row)
        for j, completion in enumerate(completions):
            raw.write(json.dumps({"trial": row["trial"], "batch_size": size,
                "repeat": repeat, "warmup": warmup, "prompt_index": j // 8,
                "sample_index": j % 8, "text": completion.text,
                "token_ids": completion.token_ids, "finish_reason": completion.finish_reason}, ensure_ascii=False) + "\n")
        raw.flush()
        write_tables(output, trials)
        print(f"[benchmark] warmup={warmup} repeat={repeat} batch={size} "
              f"seconds={seconds:.3f} tokens={row['output_tokens']} "
              f"tokens/s={row['output_tokens_per_second']:.1f} "
              f"length_stops={row['length_finished_responses']} preemptions={row['preemptions_delta']}", flush=True)

    status = "error"
    started = time.perf_counter()
    try:
        # Start only our own server on an unused port; never call the helper
        # that terminates existing vLLM processes on a shared/default port.
        server.process = start_server(model_id=server.model_id, host=server.host,
            port=server.port, gpu=server.gpu, seed=server.seed,
            load_format=server.load_format, logging_level=server.logging_level,
            gpu_memory_utilization=server.gpu_memory_utilization)
        wait_for_server(server.base_url, server.process, server.startup_timeout)
        metadata["startup_seconds"] = time.perf_counter() - started
        with urllib.request.urlopen(server.base_url + "/metrics", timeout=10) as response:
            (output / "metrics_initial.prom").write_bytes(response.read())
        for size in BATCH_SIZES:
            run_case(size, 0, True)
        for repeat, order in enumerate(schedules, 1):
            for size in order:
                run_case(size, repeat, False)
        with urllib.request.urlopen(server.base_url + "/metrics", timeout=10) as response:
            (output / "metrics_final.prom").write_bytes(response.read())
        status = "completed"
    finally:
        try:
            server.stop()
        finally:
            raw.close()
            after = hashes()
            metadata.update(status=status, source_sha256_after=after,
                            source_files_unchanged=before == after,
                            total_seconds=time.perf_counter() - started)
            groups = write_tables(output, trials)
            (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            (output / "summary.json").write_text(json.dumps({"metadata": metadata, "results": groups}, indent=2), encoding="utf-8")
            print(f"[benchmark] status={status}; original_files_unchanged={before == after}; output={output}", flush=True)


if __name__ == "__main__":
    main()
