"""Profile training_step.main without changing its source file.

This runner changes rollout_steps in memory only, wraps stage boundaries, and
writes local JSONL/CSV measurements. CUDA work on the trainer is synchronized
at measured GPU boundaries; HTTP generation timings include server/network time.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
import csv
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def instrument_main(source: str, filename: str, steps: int):
    """Compile just main(), changing its step count and timing its loop body."""
    tree = ast.parse(source, filename=filename)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assignments = 0
    loops = 0

    class Instrument(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal assignments
            if any(isinstance(t, ast.Name) and t.id == "rollout_steps" for t in node.targets):
                if not isinstance(node.value, ast.Constant):
                    raise ValueError("rollout_steps no longer has a literal default")
                node.value = ast.copy_location(ast.Constant(steps), node.value)
                assignments += 1
            return node

        def visit_For(self, node):
            nonlocal loops
            node = self.generic_visit(node)
            if (isinstance(node.target, ast.Name) and node.target.id == "k"
                    and isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"
                    and len(node.iter.args) == 1
                    and isinstance(node.iter.args[0], ast.Name)
                    and node.iter.args[0].id == "rollout_steps"):
                context = ast.Call(
                    func=ast.Attribute(ast.Name("_stage_recorder", ast.Load()), "iteration", ast.Load()),
                    args=[ast.BinOp(ast.Name("k", ast.Load()), ast.Add(), ast.Constant(1))],
                    keywords=[],
                )
                node.body = [ast.copy_location(ast.With(
                    items=[ast.withitem(context_expr=context)], body=node.body), node)]
                loops += 1
            return node

    main = Instrument().visit(main)
    if (assignments, loops) != (1, 1):
        raise ValueError(f"Unexpected main structure: assignments={assignments}, loops={loops}")
    return compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])), filename, "exec")


class Recorder:
    def __init__(self, output: Path, synchronize=lambda: None):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=False)
        self.events = (output / "events.jsonl").open("w", encoding="utf-8")
        self.synchronize = synchronize
        self.step = None
        self.mode = "initialization"
        self.phase_seconds = defaultdict(Counter)
        self.phase_calls = defaultdict(Counter)
        self.details = defaultdict(Counter)
        self.rows = []
        self.start = time.perf_counter()

    @contextmanager
    def span(self, name, gpu=False):
        if gpu:
            self.synchronize()
        started = time.perf_counter()
        status = "ok"
        try:
            yield
            if gpu:
                self.synchronize()
        except BaseException:
            status = "error"
            raise
        finally:
            seconds = time.perf_counter() - started
            self.phase_seconds[self.step][name] += seconds
            self.phase_calls[self.step][name] += 1
            self.events.write(json.dumps({
                "step": self.step, "phase": name, "seconds": seconds,
                "start_elapsed_seconds": started - self.start, "status": status,
            }) + "\n")

    @contextmanager
    def mode_scope(self, mode):
        before = self.mode
        self.mode = mode
        try:
            yield
        finally:
            self.mode = before

    @contextmanager
    def iteration(self, step):
        self.synchronize()
        self.step = step
        started = time.perf_counter()
        status = "ok"
        try:
            with self.mode_scope("rollout"):
                yield
            self.synchronize()
        except BaseException:
            status = "error"
            raise
        finally:
            elapsed = time.perf_counter() - started
            phases = dict(self.phase_seconds[step])
            top = ("rollout.weight_sync", "rollout.generation", "train.total",
                   "validation.total", "logging.wandb_log")
            row = {"step": step, "status": status, "step_seconds": elapsed,
                   **phases, **dict(self.details[step])}
            row["other_step_seconds"] = elapsed - sum(phases.get(k, 0) for k in top)
            for mode in ("train", "validation"):
                total = phases.get(f"{mode}.total", 0)
                leaves = sum(v for k, v in phases.items()
                             if k.startswith(mode + ".") and k != f"{mode}.total")
                row[f"{mode}.other_seconds"] = total - leaves
            self.rows.append(row)
            self.events.flush()
            self.write_outputs()
            print(f"[timing] step={step} total={elapsed:.3f}s "
                  f"sync={phases.get('rollout.weight_sync', 0):.3f}s "
                  f"generate={phases.get('rollout.generation', 0):.3f}s "
                  f"train={phases.get('train.total', 0):.3f}s "
                  f"validation={phases.get('validation.total', 0):.3f}s", flush=True)
            self.step = None

    def wrapper(self, fn, name, *, gpu=False, mode=None, observe=None):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            # VLLMServer also registers stop() with atexit; that callback can
            # outlive the measurement files after normal or interrupted exit.
            if self.events.closed:
                return fn(*args, **kwargs)
            phase = name() if callable(name) else name
            with ExitStack() as scopes:
                if mode is not None:
                    scopes.enter_context(self.mode_scope(mode))
                scopes.enter_context(self.span(phase, gpu))
                result = fn(*args, **kwargs)
            if observe:
                observe(result, args, kwargs)
            return result
        return wrapped

    def write_outputs(self, **extra):
        columns = ["step", "status", "step_seconds"] + sorted(
            set().union(*(set(r) for r in self.rows)) - {"step", "status", "step_seconds"})
        with (self.output / "steps.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)
        completed = [r for r in self.rows if r["status"] == "ok"]
        phases = sorted(set().union(*(set(r) for r in completed))) if completed else []
        averages = {key: statistics.mean(r.get(key, 0) for r in completed)
                    for key in phases if key not in {"step", "status"}}
        summary = {
            "measurement": "Synchronized trainer stage wall time; generation includes HTTP/server time.",
            "overlap_note": "train.total and validation.total INCLUDE their subphases; do not add them twice.",
            "completed_steps": len(completed), "step_means": averages,
            "outside_steps_phase_seconds": dict(self.phase_seconds[None]),
            "phase_calls_by_step": {str(k): dict(v) for k, v in self.phase_calls.items()},
            **extra,
        }
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate instrumentation without training")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    source_path = ROOT / "cs336_alignment/training_step.py"
    source = source_path.read_text(encoding="utf-8")
    code = instrument_main(source, str(source_path), args.steps)
    if args.dry_run:
        print(f"Instrumentation valid: {args.steps} steps; source file unchanged; no training started.")
        return
    if Path.cwd().resolve() != ROOT:
        parser.error(f"Run from project root: {ROOT}")
    bypass = [part.strip() for key in ("NO_PROXY", "no_proxy")
              for part in os.environ.get(key, "").split(",") if part.strip()]
    bypass = list(dict.fromkeys([*bypass, "127.0.0.1", "localhost", "::1"]))
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(bypass)
    sys.path.insert(0, str(ROOT))
    import torch
    import wandb
    from wandb.sdk.wandb_run import Run
    target = importlib.import_module("cs336_alignment.training_step")
    from cs336_alignment.vllm_utils import VLLMServer

    originals = [ROOT / "cs336_alignment" / n for n in (
        "training_step.py", "tool.py", "vllm_utils.py", "checkpoint.py", "tuning_prompt_ablation.py")]
    def hashes():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    before = hashes()
    output = args.output_dir or ROOT / "results" / "stage_timing" / datetime.now().strftime("%Y%m%d_%H%M%S")
    def synchronize():
        if torch.cuda.is_initialized():
            torch.cuda.synchronize(0)
    recorder = Recorder(output, synchronize)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "steps": args.steps,
        "source_sha256_before": before,
        "changes_in_memory_only": ["rollout_steps", "stage wrappers", "W&B project/name/config labels"],
        "loopback_proxy_bypass": True,
        "limitations": ["Synchronization perturbs overlap/timing.",
                        "Short run measures early training, not a replay of the finished run.",
                        "Original final-step validation is preserved (steps 0, 10, 12 for a 12-step run)."],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def observe_generation(completions, _args, _kwargs):
        d = recorder.details[recorder.step]
        d[f"{recorder.mode}.responses"] += len(completions)
        d[f"{recorder.mode}.response_tokens"] += sum(len(c.token_ids or []) for c in completions)

    def observe_advantages(result, _args, _kwargs):
        advantages = result[0]
        d = recorder.details[recorder.step]
        d["train.samples"] += advantages.numel()
        d["train.nonzero_advantage_samples"] += int((advantages != 0).sum().item())

    def observe_forward(_result, args, _kwargs):
        recorder.details[recorder.step]["train.forward_microbatches"] += 1

    original_init = wandb.init
    def timed_init(*a, **kw):
        kw["project"] = "cs336-grpo-stage-timing"
        kw["name"] = f"stage-timing-{args.steps}-steps"
        kw["config"] = {**kw.get("config", {}), "stage_timing": True, "synchronize_trainer": True}
        with recorder.span("initialization.wandb_init"):
            run = original_init(*a, **kw)
        metadata["wandb_run_id"] = run.id
        metadata["wandb_run_dir"] = run.dir
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return run

    status = "error"
    try:
        with ExitStack() as patches:
            def wrap(obj, attribute, phase, **options):
                patches.enter_context(patch.object(obj, attribute, recorder.wrapper(
                    getattr(obj, attribute), phase, **options)))
            patches.enter_context(patch.object(wandb, "init", timed_init))
            wrap(target, "get_model_and_tokenizer", "initialization.model_load", gpu=True)
            wrap(VLLMServer, "start", "initialization.server_start")
            wrap(VLLMServer, "init_weight_sync", "initialization.weight_sync_init", gpu=True)
            wrap(VLLMServer, "stop", "shutdown.server_stop")
            wrap(VLLMServer, "sync_policy_weights", lambda: f"{recorder.mode}.weight_sync", gpu=True)
            wrap(VLLMServer, "generate_completions", lambda: f"{recorder.mode}.generation", observe=observe_generation)
            wrap(target, "evaluate_policy", "validation.total", mode="validation", gpu=True)
            wrap(target, "grpo_train_step", "train.total", mode="train", gpu=True)
            wrap(target, "tokenize_prompt_and_output", "train.tokenize")
            wrap(target, "compute_rollout_rewards", lambda: f"{recorder.mode}.reward")
            wrap(target, "compute_group_normalized_rewards_grpo", "train.advantages", observe=observe_advantages)
            wrap(target, "get_response_log_probs", "train.forward_logprobs_entropy", gpu=True, observe=observe_forward)
            wrap(target, "compute_policy_gradient_loss_on_policy", "train.loss", gpu=True)
            wrap(target, "aggregate_loss_across_microbatch_sequence", "train.loss", gpu=True)
            wrap(torch.Tensor, "backward", "train.backward", gpu=True)
            wrap(target, "clip_grad_norm_", "train.clip_grad", gpu=True)
            wrap(torch.optim.AdamW, "step", "train.optimizer", gpu=True)
            wrap(torch.optim.AdamW, "zero_grad", "train.zero_grad", gpu=True)
            wrap(Run, "log", "logging.wandb_log")
            patches.enter_context(patch.object(target, "_stage_recorder", recorder, create=True))
            original_main = target.main
            try:
                exec(code, target.__dict__)
                target.main()
            finally:
                target.main = original_main
        status = "completed"
    finally:
        after = hashes()
        unchanged = before == after
        recorder.write_outputs(status=status, source_files_unchanged=unchanged,
                               total_runner_seconds=time.perf_counter() - recorder.start)
        recorder.events.close()
        metadata["source_sha256_after"] = after
        metadata["source_files_unchanged"] = unchanged
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[timing] status={status}; original_files_unchanged={unchanged}; output={output}", flush=True)


if __name__ == "__main__":
    main()
