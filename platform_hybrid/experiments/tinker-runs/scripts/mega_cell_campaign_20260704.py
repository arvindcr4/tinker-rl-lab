#!/usr/bin/env python3
"""500+ cell sampling-only Tinker measurement campaign.

Cell = (model, task_slice, G, temperature, seed). Each completed cell writes:
  - one row to cells.tsv
  - one JSON tensor file under group_tensors/
  - one MIN-REPORT manifest under manifests/
  - append-only completion state in cells_done.jsonl

The TSV and JSON files are the source of truth. W&B logging runs in a background
thread and falls back to grouped summary runs if per-cell run creation becomes
the bottleneck.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import queue
import random
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tinker
import tinker.types as T
from datasets import load_dataset

ROOT = Path("/home/claude/tinker-rl-lab")
OUT_DIR = ROOT / "experiments/results/mega_20260704"
GROUP_DIR = OUT_DIR / "group_tensors"
MANIFEST_DIR = OUT_DIR / "manifests"
LOG_DIR = ROOT / "experiments/tinker-runs/logs"
STOP_PATH = OUT_DIR / "STOP"
CELLS_TSV = OUT_DIR / "cells.tsv"
SKIPPED_TSV = OUT_DIR / "SKIPPED.tsv"
DONE_JSONL = OUT_DIR / "cells_done.jsonl"
FAIL_JSONL = OUT_DIR / "cells_failed.jsonl"
SUMMARY_JSON = OUT_DIR / "campaign_summary.json"

WANDB_ENTITY = "arvindcr4-pes-university"
WANDB_PROJECT = "zvf-audit-v2"
OLD_WANDB_PROJECT = "zvf-audit"
TOKEN_HARD_STOP = 80_000_000
PROMPTS_PER_CELL = 32

GSM_SYS = (
    "You are a math assistant. Solve the problem step by step, then give your "
    "final numerical answer inside \\boxed{}."
)
HUMANEVAL_SYS = (
    "You are an expert Python programmer. Complete the function for the given "
    "signature and docstring. Respond with only Python code."
)


@dataclass(frozen=True)
class Cell:
    model: str
    task_slice: str
    group_size: int
    temperature: float
    seed: int

    @property
    def model_family(self) -> str:
        return model_family(self.model)

    @property
    def id(self) -> str:
        raw = f"{self.model}|{self.task_slice}|{self.group_size}|{self.temperature}|{self.seed}"
        h = hashlib.sha1(raw.encode()).hexdigest()[:10]
        fam = re.sub(r"[^A-Za-z0-9]+", "-", self.model_family).strip("-")[:36]
        return f"{fam}_{self.task_slice}_G{self.group_size}_t{self.temperature:g}_s{self.seed}_{h}"


class JsonWandbLogger:
    def __init__(self, enabled: bool, queue_limit: int = 2000):
        self.enabled = enabled
        self.q: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_limit)
        self.thread: threading.Thread | None = None
        self.per_cell_failures = 0
        self.use_summary = False
        self.summary: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def start(self) -> None:
        if not self.enabled:
            return
        self.thread = threading.Thread(target=self._worker, name="wandb-bg", daemon=True)
        self.thread.start()

    def log_cell(self, row: dict[str, Any], cell: Cell) -> None:
        if not self.enabled:
            return
        try:
            self.q.put_nowait({"row": row, "cell": asdict(cell), "cell_id": cell.id})
        except queue.Full:
            self.use_summary = True
            self._stash(row, cell)

    def finish(self) -> None:
        if not self.enabled:
            return
        self.q.put(None)
        if self.thread is not None:
            self.thread.join(timeout=180)
        if self.summary:
            self._flush_summary_runs()

    def _stash(self, row: dict[str, Any], cell: Cell) -> None:
        key = (cell.model, cell.task_slice)
        slim = {k: row[k] for k in row if k not in {"reward_vectors_json"}}
        self.summary.setdefault(key, []).append(slim)

    def _worker(self) -> None:
        try:
            import wandb
        except Exception as e:
            print(f"[wandb] disabled: import failed: {e}", flush=True)
            return
        while True:
            item = self.q.get()
            if item is None:
                break
            row = item["row"]
            cell = Cell(**item["cell"])
            if self.use_summary:
                self._stash(row, cell)
                continue
            t0 = time.time()
            try:
                run = wandb.init(
                    entity=WANDB_ENTITY,
                    project=WANDB_PROJECT,
                    name=item["cell_id"],
                    config={
                        "model": cell.model,
                        "model_family": cell.model_family,
                        "task_slice": cell.task_slice,
                        "G": cell.group_size,
                        "temperature": cell.temperature,
                        "seed": cell.seed,
                    },
                    tags=["mega_20260704", "sampling-only", cell.task_slice],
                    reinit=True,
                    settings=wandb.Settings(start_method="thread"),
                )
                wandb.log({k: v for k, v in row.items() if isinstance(v, (int, float))})
                wandb.finish()
                if time.time() - t0 > 8:
                    self.use_summary = True
                    print("[wandb] switching to grouped summary runs; per-cell init is slow", flush=True)
            except Exception as e:
                self.per_cell_failures += 1
                self._stash(row, cell)
                print(f"[wandb] per-cell log failed for {item['cell_id']}: {e}", flush=True)
                if self.per_cell_failures >= 5:
                    self.use_summary = True

    def _flush_summary_runs(self) -> None:
        try:
            import wandb
        except Exception:
            return
        for (model, task), rows in sorted(self.summary.items()):
            try:
                name = f"summary_{re.sub(r'[^A-Za-z0-9]+', '-', model)[:45]}_{task}_20260704"
                run = wandb.init(
                    entity=WANDB_ENTITY,
                    project=WANDB_PROJECT,
                    name=name,
                    config={"model": model, "task_slice": task, "n_cells": len(rows), "fallback": "summary-table"},
                    tags=["mega_20260704", "summary-fallback", task],
                    reinit=True,
                    settings=wandb.Settings(start_method="thread"),
                )
                cols = sorted({k for r in rows for k in r.keys()})
                table = wandb.Table(columns=cols)
                for r in rows:
                    table.add_data(*[r.get(c) for c in cols])
                wandb.log({"cells": table, "n_cells": len(rows)})
                wandb.finish()
            except Exception as e:
                print(f"[wandb] summary log failed for {model}/{task}: {e}", flush=True)


def model_family(model: str) -> str:
    m = model
    for suffix in (":peft:262144", ":peft:131072"):
        m = m.replace(suffix, "")
    return m


def ensure_dirs() -> None:
    for p in (OUT_DIR, GROUP_DIR, MANIFEST_DIR, LOG_DIR):
        p.mkdir(parents=True, exist_ok=True)


def gsm_reward(response: str, answer: str) -> float:
    response = response.strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    for b in boxed:
        b_clean = b.strip().replace(",", "").replace(" ", "").replace("$", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            if b_clean.lower() == answer.lower():
                return 1.0
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", response)
    if nums:
        last = nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            pass
    return 0.0


_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = _CODE_FENCE.search(text)
    text = m.group(1) if m else text
    text = re.sub(r"```", "", text)
    return text.strip()


def run_humaneval_test(prompt: str, completion: str, test_code: str, entry_point: str, timeout: float = 3.0) -> float:
    code = extract_code(completion)
    if "def " not in code[:200]:
        full = prompt + "\n" + code
    else:
        full = code
    script = textwrap.dedent(
        f"""
        import math
        import re
        import sys
        from typing import *

        {full}

        {test_code}

        check({entry_point})
        """
    )
    fname = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            fname = f.name
        result = subprocess.run([sys.executable, fname], capture_output=True, timeout=timeout)
        return 1.0 if result.returncode == 0 else 0.0
    except Exception:
        return 0.0
    finally:
        if fname:
            try:
                os.unlink(fname)
            except Exception:
                pass


def load_task_examples(seed: int) -> dict[str, list[dict[str, Any]]]:
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    easy_pool = []
    hard_pool = []
    for idx, row in enumerate(gsm):
        m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if not m:
            continue
        ex = {
            "source_idx": idx,
            "question": row["question"],
            "answer": m.group(1).replace(",", "").strip(),
        }
        if idx < 500:
            easy_pool.append(ex)
        if idx >= 5000:
            hard_pool.append(ex)
    rng = random.Random(seed)
    rng.shuffle(easy_pool)
    rng.shuffle(hard_pool)

    he = load_dataset("openai/openai_humaneval", split="test")
    he_items = list(he)
    rng.shuffle(he_items)

    return {
        "gsm8k_easy": easy_pool,
        "gsm8k_hard": hard_pool,
        "humaneval_subset": he_items[:32],
    }


def make_prompt(task_slice: str, ex: dict[str, Any]) -> str:
    if task_slice.startswith("gsm8k"):
        return (
            f"<|im_start|>system\n{GSM_SYS}<|im_end|>\n"
            f"<|im_start|>user\n{ex['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return (
        f"<|im_start|>system\n{HUMANEVAL_SYS}<|im_end|>\n"
        f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def score_completion(task_slice: str, ex: dict[str, Any], completion: str) -> float:
    if task_slice.startswith("gsm8k"):
        return gsm_reward(completion, ex["answer"])
    return run_humaneval_test(ex["prompt"], completion, ex["test"], ex["entry_point"])


def metric_summary(reward_groups: list[list[float]], length_groups: list[list[int]], G: int) -> dict[str, float]:
    rewards = [r for group in reward_groups for r in group]
    lengths = [l for group in length_groups for l in group]
    n_groups = len(reward_groups)
    phats = [sum(g) / len(g) for g in reward_groups if g]
    return {
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "zvf": (sum(1 for g in reward_groups if len(set(g)) == 1) / n_groups) if n_groups else 0.0,
        "pcd": (((G - 1) / G) * (sum(p * (1 - p) for p in phats) / len(phats))) if phats else 0.0,
        "mean_completion_len": sum(lengths) / len(lengths) if lengths else 0.0,
        "std_completion_len": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
    }


def get_tokenizer(svc: tinker.ServiceClient, model: str):
    try:
        tc = svc.create_lora_training_client(base_model=model, rank=8)
        return tc.get_tokenizer()
    except Exception:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def discover_models(svc: tinker.ServiceClient, target_cells: int, max_models: int | None) -> list[str]:
    caps = svc.get_server_capabilities()
    names = [m.model_name for m in caps.supported_models]
    preferred = [
        "meta-llama/Llama-3.2-3B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3.5-9B",
        "openai/gpt-oss-20b",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3.6-35B-A3B",
    ]
    models = [m for m in preferred if m in names]
    if max_models is not None:
        models = models[:max_models]
    else:
        cells_per_model = 3 * 5 * 2 * 2
        need_models = max(5, math.ceil(target_cells / cells_per_model))
        models = models[:need_models]
    return models


def load_done_ids() -> set[str]:
    done: set[str] = set()
    if DONE_JSONL.exists():
        with DONE_JSONL.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    done.add(json.loads(line)["cell_id"])
                except Exception:
                    pass
    return done


def inventory_covered() -> set[tuple[str, str, int, float]]:
    covered: set[tuple[str, str, int, float]] = set()
    inv_dir = ROOT / "experiments/results/wandb_inventory"
    if inv_dir.exists():
        for path in inv_dir.glob("*.tsv"):
            try:
                with path.open() as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        model = row.get("model") or row.get("base_model") or row.get("model_name")
                        g = row.get("group_size") or row.get("G") or row.get("group")
                        task = row.get("task") or row.get("task_slice") or row.get("dataset") or ""
                        temp = row.get("temperature") or row.get("temp") or row.get("sampling_temperature")
                        if not model or not g:
                            continue
                        try:
                            gi = int(float(g))
                        except Exception:
                            continue
                        task_norm = normalize_task(task)
                        if task_norm is None:
                            continue
                        temps = [float(temp)] if temp not in (None, "") else ([1.0] if gi in {4, 8, 16} else [])
                        for tv in temps:
                            covered.add((model_family(model), task_norm, gi, tv))
            except Exception as e:
                print(f"[skip-load] failed reading {path}: {e}", flush=True)
    try:
        import wandb

        api = wandb.Api()
        runs = api.runs(f"{WANDB_ENTITY}/{OLD_WANDB_PROJECT}", per_page=200)
        for run in runs:
            cfg = dict(run.config or {})
            model = cfg.get("model") or cfg.get("base_model") or getattr(run, "name", "")
            g = cfg.get("G") or cfg.get("group_size") or cfg.get("group")
            if model and g:
                try:
                    gi = int(float(g))
                except Exception:
                    continue
                if gi in {4, 8, 16}:
                    covered.add((model_family(str(model)), "gsm8k_easy", gi, 1.0))
    except Exception as e:
        print(f"[skip-load] W&B old-project query failed: {e}", flush=True)
    return covered


def normalize_task(task: str) -> str | None:
    t = (task or "").lower()
    if "human" in t:
        return "humaneval_subset"
    if "gsm" in t or "qwen3" in t or "llama3" in t:
        return "gsm8k_easy"
    return None


def ordered_cells(models: list[str], covered: set[tuple[str, str, int, float]]) -> tuple[list[Cell], list[tuple[Cell, str]]]:
    tasks = ["humaneval_subset", "gsm8k_hard", "gsm8k_easy"]
    gs = [2, 32, 4, 8, 16]
    temps = [0.6, 1.0]
    seeds = [0, 1]
    cells: list[Cell] = []
    skipped: list[tuple[Cell, str]] = []
    for model in models:
        for task in tasks:
            for g in gs:
                for temp in temps:
                    for seed in seeds:
                        c = Cell(model, task, g, temp, seed)
                        if (c.model_family, task, g, temp) in covered:
                            skipped.append((c, "covered_by_inventory_or_zvf_audit"))
                        else:
                            cells.append(c)
    return cells, skipped


def write_skipped(skipped: list[tuple[Cell, str]]) -> None:
    write_header = not SKIPPED_TSV.exists() or SKIPPED_TSV.stat().st_size == 0
    with SKIPPED_TSV.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["cell_id", "model", "model_family", "task_slice", "G", "temperature", "seed", "reason"],
        )
        if write_header:
            writer.writeheader()
        for c, reason in skipped:
            writer.writerow(
                {
                    "cell_id": c.id,
                    "model": c.model,
                    "model_family": c.model_family,
                    "task_slice": c.task_slice,
                    "G": c.group_size,
                    "temperature": c.temperature,
                    "seed": c.seed,
                    "reason": reason,
                }
            )


class Campaign:
    def __init__(self, args: argparse.Namespace, cells: list[Cell]):
        self.args = args
        self.cells = cells
        self.svc = tinker.ServiceClient(base_url=None)
        self.tokenizers: dict[str, Any] = {}
        self.examples = load_task_examples(seed=12345)
        self.done_ids = load_done_ids()
        self.file_lock = asyncio.Lock()
        self.token_lock = asyncio.Lock()
        self.cumulative_tokens = self._load_existing_tokens()
        self.completed = 0
        self.failed = 0
        self.launched = 0
        self.stop_requested = False
        self.wandb = JsonWandbLogger(enabled=not args.no_wandb)

    def _load_existing_tokens(self) -> int:
        total = 0
        if CELLS_TSV.exists():
            try:
                with CELLS_TSV.open() as f:
                    for row in csv.DictReader(f, delimiter="\t"):
                        total += int(float(row.get("sampled_tokens", 0) or 0))
            except Exception:
                pass
        return total

    async def run(self) -> None:
        self.wandb.start()
        sem = asyncio.Semaphore(self.args.concurrency)
        pending = []
        for cell in self.cells:
            if cell.id in self.done_ids:
                continue
            if STOP_PATH.exists() or self.stop_requested:
                print("[stop] STOP file or signal seen before launch loop", flush=True)
                break
            async with self.token_lock:
                if self.cumulative_tokens >= TOKEN_HARD_STOP:
                    print("[budget] token hard stop reached before launch loop", flush=True)
                    break
            pending.append(asyncio.create_task(self._run_with_sem(cell, sem)))
            self.launched += 1
            if self.args.limit and self.launched >= self.args.limit:
                break
        if pending:
            await asyncio.gather(*pending)
        self.wandb.finish()
        self._write_summary()

    async def _run_with_sem(self, cell: Cell, sem: asyncio.Semaphore) -> None:
        async with sem:
            if STOP_PATH.exists() or self.stop_requested:
                return
            try:
                await asyncio.wait_for(self.run_cell(cell), timeout=self.args.cell_timeout_sec)
                self.completed += 1
            except asyncio.TimeoutError:
                self.failed += 1
                await self._record_failure(cell, "timeout")
            except Exception as e:
                self.failed += 1
                await self._record_failure(cell, f"{type(e).__name__}: {e}")

    def tokenizer_for(self, model: str):
        if model not in self.tokenizers:
            self.tokenizers[model] = get_tokenizer(self.svc, model)
        return self.tokenizers[model]

    async def tokenizer_for_async(self, model: str):
        if model not in self.tokenizers:
            self.tokenizers[model] = await asyncio.to_thread(get_tokenizer, self.svc, model)
        return self.tokenizers[model]

    async def run_cell(self, cell: Cell) -> None:
        tok = await self.tokenizer_for_async(cell.model)
        rng = random.Random((cell.seed * 1_000_003) ^ int(hashlib.sha1(cell.id.encode()).hexdigest()[:8], 16))
        pool = list(self.examples[cell.task_slice])
        rng.shuffle(pool)
        selected = pool[:PROMPTS_PER_CELL]
        sc = await asyncio.to_thread(lambda: self.svc.create_sampling_client(base_model=cell.model))
        sp = T.SamplingParams(max_tokens=self.args.max_tokens, temperature=cell.temperature, top_p=self.args.top_p)

        reward_groups: list[list[float]] = []
        length_groups: list[list[int]] = []
        prompt_indices: list[Any] = []
        sample_errors = 0

        async def one_group(idx: int, ex: dict[str, Any]) -> None:
            nonlocal sample_errors
            prompt = make_prompt(cell.task_slice, ex)
            ids = tok.encode(prompt, add_special_tokens=False)
            if len(ids) > self.args.max_prompt_tokens:
                ids = ids[-self.args.max_prompt_tokens :]
            resp = None
            for attempt in range(3):
                try:
                    resp = await sc.sample_async(
                        prompt=T.ModelInput.from_ints(ids),
                        num_samples=cell.group_size,
                        sampling_params=sp,
                    )
                    break
                except Exception:
                    if attempt == 2:
                        sample_errors += 1
                        return
                    await asyncio.sleep(2 * (attempt + 1))
            rewards: list[float] = []
            lengths: list[int] = []
            for seq in resp.sequences:
                tokens = list(seq.tokens)
                text = tok.decode(tokens, skip_special_tokens=True)
                rewards.append(score_completion(cell.task_slice, ex, text))
                lengths.append(len(tokens))
            reward_groups.append(rewards)
            length_groups.append(lengths)
            prompt_indices.append(ex.get("source_idx", ex.get("task_id", idx)))

        await asyncio.gather(*(one_group(i, ex) for i, ex in enumerate(selected)))
        if not reward_groups:
            raise RuntimeError("no successful prompt groups")

        metrics = metric_summary(reward_groups, length_groups, cell.group_size)
        sampled_tokens = sum(sum(g) for g in length_groups)
        async with self.token_lock:
            self.cumulative_tokens += sampled_tokens
            cumulative = self.cumulative_tokens
        if cumulative >= TOKEN_HARD_STOP:
            self.stop_requested = True

        tensor_path = GROUP_DIR / f"{cell.id}.json"
        manifest_path = MANIFEST_DIR / f"{cell.id}.json"
        tensor_doc = {
            "cell_id": cell.id,
            "cell": asdict(cell),
            "prompt_indices": prompt_indices,
            "reward_vectors": reward_groups,
            "completion_lengths": length_groups,
            "sample_errors": sample_errors,
        }
        manifest = {
            "cell_id": cell.id,
            "loss_form": "n/a-sampling",
            "ref_policy_kl": "n/a",
            "sampler_backend_precision": "tinker-closed",
            "per_step_zvf_path": str(tensor_path),
            "group_size_schedule": f"fixed-G={cell.group_size}",
            "heldout_split": cell.task_slice,
            "decontamination_notes": "gsm8k-train-slice" if cell.task_slice.startswith("gsm8k") else "humaneval-openai-subset",
        }
        row = {
            "cell_id": cell.id,
            "timestamp": time.time(),
            "model": cell.model,
            "model_family": cell.model_family,
            "task_slice": cell.task_slice,
            "G": cell.group_size,
            "temperature": cell.temperature,
            "seed": cell.seed,
            "n_groups": len(reward_groups),
            "sample_errors": sample_errors,
            "mean_reward": metrics["mean_reward"],
            "zvf": metrics["zvf"],
            "pcd": metrics["pcd"],
            "mean_completion_len": metrics["mean_completion_len"],
            "std_completion_len": metrics["std_completion_len"],
            "sampled_tokens": sampled_tokens,
            "cumulative_sampled_tokens": cumulative,
            "reward_vectors_json": json.dumps(reward_groups, separators=(",", ":")),
            "tensor_path": str(tensor_path),
            "manifest_path": str(manifest_path),
        }
        await self._write_cell_outputs(cell, row, tensor_path, tensor_doc, manifest_path, manifest)
        self.wandb.log_cell(row, cell)
        print(
            f"[done] {cell.id} reward={metrics['mean_reward']:.3f} zvf={metrics['zvf']:.3f} "
            f"pcd={metrics['pcd']:.4f} tokens={sampled_tokens} cumulative={cumulative}",
            flush=True,
        )

    async def _write_cell_outputs(
        self,
        cell: Cell,
        row: dict[str, Any],
        tensor_path: Path,
        tensor_doc: dict[str, Any],
        manifest_path: Path,
        manifest: dict[str, Any],
    ) -> None:
        async with self.file_lock:
            tensor_path.write_text(json.dumps(tensor_doc, indent=2) + "\n")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            write_header = not CELLS_TSV.exists() or CELLS_TSV.stat().st_size == 0
            with CELLS_TSV.open("a", newline="") as f:
                writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
            with DONE_JSONL.open("a") as f:
                f.write(json.dumps({"cell_id": cell.id, "cell": asdict(cell), "timestamp": time.time()}) + "\n")
                f.flush()
                os.fsync(f.fileno())

    async def _record_failure(self, cell: Cell, reason: str) -> None:
        async with self.file_lock:
            with FAIL_JSONL.open("a") as f:
                f.write(json.dumps({"cell_id": cell.id, "cell": asdict(cell), "reason": reason, "timestamp": time.time()}) + "\n")
                f.flush()
                os.fsync(f.fileno())
        print(f"[failed] {cell.id}: {reason}", flush=True)

    def _write_summary(self) -> None:
        summary = {
            "launched": self.launched,
            "completed_this_process": self.completed,
            "failed_this_process": self.failed,
            "cumulative_sampled_tokens": self.cumulative_tokens,
            "cells_tsv": str(CELLS_TSV),
            "done_jsonl": str(DONE_JSONL),
            "failed_jsonl": str(FAIL_JSONL),
        }
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[summary] {json.dumps(summary, sort_keys=True)}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--cell-timeout-sec", type=int, default=480)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--target-cells", type=int, default=500)
    p.add_argument("--max-models", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def install_signal_handlers(campaign_ref: dict[str, Campaign]) -> None:
    def handler(signum, frame):
        c = campaign_ref.get("campaign")
        if c is not None:
            c.stop_requested = True
        print(f"[signal] received {signum}; graceful stop requested", flush=True)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


async def async_main() -> int:
    args = parse_args()
    ensure_dirs()
    svc = tinker.ServiceClient(base_url=None)
    models = await asyncio.to_thread(discover_models, svc, args.target_cells, args.max_models)
    covered = await asyncio.to_thread(inventory_covered)
    cells, skipped = ordered_cells(models, covered)
    done_ids = load_done_ids()
    cells = [c for c in cells if c.id not in done_ids]
    if args.smoke:
        args.limit = 2
        args.concurrency = min(args.concurrency, 2)
        args.no_wandb = True
        cells = cells[:2]
    write_skipped(skipped)
    print(
        f"[plan] models={models} runnable_cells={len(cells)} skipped={len(skipped)} "
        f"already_done={len(done_ids)} smoke={args.smoke}",
        flush=True,
    )
    if not cells:
        return 0
    campaign_ref: dict[str, Campaign] = {}
    install_signal_handlers(campaign_ref)
    campaign = Campaign(args, cells)
    campaign_ref["campaign"] = campaign
    await campaign.run()
    return 0 if campaign.failed == 0 or campaign.completed > 0 else 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
