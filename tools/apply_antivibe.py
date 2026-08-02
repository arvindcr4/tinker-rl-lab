#!/usr/bin/env python3
"""apply_antivibe.py -- per-file AntiVibe deep-dive generator (compact mode).

Applies the AntiVibe framework (https://github.com/mohi-devhub/antivibe) to
every real source file in tinker-rl-lab: one educational "deep dive" markdown
document per file, mirrored under deep-dive/per-file/<module>/...

Scope (per repo-owner decision):
  * Python (.py), shell (.sh), YAML (.yaml/.yml), TOML (.toml)
  * a few key top-level JSON config files
Excluded: git metadata, virtualenvs, caches, data/lockfiles, generated
artifacts (output/, outputs/), and the .claude/worktrees/ checkouts -- which
are duplicate copies of this same codebase at other branches.

This is a deterministic, stateless implementation of AntiVibe's "compact"
output mode: Overview, Key Components, Concepts & Decisions (what + why),
and Related Code. The interactive skill lives in .claude/skills/antivibe/.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
import textwrap
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "deep-dive" / "per-file"
NOW = datetime.now(timezone.utc)
TS = NOW.strftime("%Y-%m-%d %H:%M UTC")

# --- scope -----------------------------------------------------------------
EXCLUDE_DIR_NAMES = {
    ".git", ".venv", "node_modules", "wandb", "__pycache__",
    "tinkerrl.egg-info", ".pytest_cache", ".ruff_cache", ".playwright-mcp",
}
EXCLUDE_RELPATH_PREFIXES = (".claude/worktrees", "output", "outputs")
EXCLUDE_FILENAMES = {
    "uv.lock", "settings.local.json", "modal_results_all.json",
    "xgboost_results.json", "prompt-decomposition.json",
}

INCLUDE_EXTS = {".py", ".sh", ".yaml", ".yml", ".toml"}
JSON_MAX_DEPTH = 2          # include *.json only near the repo root
JSON_MAX_SIZE = 200_000     # skip big data blobs


def is_excluded(rel: Path) -> bool:
    if any(p in EXCLUDE_DIR_NAMES for p in rel.parts):
        return True
    if any(str(rel).startswith(pre) for pre in EXCLUDE_RELPATH_PREFIXES):
        return True
    return False


def walk_source_files() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        d = Path(dirpath)
        rel_dir = d.relative_to(ROOT)
        if is_excluded(rel_dir):
            dirnames[:] = []
            continue
        dirnames[:] = [x for x in dirnames if x not in EXCLUDE_DIR_NAMES]
        for name in filenames:
            rel = rel_dir / name if str(rel_dir) != "." else Path(name)
            if is_excluded(rel):
                continue
            if name in EXCLUDE_FILENAMES:
                continue
            ext = rel.suffix.lower()
            if ext in INCLUDE_EXTS:
                files.append(rel)
            elif ext == ".json":
                size = (d / name).stat().st_size
                if len(rel.parts) <= JSON_MAX_DEPTH and size <= JSON_MAX_SIZE:
                    files.append(rel)
    return sorted(files)


# --- concept knowledge base -------------------------------------------------
# Tuples: (name, what, why, when, tradeoffs). Docs pull the subset that the
# static analysis actually detects in each file.
CONCEPTS: dict[str, tuple[str, str, str, str, str]] = {
    "torch": (
        "PyTorch tensor computation & autograd",
        "PyTorch is the numeric engine: `torch.Tensor` holds batched GPU/CPU arrays and "
        "`torch.autograd` builds the computation graph so gradients flow from a loss "
        "back to every parameter.",
        "TRL, transformers, vLLM and this repo's RL loops are all built on PyTorch, so "
        "using it directly avoids impedance mismatch between framework and training code.",
        "Anywhere gradients must reach model weights -- training, RL rollouts, LoRA "
        "adaptation, or evaluation under a different dtype.",
        "Eager execution is easy to debug but slower than compiled graphs; "
        "`torch.compile`/export recover speed at the cost of traceability.",
    ),
    "transformers": (
        "Hugging Face Transformers (pretrained models & tokenizers)",
        "The `transformers` library loads pretrained checkpoints (here Qwen3-8B) and their "
        "tokenizers behind a uniform `AutoModelForCausalLM`/`AutoTokenizer` interface.",
        "It gives one stable API over many architectures plus hosted checkpoints, which is "
        "why it is the shared backbone across every framework in this repo.",
        "Any task that starts from an existing LLM and adds training, serving, or eval.",
        "The abstraction hides internals; subtle differences between architectures can "
        "surprise you when you rely on undocumented behavior.",
    ),
    "peft": (
        "Parameter-Efficient Fine-Tuning (LoRA & friends)",
        "PEFT freezes the base weights and trains small low-rank adapter matrices "
        "(LoRA r=16 here) plus a handful of config classes (`LoraConfig`).",
        "LoRA makes the canonical 30-step GRPO run affordable and makes checkpoints tiny "
        "and shareable -- the repo freezes `peft` into every framework path.",
        "When you want to adapt a large model without retraining it or shipping full copies.",
        "Adapters cap what the model can learn and add a merge step before inference; "
        "k-bit quantized base weights complicate offloading.",
    ),
    "trl": (
        "HF TRL training library (PPO/GRPO-style RLHF)",
        "`trl` implements RLHF-style trainers (PPO, and the GRPO variant used here) that "
        "coordinate policy, reference model, reward computation, and rollout generation.",
        "It removes the burden of writing the RL loop from scratch and is one of the five "
        "frameworks whose equivalence this repo proves.",
        "When you need a maintained, well-tested RLHF loop and accept its configuration model.",
        "The library's opinionated defaults can fight custom research setups; equivalence "
        "testing exists precisely because each framework behaves slightly differently.",
    ),
    "vllm": (
        "vLLM high-throughput inference serving",
        "vLLM serves LLMs with paged attention and continuous batching, turning costly "
        "rollout generation into a fast, batched operation.",
        "RL rollouts need thousands of completions per step; vLLM's batching is what makes "
        "that tractable on limited GPU time.",
        "Whenever inference volume (not just latency) is the bottleneck -- exactly the RL "
        "rollout case.",
        "Serving adds a process boundary and a different memory footprint; small models "
        "can be slower through vLLM than a plain HF forward pass.",
    ),
    "wandb": (
        "Experiment tracking with Weights & Biases",
        "W&B records metrics, hyperparameters, and artifacts to a hosted or local run "
        "timeline, giving every training run a shareable dashboard and history.",
        "The repo treats receipts/evidence as first-class outputs, and W&B is one of the "
        "three independent channels (HF + W&B + GCS) whose agreement is the trust signal.",
        "When a run's value is in its history -- comparing sweeps, auditing, or sharing "
        "results without sending weights.",
        "Adds a network dependency and an external account; local-only runs must opt out or "
        "write a local fallback.",
    ),

    "pytest": (
        "Automated verification with pytest",
        "`pytest` discovers `test_*.py`/`*_test.py` functions and classes, runs them in "
        "isolation, and reports failures with rich introspection.",
        "This repo cannot run real GPU training in CI, so pytest tests pin *invariants* of "
        "the 30-cell matrix -- a substitute for compute.",
        "For any behavior that can be asserted without expensive hardware: parsing, "
        "dispatch, config validity, dry-run plans.",
        "Tests prove plumbing, not gradient correctness; a cell can pass all tests and "
        "still produce wrong training numbers.",
    ),
    "argparse": (
        "Command-line argument parsing",
        "`argparse` turns `sys.argv` into typed options (`--framework`, `--dry-run`) with "
        "help text and error handling for free.",
        "Every platform entry point must be runnable by humans and by shelling-out code, so "
        "a stable, documented CLI is the contract between them.",
        "When a script is invoked by people, CI, or other processes and needs explicit knobs.",
        "Boilerplate-heavy and positional-only; richer CLIs use click/typer for nesting and "
        "auto-generated help.",
    ),
    "click": (
        "Declarative CLI framework (click/typer)",
        "Click (and typer on top) builds commands from decorated functions, auto-generating "
        "help, validation, and grouped subcommands.",
        "It makes multi-command CLIs (train / eval / sweep) readable and hard to get wrong.",
        "When a tool grows beyond two or three flags and has distinct subcommands.",
        "Adds a dependency and indirection; argparse is still fine for one-shot scripts.",
    ),
    "dataclass": (
        "Data modeling with dataclasses",
        "`@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from field "
        "annotations, turning plain classes into compact value objects.",
        "The repo models specs, results, and plans as frozen dataclasses so structural "
        "equality and hashing come for free and mutation is blocked.",
        "For passive data carriers -- configs, results, plans -- especially when you want "
        "`==`/hash semantics.",
        "No validation by itself; frozen fields protect from mutation but not bad values "
        "(pair with pydantic for that).",
    ),
    "pydantic": (
        "Schema validation with Pydantic",
        "Pydantic parses and validates input data against annotated field types at "
        "construction time, raising on mismatches before they can corrupt a run.",
        "Config that enters a training run is validated here so a bad knob fails fast and "
        "loud instead of silently changing behavior.",
        "Boundaries where untrusted or hand-edited data meets logic: config files, API "
        "payloads, CLI-parsed specs.",
        "Runtime overhead is small but nonzero; strict validation can reject legitimate "
        "edge cases unless coercions are tuned.",
    ),
    "abc": (
        "Abstract base classes & the Strategy pattern",
        "`ABC` + `@abstractmethod` declare an interface that subclasses must implement, "
        "making 'plug a new backend/framework here' a compile-time-shaped contract.",
        "The `Backend` hierarchy (local/colab/modal/gcp/vast/hfspaces) is a Strategy: every "
        "backend implements `plan()`/`run()` so dispatch is uniform.",
        "When you have multiple interchangeable implementations behind one call site.",
        "Forces all variants into one shape; backends with very different I/O (serverless "
        "vs SSH) strain the interface.",
    ),
    "protocol": (
        "Structural subtyping with typing.Protocol",
        "`Protocol` describes an interface by the *attributes* something has, not by "
        "inheritance -- anything matching the shape satisfies it (duck typing with "
        "static checks).",
        "Lets the code accept `plan`-like and `run`-like objects without forcing a class "
        "hierarchy, useful in the shim layer.",
        "When many small objects share behavior but have no common ancestor.",
        "Runtime `isinstance` checks need `@runtime_checkable` and are shallow; static "
        "checkers are the real beneficiary.",
    ),


    "memoization": (
        "Caching / memoization",
        "`functools.cache`/`lru_cache` store results keyed by arguments so repeated calls "
        "with the same input return instantly.",
        "Expensive resolutions (lazy backends, registry lookups, file parsing) are computed "
        "once and reused across the dispatch path.",
        "Pure functions called repeatedly with the same arguments, where recomputation is "
        "costly.",
        "Caches hold references (memory) and can go stale; never cache across mutable state.",
    ),
    "enum": (
        "Enumerated constants",
        "`enum.Enum` names a closed set of values, preventing typos and giving each choice "
        "a stable identity and repr.",
        "Frameworks, backends, and result codes are finite sets -- Enum makes an unknown "
        "value a type error instead of a silent wrong path.",
        "Anywhere code branches on a small fixed vocabulary of strings.",
        "Adding a value is a code change everywhere it's switched on; fine because the "
        "vocabulary is truly closed.",
    ),
    "asyncio": (
        "Asynchronous I/O (asyncio / async def)",
        "`async`/`await` lets one thread interleave many I/O-bound operations (network, "
        "subprocesses) instead of blocking on each.",
        "Drivers that fan out to remote boxes or API calls benefit from overlapping "
        "wait-time; async is the idiomatic way to keep those concurrent.",
        "I/O-bound fan-out where you'd otherwise stall on many sequential round-trips; "
        "CPU-bound compute still needs threads/processes.",
        "Async code is more invasive (an async function must await other async functions) "
        "and easier to deadlock if a blocking call sneaks in.",
    ),
    "generators": (
        "Generators & lazy pipelines",
        "`yield` turns a function into a generator that produces values on demand instead "
        "of materializing a full list up front.",
        "Streaming long result sets (rollouts, log lines, remote listings) one item at a "
        "time keeps memory flat regardless of dataset size.",
        "When you iterate over something too large to hold in memory at once.",
        "Generators are single-pass and stateful; you can't rewind one, and exceptions "
        "surfaces only when you pull the next value.",
    ),
    "logging": (
        "Structured diagnostics with logging",
        "The `logging` module writes level-filtered messages to stderr/files, separating "
        "operational noise from real errors and leaving them toggleable at runtime.",
        "Runs are audited, so leaving a trail of INFO/DEBUG statements lets a reviewer "
        "reconstruct what happened without rerunning GPUs.",
        "Anywhere you'd `print` something that matters: progress, warnings, step "
        "boundaries, and fatal errors.",
        "More setup than `print`; misconfigured handler levels silently swallow the very "
        "lines you need in production.",
    ),
    "subprocess": (
        "Process orchestration (subprocess)",
        "`subprocess.run`/`Popen` spawns and captures external commands, letting Python "
        "drive shell steps, remote CLIs, and other tools as child processes.",
        "Remote backends provision a box then shell out to a driver command -- subprocess "
        "is the seam between 'plan' and 'actually run elsewhere'.",
        "When work is naturally a separate executable: `modal run`, `gcloud`, ssh commands, "
        "secondary scripts.",
        "Argument quoting/escaping and env leakage are footguns; you lose in-process "
        "debugging across the boundary.",
    ),
    "config": (
        "Configuration as declarative data (YAML/JSON/TOML)",
        "Knobs live in YAML/JSON/TOML files or tables rather than code, so a run's "
        "intent is inspectable and diffable without reading the program.",
        "A single frozen `CanonicalSpec` + preregistration files is the repo's whole "
        "comparability contract -- config-as-data is what makes runs hashable and testable.",
        "Anywhere parameters should be changeable without editing code, or compared across "
        "runs.",
        "Config can drift from what the code actually reads; validation (pydantic) is what "
        "catches a key that no longer means what it says.",
    ),


    "http": (
        "HTTP client calls",
        "`requests`/`httpx`/`aiohttp` issue HTTP requests to APIs -- model hosting, "
        "receipt uploads (HF/W&B/GCS), or remote preflight checks.",
        "Evidence must land on independent channels, and those channels are network "
        "APIs, so HTTP is how receipts and checkpoints actually get out.",
        "Any interaction with a REST endpoint: upload, download, health-check, serverless "
        "invocation.",
        "Network calls fail; you need timeouts, retries, and idempotency or a transient "
        "blip becomes a lost run.",
    ),
    "web": (
        "Web service / API layer",
        "A web framework (FastAPI/Flask) exposes endpoints so external actors can query or "
        "drive the system over HTTP with structured request/response schemas.",
        "Demos and live checkpoints are served over the web so reviewers can interact "
        "without shipping weights or code.",
        "When something must be reachable by a browser or another service at a URL.",
        "A server is a long-running, externally reachable surface -- more security and "
        "liveness concerns than a plain script.",
    ),
    "ui": (
        "Interactive UI (Streamlit/Gradio)",
        "Streamlit/Gradio turn Python functions into a browser UI with widgets, running "
        "callbacks in a local web server.",
        "Interactive demos (chat with the trained model, view results) are the fastest way "
        "to show what a checkpoint does.",
        "For a shareable demo of model behavior or a quick internal tool.",
        "A UI is stateful and imperative-feeling; it's presentation glue, not core logic, "
        "so keep the real logic in plain functions.",
    ),
    "numpy": (
        "Numeric arrays with NumPy",
        "NumPy gives dense N-d arrays and vectorized math (reductions, broadcasting) that "
        "run at C speed.",
        "Reward computation and metrics are array operations; vectorizing over a batch is "
        "both faster and more readable than Python loops.",
        "Any batched numeric transform -- rewards, accuracy, aggregations across rollouts.",
        "NumPy and torch each own their memory; converting between them copies unless you "
        "share storage carefully.",
    ),
    "pandas": (
        "Tabular data with pandas",
        "pandas DataFrames hold labeled, columnar data and offer groupby/merge/agg "
        "one-liners over CSV/JSON exports.",
        "Experiment logs and result exports aggregate nicely into tables for reporting and "
        "audits.",
        "When you'd otherwise hand-roll loops over rows of CSV/JSON results.",
        "DataFrames are heavier than raw arrays; overuse for tiny data adds import cost "
        "and ambiguity about index semantics.",
    ),
    "viz": (
        "Data visualization",
        "Matplotlib/Plotly render metrics into figures, replacing dense number tables with "
        "readable curves.",
        "The repo produces decks and figures as code so charts derive from evidence and "
        "regenerate whenever the checkout changes.",
        "When a comparison (scaling curve, loss trace, ablation) is clearer as a picture "
        "than a table.",
        "Figures need explicit styling to stay trustworthy; a miscalled axis or log scale "
        "can misrepresent the claim.",
    ),
    "parallel": (
        "Parallelism (threads / processes / futures)",
        "`concurrent.futures`/threading/multiprocessing run independent work concurrently, "
        "cutting wall-clock for fan-out tasks.",
        "Rollout generation and multi-backend dispatch are embarrassingly parallel, so "
        "pools/futures are a cheap win.",
        "Many independent units of work that share no mutable state.",
        "Threads don't speed up CPU-bound Python (GIL); processes add copy/serialization "
        "cost and need picklable arguments.",
    ),
    "regex": (
        "Text processing with regular expressions",
        "`re` matches/extracts patterns in text -- parsing logs, sanitizing identifiers, "
        "or validating formats that don't warrant a full parser.",
        "Receipts, path probes, and name munging are string-shaped; regex is the compact "
        "tool for targeted extraction.",
        "Small, well-defined text patterns where a parser is overkill.",
        "Regex is opaque and easy to get subtly wrong; complex grammars should graduate to "
        "a real parser.",
    ),
    "csv": (
        "CSV I/O",
        "`csv` reads/writes comma-separated records, the lingua franca for tabular data "
        "and result dumps.",
        "Large benchmark/results files are exchanged as CSV, so importing/exporting that "
        "format is a direct requirement.",
        "When tabular data must be human-openable or compatible with spreadsheets/other tools.",
        "CSV has no schema or types -- every field is a string, so parsing and quoting edge "
        "cases are on you.",
    ),
    "dotenv": (
        "Environment-driven configuration",
        "`dotenv`/pydantic-settings load secrets and knobs from the environment, keeping "
        "API keys out of source and letting deployment override defaults.",
        "Receipt channels need credentials that must not be committed; env vars are the "
        "standard place for them.",
        "Anything secret or environment-specific (API keys, endpoint URLs, run ids).",
        "Env vars are untyped strings and easy to forget; validate them at startup rather "
        "than mid-run.",
    ),
}


# import-module-prefix -> concept key
LIB_MAP: list[tuple[str, str]] = [
    ("torch", "torch"), ("transformers", "transformers"), ("peft", "peft"),
    ("trl", "trl"), ("vllm", "vllm"), ("wandb", "wandb"),
    ("pytest", "pytest"), ("click", "click"), ("typer", "click"),
    ("argparse", "argparse"), ("pydantic", "pydantic"),
    ("dataclasses", "dataclass"), ("abc", "abc"), ("typing", "protocol"),
    ("functools", "memoization"), ("enum", "enum"), ("asyncio", "asyncio"),
    ("logging", "logging"), ("subprocess", "subprocess"),
    ("yaml", "config"), ("json", "config"), ("tomllib", "config"),
    ("requests", "http"), ("httpx", "http"), ("aiohttp", "http"),
    ("fastapi", "web"), ("flask", "web"), ("streamlit", "ui"),
    ("gradio", "ui"), ("numpy", "numpy"), ("pandas", "pandas"),
    ("matplotlib", "viz"), ("plotly", "viz"),
    ("concurrent.futures", "parallel"), ("threading", "parallel"),
    ("multiprocessing", "parallel"), ("re", "regex"), ("csv", "csv"),
    ("dotenv", "dotenv"), ("pydantic_settings", "dotenv"),
]

# role -> (one-line sentence, concept name, concept what)
ROLES = {
    "test": (
        "a test/verification module that pins invariants of surrounding code so "
        "regressions are caught without manual checking",
        "Verification as an invariant gate",
        "Instead of asserting on trained results, these tests assert structural and "
        "dispatch invariants (config validity, dry-run plans, framework threading).",
    ),
    "train": (
        "a training path that runs gradient-based optimization (GRPO/PPO-style) over "
        "model weights",
        "Gradient-based RL training loop",
        "Rollouts are generated, scored by a reward model, and their feedback is "
        "backpropagated through a policy updated toward higher reward.",
    ),
    "eval": (
        "an evaluation/measurement script that quantifies outcomes and produces evidence",
        "Measurement as evidence",
        "It turns raw run outputs into comparable metrics and receipts rather than "
        "anecdotes.",
    ),
    "bench": (
        "a benchmarking/sweep driver that runs many configurations and compares them",
        "Batched benchmarking / sweep",
        "A sweep varies one or more knobs across fixed axes and aggregates the results "
        "for comparison.",
    ),
    "cli": (
        "an entry point that parses intent from the command line and dispatches to the "
        "underlying machinery",
        "Entry point & dispatch",
        "It translates `--framework/--backend` flags into a plan, then executes or "
        "dry-runs it, acting as the seam between human intent and framework-specific code.",
    ),
    "experiment": (
        "an experiment script that exercises a specific research configuration end-to-end",
        "Experiment script",
        "It wires a chosen model, dataset, algorithm, and backend into one reproducible "
        "run and records the outcome.",
    ),
    "config": (
        "a configuration artifact that captures knobs and settings as declarative data",
        "Configuration as data",
        "It declares parameters (models, paths, hyperparameters, deployments) without "
        "behavior, so a run's intent is inspectable and diffable.",
    ),
    "infra": (
        "deployment/infrastructure glue that provisions, packages, or schedules compute",
        "Infrastructure & provisioning",
        "It drives a cloud/serverless/container substrate (Modal, GCP, vast, Docker) to "
        "stand up the environment a run needs.",
    ),
    "lib": (
        "a library module exposing reusable building blocks to the rest of the codebase",
        "Library module",
        "It defines types, helpers, and algorithms consumed by drivers and experiments "
        "rather than performing a single top-level action.",
    ),
    "pkg": (
        "a package marker that initializes or re-exports its package namespace",
        "Package initializer (`__init__.py`)",
        "Imports make the package importable, optionally re-exporting public API or "
        "running init-time setup.",
    ),
}

# role-concept -> (name, what)
ROLE_CONCEPTS = {
    "test": ("Why tests here are invariants, not runs",
             "Each test asserts a fact that must stay true (dispatch threads the right "
             "framework, plans point at real files) -- the closest CI can get to "
             "verifying a GPU experiment without one."),
    "train": ("Why the loop is GRPO and not full RLHF",
              "GRPO is the reward-model-free-online variant this protocol froze: it "
              "relies on group-relative advantage, cutting the value critic and memory."),
    "eval": ("Comparability over raw numbers",
             "Results only matter relative to a shared frozen protocol; evaluation "
             "exists to keep every framework measured against the same yardstick."),
    "bench": ("Abstraction isolating the variable",
              "A sweep must change one axis at a time so observed differences can be "
              "attributed -- the opposite of a kitchen-sink config."),
    "cli": ("One entry, many substrates",
            "Every backend eventually re-enters the local dispatch, so the CLI is both "
            "the human interface and the remote-on-box interface."),
    "experiment": ("Frozen protocol over flexibility",
                   "Experiments intentionally give up knob freedom in exchange for "
                   "equivalence -- comparability beats configurability here."),
    "config": ("Declarative contracts beat code constants",
               "Encoding settings as data (not literals) makes runs inspectable, "
               "diffable, and hashable -- prerequisites for the audit trail."),
    "infra": ("Provision-then-reenter pattern",
              "Remote backends rent a box, then run the same local dispatch on it -- "
              "one code path, many substrates."),
    "lib": ("DRY across drivers",
            "Shared helper modules stop five framework drivers from each re-solving the "
            "same problem in five slightly different ways."),
    "pkg": ("Explicit package boundaries",
            "An `__init__.py` documents what is public and runs any registry wiring, "
            "so importers depend on a stable API, not internals."),
}


# --- analyzers ---------------------------------------------------------------
def _fn_info(node) -> dict:
    return {
        "name": node.name,
        "async": isinstance(node, ast.AsyncFunctionDef),
        "decorators": [ast.unparse(d) for d in node.decorator_list] or [],
        "doc": (ast.get_docstring(node) or "").strip().replace("\n", " ")[:140],
    }


def analyze_python(rel: Path, text: str) -> dict:
    facts = {
        "kind": "config", "lines": len(text.splitlines()), "imports": [],
        "libs": set(), "functions": [], "classes": [], "patterns": set(),
        "docstring": None, "has_main": False, "decorators": [],
        "keys": [], "kind_note": None,
    }
    try:
        tree = ast.parse(text)
    except SyntaxError:
        facts["kind_note"] = "(could not parse as Python; structural facts omitted)"
        return facts

    if doc := ast.get_docstring(tree):
        facts["docstring"] = doc.strip().replace("\n", " ")[:200]

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                names = [node.module] if node.module else []
            for n in names:
                facts["imports"].append(n)
                for prefix, key in LIB_MAP:
                    if n == prefix or n.startswith(prefix + "."):
                        facts["libs"].add(key)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                s = ast.unparse(d)
                if "cache" in s or "lru_cache" in s:
                    facts["patterns"].add("memoization")
                if "dataclass" in s:
                    facts["patterns"].add("dataclass")
                if "abstractmethod" in s:
                    facts["patterns"].add("abc")
        elif isinstance(node, (ast.Yield,)) and "__main__" not in str(type(node)):
            facts["patterns"].add("generators")
        elif isinstance(node, ast.ExceptHandler):
            facts["patterns"].add("error_handling")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info = _fn_info(node)
            facts["functions"].append(info)
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Yield,)):
                    facts["patterns"].add("generators")
        elif isinstance(node, ast.ClassDef):
            methods = [_fn_info(m) for m in node.body
                       if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            bases = [ast.unparse(b) for b in node.bases] or []
            bstr = ",".join(bases)
            for d in node.decorator_list:
                if "dataclass" in ast.unparse(d):
                    facts["patterns"].add("dataclass")
            if any("ABC" in b for b in bases):
                facts["patterns"].add("abc")
            if any("Protocol" in b for b in bases):
                facts["patterns"].add("protocol")
            facts["classes"].append({
                "name": node.name, "bases": bases,
                "doc": (ast.get_docstring(node) or "").strip().replace("\n", " ")[:140],
                "methods": methods,
            })

    # main guard
    for node in tree.body:
        if (isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and "__name__" in ast.unparse(node.test)):
            facts["has_main"] = True

    facts["kind"] = "config" if rel.name.lower().startswith(("config", "settings")) else "py"
    return facts


def analyze_shell(rel: Path, text: str) -> dict:
    facts = {
        "kind": "sh", "lines": len(text.splitlines()), "imports": [],
        "libs": set(), "functions": [], "classes": [], "patterns": set(),
        "docstring": None, "has_main": False, "decorators": [],
        "keys": [], "kind_note": None, "commands": [], "shebang": None,
    }
    lines = text.splitlines()
    if lines and lines[0].startswith("#!"):
        facts["shebang"] = lines[0][2:].strip()
    header = []
    start = 1 if lines and lines[0].startswith("#!") else 0
    for ln in lines[start:]:
        if ln.startswith("#"):
            header.append(ln.lstrip("#").strip())
        elif not ln.strip():
            continue
        else:
            break
    if header:
        facts["docstring"] = header[0][:200]
    for m in re.finditer(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*\)\s*\{", text, re.M):
        facts["functions"].append({"name": m.group(1), "decorators": [], "doc": "", "async": False})
    for cmd in ("docker", "modal", "curl", "wget", "pip", "pip3", "git", "gh",
                "python", "python3", "uv", "bentoml", "deploy", "wandb",
                "huggingface", "gcloud"):
        if re.search(rf"(^|[^a-zA-Z])\b{cmd}\b", text):
            facts["commands"].append(cmd)
    facts["libs"].add("shell")
    return facts


def analyze_yaml_toml(rel: Path, text: str) -> dict:
    facts = {
        "kind": "config", "lines": len(text.splitlines()), "imports": [],
        "libs": set(), "functions": [], "classes": [], "patterns": set(),
        "docstring": None, "has_main": False, "decorators": [],
        "keys": [], "kind_note": None,
    }
    data = None
    if rel.suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(text)
            facts["libs"].add("yaml")
        except Exception:
            pass
    else:
        try:
            data = tomllib.loads(text)
        except Exception:
            pass
    if isinstance(data, dict):
        facts["keys"] = list(data.keys())
        keys = set(facts["keys"])
        if "jobs" in keys or "on" in keys:
            facts["kind_note"] = "CI workflow (GitHub Actions)"
        elif "services" in keys or "volumes" in keys:
            facts["kind_note"] = "container orchestration (docker-compose)"
        elif any(k in keys for k in ("model", "training", "peft", "data")):
            facts["kind_note"] = "experiment / training configuration"
        elif "repos" in keys or "hooks" in keys:
            facts["kind_note"] = "tool configuration"
        elif "dependencies" in keys or "channels" in keys:
            facts["kind_note"] = "environment manifest"
        else:
            facts["kind_note"] = "configuration"
        facts["patterns"].add("config")
    else:
        facts["kind_note"] = "structure could not be parsed"
    return facts


# --- role detection -----------------------------------------------------------
def detect_role(rel: Path, facts: dict) -> str:
    name = rel.name.lower()
    path = str(rel).lower()
    if rel.name == "__init__.py":
        return "pkg"
    if "test" in name or "tests/" in path.replace("\\", "/"):
        return "test"
    if any(k in name for k in ("bench", "sweep", "scale")):
        return "bench"
    if any(k in name for k in ("train", "grpo", "ppo", "rlhf", "idpo")):
        return "train"
    if any(k in name for k in ("eval", "verify", "check", "audit", "analyz", "measure")):
        return "eval"
    if any(k in name for k in ("run", "main", "cli", "launch", "entry", "serve")):
        return "cli"
    if facts["kind"] == "config":
        return "config"
    if any(k in name for k in ("deploy", "infra", "docker", "provision")):
        return "infra"
    if "experiment" in path:
        return "experiment"
    return "lib"


def analyze_json(rel: Path, text: str) -> dict:
    facts = {
        "kind": "config", "lines": len(text.splitlines()), "imports": [],
        "libs": set(), "functions": [], "classes": [], "patterns": set(),
        "docstring": None, "has_main": False, "decorators": [],
        "keys": [], "kind_note": None,
    }
    try:
        data = json.loads(text)
    except Exception:
        facts["kind_note"] = "JSON could not be parsed"
        return facts
    if isinstance(data, dict):
        facts["keys"] = list(data.keys())
        facts["kind_note"] = "JSON configuration / manifest"
        facts["patterns"].add("config")
    elif isinstance(data, list):
        facts["keys"] = []
        facts["kind_note"] = f"JSON array with {len(data)} entries"
        facts["patterns"].add("config")
    return facts


def analyze(rel: Path, text: str) -> dict:
    if rel.suffix == ".py":
        return analyze_python(rel, text)
    if rel.suffix == ".sh":
        return analyze_shell(rel, text)
    if rel.suffix == ".json":
        return analyze_json(rel, text)
    return analyze_yaml_toml(rel, text)


# --- rendering ---------------------------------------------------------------
def render_doc(rel: Path, facts: dict, role: str) -> str:
    lines: list[str] = []
    lines.append(f"# Deep Dive: `{rel}`")
    lines.append("")
    lines.append(f"> AntiVibe &middot; compact mode &middot; {TS} &middot; source: `{rel}` "
                 f"({facts['lines']} lines)")
    lines.append("")

    # Overview
    lines.append("## Overview")
    role_sentence, _rc_name, rc_what = ROLES[role]
    name = rel.name
    libs = sorted(facts["libs"])
    libs_txt = (", ".join(libs)) if libs else "the standard library only"
    lines.append(f"`{name}` is {role_sentence}. {rc_what.strip()}")
    lines.append(f"It leans on **{libs_txt}** to do its work.")
    if facts["kind_note"]:
        lines.append(f"*Kind:* {facts['kind_note']}")
    if facts["docstring"]:
        lines.append(f"*Self-description:* \"{facts['docstring'][:160]}\"")
    if role == "pkg":
        lines.append("As a package init, it defines the *namespace* more than the behavior.")
    lines.append("")

    # Key Components
    lines.append("## Key Components")
    items = []
    for c in facts["classes"]:
        methods = ", ".join(m["name"] for m in c["methods"][:8])
        detail = f" ({len(c['methods'])} methods: {methods})" if methods else ""
        base = f" (bases: {', '.join(c['bases'][:3])})" if c["bases"] else ""
        items.append(f"- `{c['name']}`{base} -- {c['doc'] or 'class'}{detail}")
    for f in facts["functions"][:12]:
        deco = f" [{', '.join(f['decorators'][:3])}]" if f["decorators"] else ""
        items.append(f"- `{f['name']}()`{deco} -- {f['doc'] or 'function'}")
    if facts["keys"]:
        items.append("- Top-level keys: `" + "`, `".join(str(k) for k in facts["keys"][:14]) + "`")
    if facts.get("commands"):
        cmds = sorted(facts.get("commands", []))
        items.append("- Shell tools invoked: `" + "`, `".join(cmds) + "`")
    if facts.get("shebang"):
        items.append(f"- Shebang: `{facts.get('shebang')}`")
    if facts["has_main"]:
        items.append("- Has a `if __name__ == \"__main__\"` entry point")
    lines.extend(items if items else ["- _(no top-level components detected)_"])
    lines.append("")

    # Concepts
    lines.append("## Concepts & Decisions")
    lines.append(f"### {ROLE_CONCEPTS[role][0]}")
    lines.append(f"- **What**: {ROLE_CONCEPTS[role][1]}")
    lines.append("")

    shown = set()
    for key in list(facts["patterns"]) + list(facts["libs"]):
        key = "memoization" if key == "memoization" else key
        if key in CONCEPTS and key not in shown:
            shown.add(key)
            name_, what, why, when, trade = CONCEPTS[key]
            lines.append(f"### {name_}")
            lines.append(f"- **What**: {what}")
            lines.append(f"- **Why used here**: {why}")
            lines.append(f"- **When**: {when}")
            lines.append(f"- **Trade-offs**: {trade}")
            lines.append("")
    if not shown:
        lines.append("- _(no notable library concepts detected)_")
    lines.append("")

    # Related code
    lines.append("## Related Code")
    rels = related_files(rel, facts)
    lines.extend(rels if rels else ["- Stand-alone: imports nothing else local."])
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by AntiVibe per-file pass &middot; {TS} &middot; run "
                 f"`/antivibe` (or the antivibe skill) on this file for a full-mode "
                 f"drill-down.*")
    return "\n".join(lines).rstrip() + "\n"


def related_files(rel: Path, facts: dict) -> list[str]:
    out = []
    seen = set()
    for imp in facts["imports"]:
        cand = ROOT / (imp.replace(".", "/") + ".py")
        if cand.exists():
            r = cand.relative_to(ROOT)
            if str(r) != str(rel):
                out.append(f"- `{imp}` &rarr; local `{r}`")
                seen.add(str(r))
    # siblings in the same directory (up to 6)
    sib_dir = ROOT / rel.parent if str(rel.parent) != "." else ROOT
    sib_count = 0
    for child in sorted(sib_dir.iterdir()):
        if child.suffix.lower() in INCLUDE_EXTS and child.name != rel.name:
            r = child.relative_to(ROOT)
            if str(r) not in seen and sib_count < 6:
                out.append(f"- sibling `{r}`")
                seen.add(str(r))
                sib_count += 1
    return out


def output_path(rel: Path, used: dict[Path, Path]) -> Path:
    base = rel.with_suffix(".md")
    if base not in used:
        used[base] = rel
        return OUT_ROOT / base
    # extension collision (e.g. foo.py vs foo.sh): disambiguate
    return OUT_ROOT / rel.with_suffix(f".{rel.suffix.lstrip('.')}.md")


def write_index(index_rows: list[tuple[str, str, str, str]]) -> None:
    idx = OUT_ROOT / "INDEX.md"
    header = [
        "# AntiVibe Per-File Deep Dives -- Index",
        "",
        f"> Generated {TS} by `tools/apply_antivibe.py` (AntiVibe compact mode).",
        "",
        "One educational deep dive per real source file, mirrored under this folder.",
        "Scope: Python (.py), shell (.sh), YAML/TOML, and key top-level JSON. Excluded:",
        "git metadata, virtualenvs, caches, data/lockfiles, generated artifacts",
        "(`output/`, `outputs/`), and the `.claude/worktrees/` checkouts (duplicate",
        "copies of this same codebase at other branches).",
        "",
        "Start with the repo-level dive: `../REPO-OVERVIEW-2026-08-02.md`, then use the",
        "module index below to jump to any file. Strip out-of-date docs and regenerate",
        "with: `python tools/apply_antivibe.py`.",
        "",
        "## Files by module",
        "",
    ]
    lines = list(header)
    current = None
    for rel, out, role, nlines in index_rows:
        relp = Path(rel)
        mod = relp.parts[0] if len(relp.parts) > 1 else "(root)"
        if mod != current:
            current = mod
            lines.append(f"### `{mod}`")
            lines.append("")
            lines.append("| Source | Deep dive | Role | Lines |")
            lines.append("|--------|-----------|------|------:|")
        link = Path(out).relative_to(OUT_ROOT)
        lines.append(f"| `{rel}` | [{link}]({link.as_posix()}) | {role} | {nlines} |")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by AntiVibe per-file pass.*")
    (OUT_ROOT / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    files = walk_source_files()
    os.makedirs(OUT_ROOT, exist_ok=True)
    used: dict[Path, Path] = {}
    index_rows: list[tuple[str, str, str, str]] = []
    written = skipped = 0
    role_counts: dict[str, int] = {}
    for rel in files:
        try:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            skipped += 1
            continue
        facts = analyze(rel, text)
        role = detect_role(rel, facts)
        role_counts[role] = role_counts.get(role, 0) + 1
        out = output_path(rel, used)
        out.parent.mkdir(parents=True, exist_ok=True)
        doc = render_doc(rel, facts, role)
        out.write_text(doc, encoding="utf-8")
        index_rows.append((str(rel), str(out), role, str(facts["lines"])))
        written += 1
    write_index(index_rows)

    print(f"AntiVibe per-file pass complete.")
    print(f"  scanned files : {len(files)}")
    print(f"  docs written  : {written}")
    print(f"  skipped (empty): {skipped}")
    print(f"  output dir    : {OUT_ROOT}")
    print("  by role       : " + ", ".join(f"{k}={v}" for k, v in sorted(role_counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

