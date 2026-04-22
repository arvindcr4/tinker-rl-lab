"""
Push a HF model card + eval report after WebArena-verified eval.

Reads the aggregated JSON produced by aggregate.py plus the per-shard JSONL
files, and creates TWO HuggingFace Hub repositories:

  1. DATASET repo: raw per-shard JSONL + aggregated final.json
     (e.g. arvindcr4/webarena-results-20260422)

  2. MODEL repo: README.md model card pointing at the base model, with eval
     results, methodology, and a link to the dataset repo above
     (e.g. arvindcr4/webarena-eval-qwen3-8b-20260422)

Needs HF_TOKEN env var set.

Usage:
    python -m experiments.webarena.push_model_card \\
        --final results/final.json \\
        --shards 'results/*.jsonl' \\
        --base-model Qwen/Qwen3-8B \\
        --dataset-repo arvindcr4/webarena-results-20260422 \\
        --model-repo  arvindcr4/webarena-eval-qwen3-8b-20260422 \\
        --benchmark webarena_verified --private
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from glob import glob
from pathlib import Path
from textwrap import dedent


def _render_model_card(
    *, base_model: str, benchmark: str, final: dict, dataset_repo: str, run_id: str,
) -> str:
    overall = final.get("overall", {})
    by_bench = final.get("by_benchmark", {})
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # YAML frontmatter: tags, license, datasets, metrics, model-index
    frontmatter = dedent(f"""\
        ---
        license: other
        base_model: {base_model}
        datasets:
          - {dataset_repo}
        tags:
          - browsergym
          - webarena
          - evaluation
          - agent
          - react
        pipeline_tag: text-generation
        library_name: transformers
        model-index:
          - name: {base_model} on {benchmark}
            results:
              - task:
                  type: web-navigation
                  name: BrowserGym / {benchmark}
                dataset:
                  type: webarena
                  name: {benchmark}
                metrics:
                  - type: success_rate
                    value: {overall.get("success_rate", 0.0):.4f}
                    name: Success Rate
                  - type: mean_reward
                    value: {overall.get("mean_reward", 0.0):.4f}
                    name: Mean Reward
        ---
        """)

    # Human prose
    tbl_rows = []
    for name, s in sorted(by_bench.items()):
        tbl_rows.append(
            f"| {name} | {s['n']} | {s['success_rate']:.3f} | {s['mean_reward']:.3f} | "
            f"{s['mean_num_steps']:.1f} | {s['mean_valid_actions']:.1f} | {s['episodes_with_error']} |"
        )
    tbl = "\n".join(tbl_rows) if tbl_rows else "| — | 0 | 0.000 | 0.000 | 0 | 0 | 0 |"

    body = dedent(f"""\
        # {base_model} — BrowserGym {benchmark} evaluation

        Eval of `{base_model}` (base, no fine-tuning) on BrowserGym's `{benchmark}`
        benchmark using a true multi-turn ReAct loop. Evaluated on {today}.

        - **Run ID:** `{run_id}`
        - **Success rate:** **{overall.get("success_rate", 0.0) * 100:.2f}%**
        - **Mean reward:** {overall.get("mean_reward", 0.0):.4f}
        - **Episodes:** {overall.get("n", 0)} ({overall.get("episodes_with_error", 0)} env errors)
        - **Mean steps / episode:** {overall.get("mean_num_steps", 0.0):.1f}
        - **Mean valid actions / episode:** {overall.get("mean_valid_actions", 0.0):.1f}

        ## Results by benchmark prefix

        | Benchmark | N | Success Rate | Mean Reward | Mean Steps | Mean Valid Actions | Env Errors |
        |---|---:|---:|---:|---:|---:|---:|
        {tbl}

        ## Methodology

        - **Agent architecture.** Single-turn ReAct over BrowserGym's HighLevelActionSet.
          Each step: feed `axtree_txt` + last-action error back to the model; sample one
          action with `Thought:`/`Action:` format; execute with `env.step`; repeat until
          terminated/truncated or `max_steps` reached.
        - **Observation.** AXTree, capped at 24k chars (head-and-tail truncation beyond).
        - **Sampling.** Tinker `SamplingClient` (`sample_async`), T=0 greedy, 256-token
          action budget.
        - **Reward.** Task-terminal reward from BrowserGym's `_task_validate`, taken as
          `max(total_reward_over_episode, max_reward_over_episode)` clipped to `[0,1]`.
        - **Parallelism.** 10-worker Managed Instance Group on GCP, each worker runs an
          isolated WebArena docker stack with 5 concurrent Chromiums (50-way).
        - **No cross-task interference.** One WebArena stack per VM.

        ## Reproduction

        ```sh
        git clone https://github.com/arvindcr4/tinker-rl-lab
        cd tinker-rl-lab

        # 1. Build GCP image with WebArena dockers preloaded (~60 min, one-time)
        GCP_PROJECT=<your-project> ./infra/gcp/build_webarena_image.sh

        # 2. Deploy 10-VM Managed Instance Group
        GCP_PROJECT=<your-project> BUCKET=<your-bucket> NUM_WORKERS=10 \\
            ./infra/gcp/deploy_mig.sh

        # 3. Aggregate
        gsutil cp 'gs://<your-bucket>/<RUN_ID>/*.jsonl' results/
        python -m experiments.webarena.aggregate \\
            --inputs 'results/*.jsonl' --out results/final.json
        ```

        Full raw per-shard results are in the dataset repo
        [`{dataset_repo}`](https://huggingface.co/datasets/{dataset_repo}).

        ## Intended use

        This repo holds an **eval report for a base model**, not a fine-tune. The base
        model weights are those of `{base_model}` — see that repo for the model itself.

        ## Limitations

        - Scores on WebArena are sensitive to docker state drift between runs. Each VM
          boots from a clean snapshot to mitigate this, but cross-run numbers are only
          directly comparable when the docker image hashes match.
        - The map site uses a public endpoint, not a self-hosted OSM tile server — map
          tasks may be slightly lossier.
        - No training; this is inference only on the released base model.
        """)
    return frontmatter + "\n" + body


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--final", required=True, help="final.json from aggregate.py")
    p.add_argument("--shards", required=True,
                   help="Glob for per-shard JSONLs (e.g. 'results/*.jsonl')")
    p.add_argument("--base-model", required=True, help="e.g. Qwen/Qwen3-8B")
    p.add_argument("--dataset-repo", required=True,
                   help="HF dataset repo to push JSONLs + final.json (user/name)")
    p.add_argument("--model-repo", required=True,
                   help="HF model repo to push the eval card README (user/name)")
    p.add_argument("--benchmark", default="webarena_verified")
    p.add_argument("--run-id", default=datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--private", action="store_true",
                   help="Create both repos as private")
    args = p.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN not set", file=__import__("sys").stderr)
        return 2

    from huggingface_hub import HfApi, create_repo
    api = HfApi()

    final = json.loads(Path(args.final).read_text())

    # -------------------------------------------------------------------------
    # 1) Dataset repo: raw JSONL shards + aggregated final.json
    # -------------------------------------------------------------------------
    print(f"==> Dataset repo: {args.dataset_repo}")
    create_repo(args.dataset_repo, repo_type="dataset",
                exist_ok=True, private=args.private)
    for shard in sorted(glob(args.shards)):
        remote_name = f"{args.run_id}/{Path(shard).name}"
        print(f"   -> {remote_name}")
        api.upload_file(
            path_or_fileobj=shard,
            path_in_repo=remote_name,
            repo_id=args.dataset_repo,
            repo_type="dataset",
            commit_message=f"add shard {Path(shard).name} (run {args.run_id})",
        )
    api.upload_file(
        path_or_fileobj=args.final,
        path_in_repo=f"{args.run_id}/final.json",
        repo_id=args.dataset_repo,
        repo_type="dataset",
        commit_message=f"add aggregated final.json (run {args.run_id})",
    )

    # Write a dataset README if not already present
    ds_readme = dedent(f"""\
        ---
        license: other
        tags:
          - webarena
          - browsergym
          - evaluation
        ---

        # WebArena eval shards

        Per-shard raw `results_shard_k.jsonl` + aggregated `final.json` from WebArena
        parallel eval runs of `{args.base_model}`.

        - **Runs:** grouped by `run_id` (UTC timestamp).
        - **Schema (JSONL):** one episode per line, fields = `env_id`, `seed`, `score`,
          `num_steps`, `valid_action_count`, `steps[]`, `wall_time_sec`, `error`.

        Model card (with methodology and scores): [`{args.model_repo}`](https://huggingface.co/{args.model_repo}).
        """)
    api.upload_file(
        path_or_fileobj=ds_readme.encode(),
        path_in_repo="README.md",
        repo_id=args.dataset_repo,
        repo_type="dataset",
        commit_message="update dataset README",
    )

    # -------------------------------------------------------------------------
    # 2) Model repo: README.md with eval report + results table
    # -------------------------------------------------------------------------
    print(f"==> Model repo:   {args.model_repo}")
    create_repo(args.model_repo, repo_type="model",
                exist_ok=True, private=args.private)
    card_md = _render_model_card(
        base_model=args.base_model, benchmark=args.benchmark,
        final=final, dataset_repo=args.dataset_repo, run_id=args.run_id,
    )
    api.upload_file(
        path_or_fileobj=card_md.encode(),
        path_in_repo="README.md",
        repo_id=args.model_repo,
        repo_type="model",
        commit_message=f"eval report: {args.benchmark} run {args.run_id}",
    )
    # Also embed the final.json for machine-readable access
    api.upload_file(
        path_or_fileobj=args.final,
        path_in_repo="eval_final.json",
        repo_id=args.model_repo,
        repo_type="model",
        commit_message=f"eval metrics json: run {args.run_id}",
    )

    print("")
    print(f"Dataset:  https://huggingface.co/datasets/{args.dataset_repo}")
    print(f"Model:    https://huggingface.co/{args.model_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
