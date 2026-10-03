#!/usr/bin/env bash
#
# ralph.sh — Ralph Loop outer driver for unattended `muse` runs.
#
# Each iteration runs ONE fresh headless session (no context bloat), then a
# verify gate: pass -> commit, fail -> stash (recoverable), repeat.
#
# Usage:
#   tools/harness/ralph.sh [--max-iters N] [--cycle-timeout SEC]
#                          [--max-steps N] [--effort E] [--verify CMD]
#                          [--task-file F] [--workdir D] [--allow-dirty]
#                          [--dry-run]
#
# Env overrides: MAX_ITERS CYCLE_TIMEOUT MAX_STEPS EFFORT VERIFY_CMD
#                TASK_FILE WORKDIR
# Stop a run:    touch tools/harness/STOP   (checked between iterations)
set -uo pipefail

HARNESS="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HARNESS/../.." && pwd)"
WORKDIR="${WORKDIR:-$ROOT}"
MAX_ITERS="${MAX_ITERS:-0}"            # 0 = unlimited, stop via STOP file
CYCLE_TIMEOUT="${CYCLE_TIMEOUT:-1800}" # wall clock per iteration (s)
MAX_STEPS="${MAX_STEPS:-60}"           # --max-model-steps per iteration
EFFORT="${EFFORT:-medium}"
VERIFY_CMD="${VERIFY_CMD:-make lint-ruff && make test}"
TASK_FILE="${TASK_FILE:-backlog.md}"
DRY_RUN=0
ALLOW_DIRTY=0
STOP_FILE="$HARNESS/STOP"
STATE_FILE="$HARNESS/.ralph-state"
LOG_DIR="$HARNESS/logs"
MAX_CONSEC_FAIL=3

while [ $# -gt 0 ]; do
  case "$1" in
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    --cycle-timeout) CYCLE_TIMEOUT="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --effort) EFFORT="$2"; shift 2 ;;
    --verify) VERIFY_CMD="$2"; shift 2 ;;
    --task-file) TASK_FILE="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

command -v muse >/dev/null || { echo "error: muse CLI not found" >&2; exit 1; }
if command -v timeout >/dev/null; then TIMEOUT_BIN=timeout;
elif command -v gtimeout >/dev/null; then TIMEOUT_BIN=gtimeout;
else echo "error: need GNU timeout (brew install coreutils)" >&2; exit 1; fi
[ -d "$WORKDIR/.git" ] || [ -f "$WORKDIR/.git" ] || { echo "error: $WORKDIR is not a git checkout" >&2; exit 1; }
if [ "$DRY_RUN" = 0 ] && [ "$ALLOW_DIRTY" = 0 ] && [ -n "$(git -C "$WORKDIR" status --porcelain)" ]; then
  echo "error: dirty tree — commit/stash first or pass --allow-dirty" >&2; exit 1
fi

mkdir -p "$LOG_DIR"
ITER=0; CONSEC_FAIL=0
# shellcheck source=/dev/null
[ -f "$STATE_FILE" ] && . "$STATE_FILE"
save_state() { printf 'ITER=%s\nCONSEC_FAIL=%s\n' "$ITER" "$CONSEC_FAIL" > "$STATE_FILE"; }

log() { printf '[ralph %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

esc() { printf '%s' "$1" | sed -e 's/[&|\]/\\&/g'; }
build_prompt() {
  sed -e "s|{{ITER}}|$(esc "$ITER")|g" -e "s|{{TASK_FILE}}|$(esc "$TASK_FILE")|g" \
      -e "s|{{WORKDIR}}|$(esc "$WORKDIR")|g" -e "s|{{VERIFY_CMD}}|$(esc "$VERIFY_CMD")|g" \
      "$HARNESS/PROMPT.md"
}

log "workdir=$WORKDIR max_iters=$MAX_ITERS timeout=${CYCLE_TIMEOUT}s verify=[$VERIFY_CMD]"
[ "$DRY_RUN" = 1 ] && { log "DRY RUN — prompt for iter $((ITER + 1)):"; ITER=$((ITER + 1)); build_prompt; exit 0; }

while :; do
  [ -f "$STOP_FILE" ] && { log "STOP file present, exiting"; rm -f "$STOP_FILE"; break; }
  [ "$MAX_ITERS" -gt 0 ] && [ "$ITER" -ge "$MAX_ITERS" ] && { log "max iters reached"; break; }
  ITER=$((ITER + 1))
  TS="$(date +%Y%m%d-%H%M%S)"
  JSONL="$LOG_DIR/ralph-${ITER}-${TS}.jsonl"

  log "=== iter $ITER: muse exec ==="
  # --yolo disables approval prompts (required unattended); workspace is
  # already trusted by explicit user setup. Never add push rights here.
  PROMPT_TMP="$(mktemp -t ralph-prompt.XXXXXX)"
  build_prompt > "$PROMPT_TMP"
  set +e
  $TIMEOUT_BIN "$CYCLE_TIMEOUT" muse exec --json --yolo \
      --workspace "$WORKDIR" --max-model-steps "$MAX_STEPS" \
      --reasoning-effort "$EFFORT" --prompt-file "$PROMPT_TMP" >"$JSONL" 2>&1
  rc=$?
  set -e
  rm -f "$PROMPT_TMP"

  if [ $rc -eq 124 ]; then
    log "iter $ITER TIMEOUT after ${CYCLE_TIMEOUT}s"
  elif [ $rc -ne 0 ]; then
    log "iter $ITER muse exited rc=$rc (see $JSONL)"
  fi

  if [ -z "$(git -C "$WORKDIR" status --porcelain)" ]; then
    log "iter $ITER: no changes"
    [ $rc -eq 0 ] && CONSEC_FAIL=0 || CONSEC_FAIL=$((CONSEC_FAIL + 1))
  elif [ $rc -ne 0 ]; then
    git -C "$WORKDIR" stash push -u -m "ralph-failed-iter-${ITER}-${TS}" >/dev/null
    log "iter $ITER: exec failed, changes stashed"
    CONSEC_FAIL=$((CONSEC_FAIL + 1))
  else
    log "iter $ITER: verifying [$VERIFY_CMD]"
    if (cd "$WORKDIR" && eval "$VERIFY_CMD" >/dev/null 2>&1); then
      git -C "$WORKDIR" add -A
      git -C "$WORKDIR" commit -qm "chore(ralph): cycle $ITER $TS" --no-verify
      log "iter $ITER: PASS, committed"
      CONSEC_FAIL=0
    else
      git -C "$WORKDIR" stash push -u -m "ralph-failed-iter-${ITER}-${TS}" >/dev/null
      log "iter $ITER: VERIFY FAILED, changes stashed (recover: git stash list)"
      CONSEC_FAIL=$((CONSEC_FAIL + 1))
    fi
  fi

  save_state
  if [ "$CONSEC_FAIL" -ge "$MAX_CONSEC_FAIL" ]; then
    log "$MAX_CONSEC_FAIL consecutive failures, stopping for human review"
    exit 1
  fi
done
log "done after $ITER iterations"
