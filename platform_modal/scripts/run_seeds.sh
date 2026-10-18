#!/bin/bash
# Protocol v2: run_seeds.sh [--seed N ...] -- executable [argument ...]
# Command arguments are never reparsed as shell syntax.
set -euo pipefail
SEEDS=()
while (( $# )); do
    case "$1" in
        --seed) [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo 'Invalid seed' >&2; exit 2; }; SEEDS+=("$2"); shift 2 ;;
        --) shift; break ;;
        *) echo 'Usage: run_seeds.sh [--seed N ...] -- executable [arguments ...]' >&2; exit 2 ;;
    esac
done
(( $# )) || { echo 'Missing command' >&2; exit 2; }
(( ${#SEEDS[@]} )) || SEEDS=(42 123 456 789 1024 2048 4096 8192 16384 32768)
COMMAND=("$@")
FAILED=0
for SEED in "${SEEDS[@]}"; do
    START=$(date -u +%FT%TZ)
    if "${COMMAND[@]}" --seed "$SEED"; then STATUS=0; else STATUS=$?; FAILED=$((FAILED+1)); fi
    printf 'seed=%s\tstart=%s\tend=%s\texit_status=%s\tcommand=' "$SEED" "$START" "$(date -u +%FT%TZ)" "$STATUS"
    printf '%q ' "${COMMAND[@]}" --seed "$SEED"
    printf '\n'
done
(( FAILED == 0 ))
