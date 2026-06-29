#!/bin/bash
# Auto-generated exhaustive hyperparameter sweep script
set -euo pipefail

echo 'Starting exhaustive hyperparameter sweep...'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs64_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs64_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs64_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs64_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs64_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs64_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs128_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs128_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs128_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs128_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs128_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs128_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs256_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs256_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs256_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs256_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs256_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora8_bs256_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs64_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs64_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs64_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs64_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs64_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs64_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs128_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs128_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs128_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs128_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs128_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs128_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs256_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs256_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs256_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs256_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs256_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora16_bs256_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs64_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs64_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs64_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs64_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs64_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs64_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs128_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs128_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs128_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs128_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs128_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs128_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs256_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs256_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs256_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs256_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs256_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora32_bs256_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs64_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs64_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs64_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs64_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs64_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs64_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs128_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs128_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs128_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs128_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs128_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs128_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs256_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs256_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs256_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs256_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs256_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr1e-05_lora64_bs256_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs64_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs64_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs64_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs64_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs64_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs64_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs128_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs128_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs128_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs128_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs128_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs128_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs256_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs256_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs256_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs256_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs256_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora8_bs256_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs64_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs64_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs64_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs64_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs64_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs64_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs128_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs128_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs128_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs128_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs128_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs128_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs256_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs256_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs256_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs256_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs256_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora16_bs256_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs64_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs64_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs64_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs64_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs64_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs64_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs128_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs128_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs128_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs128_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs128_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs128_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs256_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs256_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs256_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs256_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs256_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora32_bs256_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs64_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs64_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs64_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs64_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs64_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs64_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs128_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs128_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs128_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs128_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs128_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs128_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs256_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs256_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs256_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs256_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs256_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr3e-05_lora64_bs256_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs64_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs64_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs64_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs64_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs64_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs64_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs128_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs128_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs128_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs128_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs128_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs128_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs256_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs256_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs256_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs256_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs256_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora8_bs256_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs64_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs64_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs64_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs64_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs64_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs64_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs128_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs128_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs128_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs128_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs128_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs128_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs256_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs256_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs256_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs256_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs256_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora16_bs256_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs64_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs64_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs64_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs64_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs64_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs64_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs128_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs128_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs128_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs128_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs128_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs128_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs256_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs256_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs256_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs256_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs256_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora32_bs256_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs64_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs64_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs64_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs64_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs64_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs64_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs128_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs128_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs128_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs128_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs128_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs128_gs32.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs256_gs8.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs256_gs8.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs256_gs16.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs256_gs16.yaml --seed 42
echo '---------------------------------------------------'

echo 'Running: python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs256_gs32.yaml --seed 42'
python3 atropos/train_grpo_unsloth.py --config atropos/configs/sweep_results/config_lr0.0001_lora64_bs256_gs32.yaml --seed 42
echo '---------------------------------------------------'

