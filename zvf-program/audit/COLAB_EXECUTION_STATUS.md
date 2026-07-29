# Colab execution status

Date: 2026-07-20
Backend: Google Colab CLI 0.6.0, OAuth2 account `arvindcr4@gmail.com`
Confirmed accelerator: NVIDIA A100-SXM4-40GB

## Frozen E1 campaign: complete (40/40 independently validated)

All forty confirmatory units finished scientifically: all eight GRPO seeds (11, 23, 37, 53,
71, 89, 107, and 131), all eight DAPO seeds, GSPO seeds
11, 23, 37, 53, 71, 89, 107, and 131, all eight Dr.GRPO seeds, and
AERO seeds 11, 23, 37, 53, 71, 89, 107, and 131 on
the pinned `Qwen/Qwen3-8B` and `openai/gsm8k` revisions. Every unit used the
frozen 30 optimizer steps and 500 held-out examples; the GRPO and GSPO units
sampled 480 training completions each, while DAPO's preregistered dynamic
sampling realized 1,824, 2,112, 1,472, 1,664, 1,648, 1,584, 1,840, and 1,728 rollouts. All used LoRA
on a single Colab A100 and the package lock recorded in
`preregistration_colab_a100_amendment.json`. A final campaign-wide integrity
pass then found six legacy manifests that predated mandatory completion hashes:
GRPO seeds 11, 89, and 107; DAPO seed 131; and GSPO seeds 11 and 71. Their
checkpoint-30 model state, finished W&B runs, and prior manifests remain
preserved, but they did not count as accepted evidence until repaired.
Exact-checkpoint evaluation-only replay repaired all six records. DAPO seed 131 is
repaired and independently re-verified at exact private commit
`39f916902470b3b800af5c8d60d398a164cd2b95`; GRPO seed 89 is also repaired at
exact private commit `f88da35f8939dc7ff74ed0b37a004fa8c78379a8`; GRPO seed 107
passed at exact private commit `ffcbfffd322a181e82c0d3a552ec611432dce471`;
and GRPO seed 11 passed at exact private commit
`b39028ea32b042247e7ecd3ee228b8b302c55226`.
GSPO seed 11 passed at exact private commit
`f064274b7c79372fe3fa1501737ae8a5398bec07`. GSPO seed 71 then resumed
from its exact checkpoint-30 source and verified 416-row private ledger,
completed all 500 held-out rows, and passed at exact private commit
`c0c0b968a61a9ade251b5b7e6ece3119197dc1b1` with 319 correct (`0.6380`),
500 valid unique completion hashes, and byte-identical manifest SHA-256
`0243e256a3cdae62d30cad889f8bdf19bfe6d7cc2862edc60edffc23586ac3ed`.
The campaign verifier reports `COMPLETE`: 40 locally validated and 40 remotely
verified units with zero errors. The frozen aggregate reports `COMPLETE` and
emits DAPO `DISAPPEARS`; GSPO, Dr.GRPO, and AERO `INCONCLUSIVE`. All Colab
sessions are released and no campaign or recovery process remains active.

| Arm | Seed | Held-out exact match | Last-10 reward | Mean ZVF | Mean GU | Collapse | Effective time |
|---|---:|---:|---:|---:|---:|---|---:|
| GRPO | 11 | 325/500 = 0.6500 | 0.6875 | 0.7000 | 0.3000 | false | 16,356.13 s |
| GRPO | 23 | 318/500 = 0.6360 | 0.7375 | 0.7167 | 0.2833 | false | 10,449.86 s |
| GRPO | 37 | 317/500 = 0.6340 | 0.6250 | 0.6417 | 0.3583 | false | 10,121.13 s |
| GRPO | 53 | 317/500 = 0.6340 | 0.6250 | 0.6917 | 0.3083 | false | 9,973.88 s |
| GRPO | 71 | 317/500 = 0.6340 | 0.6438 | 0.7250 | 0.2750 | false | 10,256.65 s |
| GRPO | 89 | 310/500 = 0.6200 | 0.7438 | 0.6417 | 0.3583 | false | 12,572.20 s |
| GRPO | 107 | 317/500 = 0.6340 | 0.6500 | 0.7667 | 0.2333 | false | 12,839.05 s |
| GRPO | 131 | 311/500 = 0.6220 | 0.6500 | 0.6583 | 0.3417 | false | 6,895.43 s |
| DAPO | 11 | 324/500 = 0.6480 | 0.5250 | 0.0000 | 1.0000 | false | 20,236.89 s |
| DAPO | 23 | 312/500 = 0.6240 | 0.5500 | 0.0000 | 1.0000 | false | 11,717.90 s |
| DAPO | 37 | 314/500 = 0.6280 | 0.5437 | 0.0000 | 1.0000 | false | 14,477.75 s |
| DAPO | 53 | 318/500 = 0.6360 | 0.5188 | 0.0000 | 1.0000 | false | 18,206.64 s |
| DAPO | 71 | 317/500 = 0.6340 | 0.5813 | 0.0000 | 1.0000 | false | 14,380.70 s |
| DAPO | 89 | 318/500 = 0.6360 | 0.5500 | 0.0000 | 1.0000 | false | 18,362.96 s |
| DAPO | 107 | 317/500 = 0.6340 | 0.4938 | 0.0000 | 1.0000 | false | 10,327.54 s |
| DAPO | 131 | 316/500 = 0.6320 | 0.6125 | 0.0000 | 1.0000 | false | 21,058.36 s |
| GSPO | 11 | 320/500 = 0.6400 | 0.6938 | 0.7083 | 0.2917 | false | 14,823.76 s |
| GSPO | 23 | 319/500 = 0.6380 | 0.7313 | 0.6750 | 0.3250 | false | 10,210.87 s |
| GSPO | 37 | 321/500 = 0.6420 | 0.6312 | 0.6833 | 0.3167 | false | 10,652.61 s |
| GSPO | 53 | 316/500 = 0.6320 | 0.6500 | 0.6833 | 0.3167 | false | 10,147.75 s |
| GSPO | 71 | 319/500 = 0.6380 | 0.6500 | 0.7417 | 0.2583 | false | 9,686.38 s |
| GSPO | 89 | 323/500 = 0.6460 | 0.7438 | 0.6417 | 0.3583 | false | 10,407.25 s |
| GSPO | 107 | 320/500 = 0.6400 | 0.6688 | 0.7250 | 0.2750 | false | 10,521.18 s |
| GSPO | 131 | 314/500 = 0.6280 | 0.6563 | 0.6833 | 0.3167 | false | 10,405.58 s |
| Dr.GRPO | 11 | 321/500 = 0.6420 | 0.6688 | 0.6750 | 0.3250 | false | 7,223.15 s |
| Dr.GRPO | 23 | 313/500 = 0.6260 | 0.7750 | 0.7250 | 0.2750 | false | 10,412.68 s |
| Dr.GRPO | 37 | 311/500 = 0.6220 | 0.6688 | 0.6917 | 0.3083 | false | 10,403.87 s |
| Dr.GRPO | 53 | 318/500 = 0.6360 | 0.6500 | 0.6833 | 0.3167 | false | 10,272.23 s |
| Dr.GRPO | 71 | 310/500 = 0.6200 | 0.6688 | 0.6667 | 0.3333 | false | 9,993.44 s |
| Dr.GRPO | 89 | 314/500 = 0.6280 | 0.7250 | 0.7250 | 0.2750 | false | 10,167.33 s |
| Dr.GRPO | 107 | 314/500 = 0.6280 | 0.6375 | 0.7833 | 0.2167 | false | 10,293.05 s |
| Dr.GRPO | 131 | 323/500 = 0.6460 | 0.6438 | 0.7417 | 0.2583 | false | 10,209.47 s |
| AERO | 11 | 315/500 = 0.6300 | 0.6762 | 0.6667 | 0.3333 | false | 14,297.81 s |
| AERO | 23 | 314/500 = 0.6280 | 0.7236 | 0.7667 | 0.2333 | false | 14,201.01 s |
| AERO | 37 | 318/500 = 0.6360 | 0.6383 | 0.7167 | 0.2833 | false | 22,429.32 s |
| AERO | 53 | 316/500 = 0.6320 | 0.6556 | 0.6333 | 0.3667 | false | 12,146.46 s |
| AERO | 71 | 316/500 = 0.6320 | 0.6856 | 0.6917 | 0.3083 | false | 9,412.92 s |
| AERO | 89 | 315/500 = 0.6300 | 0.7115 | 0.6500 | 0.3500 | false | 21,316.57 s |
| AERO | 107 | 315/500 = 0.6300 | 0.6842 | 0.7917 | 0.2083 | false | 8,647.25 s |
| AERO | 131 | 320/500 = 0.6400 | 0.7026 | 0.6833 | 0.3167 | false | 14,319.88 s |

Effective time is the accepted audit record's scientific-path wall clock. For
recovered GRPO units it combines the retained training phase with the successful
resumable evaluation; it excludes launcher setup and abandoned attempts, so it
is not an estimate of total Colab credits consumed.

Remote evidence:

- Seed 11: finished W&B run [`78393a67`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/78393a67) and repaired private HF commit [`b39028ea32b0`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-grpo-s11-78393a67/tree/b39028ea32b042247e7ecd3ee228b8b302c55226).
- Seed 23: finished W&B run [`0057b069`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/0057b069) and private HF commit [`de7dc778c412`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-grpo-s23-0057b069/tree/de7dc778c41231ddd5b2cff77f127f54746040f0).
- Seed 37: finished W&B run [`54592080`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/54592080) and private HF commit [`2755e14f26f2`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-grpo-s37-54592080/tree/2755e14f26f2712480893285bcbb1bceb5315053).
- Seed 53: finished W&B run [`c13bbc2b`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/c13bbc2b) and private HF commit [`0a4145834057`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-grpo-s53-c13bbc2b/tree/0a41458340579fe11166157338e8990a3567c801).
- Seed 71: finished W&B run [`f6d81075`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/f6d81075) and private HF commit [`821fa6b4504a`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-grpo-s71-f6d81075/tree/821fa6b4504a4d0a5737f450d28e56cc94506ad3).
- Seed 89: finished W&B run [`2d3a2e89`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/2d3a2e89) and repaired private HF commit [`f88da35f8939`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-grpo-s89-2d3a2e89/tree/f88da35f8939dc7ff74ed0b37a004fa8c78379a8).
- Seed 107: finished W&B run [`5f0a7bf0`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/5f0a7bf0) and repaired private HF commit [`ffcbfffd322a`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-grpo-s107-5f0a7bf0/tree/ffcbfffd322a181e82c0d3a552ec611432dce471).
- Seed 131: finished W&B run [`6dd77cf1`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/6dd77cf1) and private HF commit [`7786d5e94527`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-grpo-s131-6dd77cf1/tree/7786d5e9452729bed8d029bc2e69cfe0f89e1d06).
- DAPO seed 11: finished W&B run [`e629158e`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/e629158e) and private HF commit [`33f158dfabf2`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-dapo-s11-e629158e/tree/33f158dfabf2ab0056cc213eb9c43755f5941eef).
- DAPO seed 23: finished W&B run [`c0d53921`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/c0d53921) and private HF commit [`08d3a34da450`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-dapo-s23-c0d53921/tree/08d3a34da45050b1d4787de45e5d493555fb464c).
- DAPO seed 37: finished W&B run [`605e7589`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/605e7589) and private HF commit [`bb7fb354be9f`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-dapo-s37-605e7589/tree/bb7fb354be9f02461e94f69d226b3566e8417368).
- DAPO seed 53: finished W&B run [`a4c73e0e`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/a4c73e0e) and private HF commit [`194c9d2ea0d7`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-dapo-s53-a4c73e0e/tree/194c9d2ea0d72cb93c751a314bbec9cbdcf7b5c0).
- DAPO seed 71: finished W&B run [`98b3f510`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/98b3f510) and private HF commit [`59406ac609a1`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-dapo-s71-98b3f510/tree/59406ac609a1c7539b68891f1bb639f8e8893dc9).
- DAPO seed 89: finished W&B run [`c0e809d5`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/c0e809d5) and private HF commit [`c290319779e4`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-dapo-s89-c0e809d5/tree/c290319779e45c6ca690d17fd666faa7a8e73813).
- DAPO seed 107: finished W&B run [`f56135ec`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/f56135ec) and private HF commit [`6e60484cfd40`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-dapo-s107-f56135ec/tree/6e60484cfd40dd1d6b17ee92b18af93e12f7a011).
- DAPO seed 131: finished W&B run [`95076dd5`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/95076dd5) and repaired private HF commit [`39f916902470`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-dapo-s131-95076dd5/tree/39f916902470b3b800af5c8d60d398a164cd2b95).
- GSPO seed 11: finished W&B run [`5f4fc0d3`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/5f4fc0d3) and repaired private HF commit [`f064274b7c79`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-gspo-s11-5f4fc0d3/tree/f064274b7c79372fe3fa1501737ae8a5398bec07).
- GSPO seed 23: finished W&B run [`1c16ac07`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/1c16ac07) and private HF commit [`d1a6f1879bde`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-gspo-s23-1c16ac07/tree/d1a6f1879bdee3eb41f65cdfc1ce34606641d828).
- GSPO seed 37: finished W&B run [`60e5b791`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/60e5b791) and private HF commit [`a9c9ee88756d`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-gspo-s37-60e5b791/tree/a9c9ee88756d4c10e67925964f7b4a3662e354ad).
- GSPO seed 53: finished W&B run [`ada5a2c2`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/ada5a2c2) and private HF commit [`6b58a60fb73d`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-gspo-s53-ada5a2c2/tree/6b58a60fb73d0ac1f34e77e52aac9083877fb7b7).
- GSPO seed 71: finished W&B run [`10ef44ab`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/10ef44ab) and final repaired private HF commit [`c0c0b968a61a`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-gspo-s71-10ef44ab/tree/c0c0b968a61a9ade251b5b7e6ece3119197dc1b1).
- GSPO seed 89: finished W&B run [`3ec7b7f9`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/3ec7b7f9) and private HF commit [`40695b92925a`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-gspo-s89-3ec7b7f9/tree/40695b92925a2a11cf2a63fe6bfa3b65715004c6).
- GSPO seed 107: finished W&B run [`879174ca`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/879174ca) and private HF commit [`2b521a728689`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-gspo-s107-879174ca/tree/2b521a728689e87afc7ac027108a4e3725468820).
- GSPO seed 131: finished W&B run [`726607de`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/726607de) and private HF commit [`5753775e1976`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-gspo-s131-726607de/tree/5753775e1976d3441f8643fff2dc852a53e16da3).
- Dr.GRPO seed 11: finished W&B run [`52417d14`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/52417d14) and private HF commit [`dbcd0740bc95`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-drgrpo-s11-52417d14/tree/dbcd0740bc95d0500713dfbcd19e667cdd8555a3).
- Dr.GRPO seed 23: finished W&B run [`63941052`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/63941052) and private HF commit [`4318ffd9026b`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-drgrpo-s23-63941052/tree/4318ffd9026b64749561c033ab7c0f3e3841cb84).
- Dr.GRPO seed 37: finished W&B run [`5610e4f2`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/5610e4f2) and private HF commit [`26e36e5e915f`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-drgrpo-s37-5610e4f2/tree/26e36e5e915f05922d9792014e2afecd18f6f364).
- Dr.GRPO seed 53: finished W&B run [`1215e9cb`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/1215e9cb) and private HF commit [`5bef04e01ba9`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-drgrpo-s53-1215e9cb/tree/5bef04e01ba9345493e014b5ea301f295122d378).
- Dr.GRPO seed 71: finished W&B run [`918b9a8a`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/918b9a8a) and private HF commit [`a8cd65ea26f0`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-drgrpo-s71-918b9a8a/tree/a8cd65ea26f03909ab84b2f5f3601b64ba0ae0c5).
- Dr.GRPO seed 89: finished W&B run [`faea790b`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/faea790b) and private HF commit [`1977a08e30b3`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-drgrpo-s89-faea790b/tree/1977a08e30b30ca0db3a48f3bafaebfc2beb22cc).
- Dr.GRPO seed 107: finished W&B run [`e033d448`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/e033d448) and private HF commit [`8978e087b78f`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-drgrpo-s107-e033d448/tree/8978e087b78f65f607f809f67bd7cffb89a5f381).
- Dr.GRPO seed 131: finished W&B run [`31ef6f34`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/31ef6f34) and private HF commit [`4b05547ae98a`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-drgrpo-s131-31ef6f34/tree/4b05547ae98a97a5cf069249e3a642e9b600e057).
- AERO seed 11: finished W&B run [`0d4c16f9`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/0d4c16f9) and private HF commit [`2affa816a795`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-aero-s11-0d4c16f9/tree/2affa816a79574e2c6e9b38915b2f58daf5fd9cf).
- AERO seed 23: finished W&B run [`3573a421`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/3573a421) and private HF commit [`92123d35e401`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-aero-s23-3573a421/tree/92123d35e401ff9c6c42494a383116cfbd91479e).
- AERO seed 37: finished W&B run [`7547ba19`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/7547ba19) and private HF commit [`613d3f70cc0f`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-aero-s37-7547ba19/tree/613d3f70cc0fb89eae5775a398d6f66136d14f49).
- AERO seed 53: finished W&B run [`bd8d23f9`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/bd8d23f9) and private HF commit [`448707a7a1dc`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-aero-s53-bd8d23f9/tree/448707a7a1dce3f705204d90d06598a971cfbed2).
- AERO seed 71: finished W&B run [`c73138a3`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/c73138a3) and private HF commit [`5168da9f98c3`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-aero-s71-c73138a3/tree/5168da9f98c37cd432e60b14443436daaaed6951).
- AERO seed 89: finished W&B run [`201279c6`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/201279c6) and private HF commit [`43421d47afa9`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-aero-s89-201279c6/tree/43421d47afa9a82453e883508a634f55ff770f14).
- AERO seed 107: finished W&B run [`853facf3`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/853facf3) and private HF commit [`42980267a519`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-aero-s107-853facf3/tree/42980267a519c028dd0e5a0592c0a106933fb0b7).
- AERO seed 131: finished W&B run [`bdb898f5`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/bdb898f5) and private HF commit [`ec9931f7fe85`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-aero-s131-bdb898f5/tree/ec9931f7fe85e6b180dbb85d6c20c3e1646f5272).
- Full Trainer checkpoints, including optimizer, scheduler, trainer state, and
  RNG state, are stored at steps 5, 10, 15, 20, 25, and 30 in all forty repositories.
  Each also contains the final LoRA adapter, tokenizer, 500-row held-out trace,
  and `run_manifest.json`; GRPO seeds 11 and 131, all eight DAPO seeds,
  GSPO seeds 11, 23, 37, 53, 71, 89, 107, and 131, and all eight Dr.GRPO
  seeds additionally
  retain resumable evaluation progress files.
- Local accepted records and manifests are
  `results/full/grpo-seed-{11,23,37,53,71,89,107,131}.json` and
  `results/full/manifests/grpo-seed-{11,23,37,53,71,89,107,131}.json`, plus
  `results/full/dapo-seed-{11,23,37,53,71,89,107,131}.json` and
  `results/full/manifests/dapo-seed-{11,23,37,53,71,89,107,131}.json`, plus
  `results/full/gspo-seed-{11,23,37,53,71,89,107,131}.json` and
  `results/full/manifests/gspo-seed-{11,23,37,53,71,89,107,131}.json`, plus
  `results/full/drgrpo-seed-{11,23,37,53,71,89,107,131}.json` and
  `results/full/manifests/drgrpo-seed-{11,23,37,53,71,89,107,131}.json`, plus
  `results/full/aero-seed-{11,23,37,53,71,89,107,131}.json` and
  `results/full/manifests/aero-seed-{11,23,37,53,71,89,107,131}.json`.

At the final-integrity-audit checkpoint, the hardened frozen aggregator accepted
39/40 records. It rejected the remaining legacy record for missing
`completion_sha256` values, reported `INCOMPLETE`, and emitted no verdict. That
state is retained here as chronology; the exact-checkpoint replay documented
below subsequently restored a fully hashed 500-row trace for every unit.

At 2026-07-15 15:27 UTC, Colab reclaimed the three active A100 sessions. The
associated W&B runs correctly changed to `crashed`; no completed unit was
affected. Private HF recovery evidence remains intact: seed 89 has checkpoint
30 plus 288/500 evaluation rows, seed 107 has checkpoint 30 plus 256/500 rows,
and seed 131 has checkpoint 25. New A100 assignments then returned the explicit
backend error that the account lacked current A100 quota or entitlement. The
campaign supervisor now ignores zombie wrappers, recognizes those remote
checkpoints even after a later step-zero assignment failure, applies a global
15-minute assignment backoff, and selects evaluation-only recovery for seeds
89/107 and exact-source checkpoint resume for seed 131. Its focused campaign,
resume, verifier, aggregator, and obligations suite passes 39 tests.

At 2026-07-15 15:45 UTC, the first guarded post-hardening retry selected the
intended recovery path for every interrupted unit, then all three A100
assignments failed with backend HTTP 400 before a VM was allocated. The active
OAuth2 token is current and belongs to `arvindcr4@gmail.com`, Colab CLI 0.6.0 is
the latest available release, and the account can still query assignments; the
failure is therefore specific to A100 allocation rather than authentication or
the campaign runner. The private HF repositories and crashed W&B records were
unchanged. Substituting T4/L4 is not admissible because the frozen amendment
locks `runtime.accelerator` to `Google Colab A100`, and exact-source resume also
enforces the source accelerator. The supervisor remains live under its guarded
retry cooldown pending restoration of A100 quota/entitlement.

At 2026-07-15 16:00 UTC, the second guarded retry again failed before VM
allocation. It exposed a supervisor provenance edge case: after a failed
evaluation launch, the wrapper result pointed to its evaluation request, so the
next selector could temporarily classify seeds 89/107 as new units even though
their immutable training requests and checkpoint-30 repositories still existed.
The selector now follows nested `source_request` links back to the canonical
confirmatory request, and pre-allocation failures no longer consume the
scientific retry budget. Final-attempt children are also counted as active before
retry-limit evaluation. The 39-test suite and lint pass, the quota-only attempts
were refunded to zero, and the repaired persistent supervisor is running in
terminal session `57844`; no remote evidence changed.

The repaired behavior was exercised live at 2026-07-15 16:15 UTC. Seeds 89
and 107 were again selected as evaluation recovery from checkpoint 30, seed 131
as exact-source checkpoint resume, and Colab again rejected all three A100
allocations. The supervisor emitted a retry-credit receipt for every unit,
returned all three attempt counters to zero, and entered the next global
cooldown without creating a VM or changing HF/W&B evidence.

At 2026-07-16 08:36 UTC, A100 allocation resumed. GRPO seed 107 entered
evaluation-only recovery on an A100 and began reconstructing checkpoint 30; six
checkpoint files were present locally by 09:02 UTC while Hugging Face continued
returning intermittent HTTP 504 responses for the remaining metadata and files.
DAPO seed 11 began a new frozen confirmatory run on a second A100. Its W&B run
`e629158e` remained `running`, recorded global step 3, and the trainer entered
step 4/30 by 09:02 UTC. Its first durable private-HF checkpoint is due at step 5.

The seed-89 evaluation recovery and seed-131 exact-source resume each obtained
an A100 but exited on Hugging Face HTTP 504 failures. Those failures did not
invalidate their preserved source checkpoints. The supervisor now retries
transient remote execution failures inside the same Colab session, identifies
provider failures from the final runner evidence, refunds them exactly once in
a persisted credit ledger, and applies a shared 15-minute dependency backoff.
The live reconciliation returned seed 89 and seed 131 to zero consumed attempts.
The focused audit discovery suite passes 31 tests; compilation, Ruff, and diff
checks pass. The supervisor also regenerates the fail-closed frozen aggregate
whenever the independently validated count changes. The frozen experiment
sources and stack fingerprint were not changed by this launcher-only hardening.

At that point, the persistent supervisor observed three server-side Colab
assignments: the two named campaign A100 sessions and one unnamed CPU session.
It therefore correctly held campaign capacity at zero under the proven
three-session ceiling instead of oversubscribing the account. Validated evidence
remains fail-closed until a running unit finishes all 30 steps, the 500-row held-out
evaluation, W&B finalization, private-HF checkpoint verification, and local
strict reconciliation.

At 2026-07-16 09:11 UTC, GRPO seed 107 had made no additional checkpoint-file
or log progress for more than 30 minutes while Hugging Face continued returning
504 responses. Its remote checkpoint 30 and 256-row evaluation progress were
already durable, so the idle recovery A100 was safely released. The hung local
Colab transport was then terminated, allowing the wrapper to write a step-7
failure record containing the 504 evidence. At 09:12 UTC the supervisor applied
the persisted provider-failure credit, returned seed 107 to zero attempts, and
extended dependency backoff to approximately 09:27 UTC. Seeds 89, 107, and 131
therefore all remain eligible for their exact recovery paths with zero consumed
scientific attempts. DAPO seed 11 continues on the remaining campaign A100.
Its W&B run recorded global step 4 and the trainer entered step 5/30 at 09:13
UTC. At 09:18 UTC, step 5 completed and the trainer entered step 6. The complete
checkpoint was independently verified in the private repository at commit
`a6d3d8d619df4c7cd99efd4e65ed34ef11253bdd`: adapter, optimizer, scheduler,
trainer state, and RNG state are all present. DAPO seed 11 is therefore
recoverable from its first frozen checkpoint, although it is not a validated
unit until all 30 steps and the 500-row held-out evaluation finish.
The exact-source resume runner also accepted this unit's immutable request in
dry-run mode, reproducing fingerprint
`e629158e48d9db298e38e72b815d39707ed10b7768f506d500a72fd137ac0456`, frozen
stack `4737be74fdc97dd400b846e93c6b0c03f443ee65da2cc89697bf5b078d16aa0d`,
the A100 requirement, all six package pins, and the content-addressed source
snapshots without allocating another session.

At 2026-07-16 10:15 UTC, GRPO seed 89 completed evaluation-only recovery from
its immutable step-30 checkpoint. Independent reconciliation confirmed a
finished W&B run, private HF commit
`2a125e30189a2481dd1750cb5bbd2071ab3b5cd5`, complete checkpoints at steps
5/10/15/20/25/30, the final adapter, matching frozen stack and unit
fingerprints, and a contiguous 500-row held-out trace with 310 correct answers
(`0.6200`). The frozen aggregate refreshed to 6/40 with 34 missing units, zero
errors, and no verdict. The released A100 was immediately backfilled with GRPO
seed 107 evaluation recovery from its preserved 256/500 progress state. DAPO
seed 11 continues on the other A100; its private step-10 checkpoint was also
independently verified before training advanced beyond it.

At 2026-07-16 11:09 UTC, GRPO seed 107 completed evaluation-only recovery
from its immutable step-30 checkpoint and preserved 256-row evaluation state.
Independent reconciliation confirmed the exact frozen unit and stack
fingerprints, a finished W&B run, private HF commit
`99ca971725c694340ad475e9c03d7248a21a46eb`, all six complete checkpoint trees,
the final adapter, and an identical local/remote manifest containing a
contiguous 500-row trace with 317 correct answers (`0.6340`). The aggregate
advanced to 7/40 with 33 missing units, zero errors, and no verdict. The
released A100 was immediately backfilled with GRPO seed 131 through its exact
source checkpoint-resume path. DAPO seed 11 reached step 21/30; its private
step-15 and step-20 checkpoints were independently verified with all mandatory
resume artifacts.

At 2026-07-16 11:24 UTC, the newly available third A100 slot exposed a Colab
CLI transport edge case: DAPO seeds 23 and 37 lost their kernel WebSocket during
setup, before training or W&B initialization. The idle sessions were released,
both failures were classified as provider-side and credited back to zero unit
attempts, and no scientific evidence was created or changed. The launcher now
recognizes `Connection was lost.` as transient and enforces a local watchdog
30 seconds beyond every Colab `--timeout`, preventing a dead CLI transport from
occupying a campaign slot indefinitely. The 33-test focused audit suite,
byte-compilation, Ruff, and diff checks pass. After supervisor reload and live
reconciliation, DAPO seed 53 launched cleanly on the third A100. GRPO seed 131
has independently verified checkpoint 30 and is evaluating; DAPO seed 11 and
DAPO seed 53 are training on the other two A100s. DAPO seed 11 subsequently
advanced to step 26/30, with its complete step-25 resume checkpoint independently
verified at private HF commit
`2fa4345adf662b3285db8b6ae4455c13c16eb283`. DAPO seed 53 subsequently
advanced to step 6/30; its first complete resume checkpoint was independently
verified at private HF commit
`91e59c4dfec866a9efd32691acf7cccab165e0e9`. GRPO seed 131 reached 216/500
held-out rows with continuously persisted HF evaluation progress.

At 2026-07-16 13:03 UTC, GRPO seed 131 completed exact-source recovery from
its immutable checkpoint-25 request and finished the 500-row held-out
evaluation. Independent reconciliation confirmed 311 correct answers
(`0.6220`), a contiguous trace, matching frozen stack and unit fingerprints,
a finished W&B run, and exact private HF commit
`7786d5e9452729bed8d029bc2e69cfe0f89e1d06`. That revision contains all six
complete checkpoint trees, the final adapter, and a `run_manifest.json`
byte-identical to the local manifest. The aggregate advanced to 8/40 with 32
missing units, zero errors, and no verdict. The completed Colab session was
confirmed idle and released after a benign post-result CLI stream-cleanup
error; the scientific result and wrapper return code were already complete and
zero. The supervisor immediately backfilled the slot with DAPO seed 23. DAPO
seed 11 is evaluating from its independently verified checkpoint 30, and DAPO
seed 53 continues training from independently verified checkpoints 5 and 10.

At 2026-07-16 14:20 UTC, DAPO seed 11 completed all 30 steps and the 500-row
held-out evaluation with 324 correct answers (`0.6480`). Independent
reconciliation confirmed finished W&B run `e629158e`, exact private HF commit
`33f158dfabf2ab0056cc213eb9c43755f5941eef`, the final adapter, all six
11-file checkpoint trees with optimizer/scheduler/trainer/RNG state, matching
frozen stack and DAPO treatment fingerprints, and a contiguous 500-row remote
manifest byte-identical to the local manifest. The wrapper completed with
return code zero and no failed step. The aggregate advanced to 9/40 with 31
missing units, zero errors, and no verdict. The idle A100 was explicitly
released. The supervisor backfilled the slot with DAPO seed 37, but that setup
lost its kernel WebSocket before training or W&B initialization and the local
watchdog terminated the stale CLI transport. No VM remains allocated and no
scientific state was created; the supervisor credited the attempt back to zero
and entered dependency backoff until approximately 14:40 UTC. DAPO seed 53
continues beyond its independently verified checkpoint 25 and DAPO seed 23
continues beyond its independently verified checkpoint 10 on the two healthy
A100 sessions.

At 2026-07-16 14:41 UTC, the cooldown expired and DAPO seed 37 obtained another
A100, but the fresh VM again lost its kernel WebSocket during setup before W&B
or training initialization. This was its third transport-only setup loss; the
session was released, no scientific state was created, and the attempt was
credited back to zero under a new dependency cooldown ending near 14:56 UTC.
Because repeated credited failures on the first missing unit could otherwise
pin the third slot indefinitely, the supervisor now prioritizes never-attempted
eligible units and then the least-recently attempted retry, preserving frozen
preregistration order as the tie-breaker. The 20 focused supervisor tests,
Ruff, byte-compilation, and diff checks pass. The supervisor was gracefully
reloaded without interrupting the two child runners; its next third-slot launch
will rotate to an untouched frozen unit. Meanwhile DAPO seed 53 completed step
30, its full resume checkpoint was independently verified at private HF commit
`93c133229fad2462da10fd6128e1f10f05be57ad`, and it entered held-out
evaluation. DAPO seed 23 continues beyond verified checkpoint 10.

At 2026-07-16 14:57 UTC, the cooldown expired and the reloaded selector worked
as intended: it rotated the third slot to untouched DAPO seed 71 rather than a
fourth consecutive DAPO seed-37 setup attempt. The new A100 session completed
package installation and the pinned environment check, then entered the secure
runner. It subsequently initialized running W&B run `98b3f510` with a fresh
heartbeat and created private HF repository
`arvindcr4/tinker-rl-lab-e1-dapo-s71-98b3f510`. It then reached independently
verified private checkpoint 10 at exact HF commit
`ff2d368842192d5da2c35d9657f4e7d71982e798`; the 11-file tree contains all six
resume-critical artifacts and reports trainer global step 10. DAPO seed 53 continued
held-out evaluation; its private progress commits through 480/500 were
independently verified for exact revision, 480 contiguous indices, 301 correct
(62.7%), checkpoint 30, held-out size 500, and the frozen unit fingerprint.
DAPO seed 23 reached independently verified private checkpoint 20 at exact HF
commit `c56ec2adeafc6fe62b2a197fa77ebcf92a348603`; its 11-file tree contains all
six resume-critical artifacts and reports trainer global step 20.

At 2026-07-16 16:28 UTC, DAPO seed 53 completed its 500-row held-out
evaluation with 318 correct answers (`0.6360`). Independent reconciliation
confirmed finished W&B run `a4c73e0e`, exact private HF commit
`194c9d2ea0d72cb93c751a314bbec9cbdcf7b5c0`, the final adapter, all six
11-file checkpoint trees with optimizer/scheduler/trainer/RNG state, matching
frozen stack and DAPO treatment fingerprints, and a contiguous 500-row remote
manifest byte-identical to the local manifest. The wrapper completed with
return code zero and no failed step. The aggregate advanced to 10/40 with 30
missing units, zero errors, and no verdict. A Colab CLI standard-stream error
occurred only after the completed result was written; the idle VM was then
explicitly released without affecting the accepted evidence. The supervisor
continues DAPO seeds 23 and 71 and will backfill the freed third slot.

At 2026-07-16 16:34 UTC, Colab reclaimed the DAPO seed-23 and seed-71 VMs
after their latest remote heartbeats. Their exact private-HF recovery points
remain intact at checkpoint 20 (`c56ec2adeafc6fe62b2a197fa77ebcf92a348603`)
and checkpoint 10 (`ff2d368842192d5da2c35d9657f4e7d71982e798`), respectively. The
local Colab transports did not notice the remote reclamation and would have
remained blocked behind their six-hour execution timeout, so they were
terminated without deleting any scientific artifact. After Colab capacity was
restored, both exact-source recovery launchers allocated fresh A100s, passed
the frozen environment check, and confirmed resume from those precise remote
checkpoints. A third fresh A100 was allocated to DAPO seed 89, which also
passed the environment check. The supervisor now compares each active
transport's exact session name with three consecutive successful `colab
sessions` probes before reaping an absent transport; 24 focused tests,
byte-compilation, Ruff, and diff checks pass. The campaign again occupies the
proven three-session limit. The task heartbeat now explicitly notifies this
thread on future Colab credit, quota, or accelerator-entitlement rejection.
The recovered DAPO seed-23 run subsequently completed all 30 steps and the
500-row held-out evaluation with 312 correct answers (`0.6240`). Independent
reconciliation confirmed finished W&B run `c0d53921`, exact private HF commit
`08d3a34da45050b1d4787de45e5d493555fb464c`, all six complete 11-file
checkpoint trees, the final adapter, frozen stack and treatment fingerprints,
and a contiguous remote manifest byte-identical to the local manifest. Its
post-result Colab stream remained open after W&B and HF had finalized, so a new
strict remote-only reconciliation mode rebuilt the accepted local record from
those immutable artifacts without allocating another GPU. The stale idle A100
was released, the aggregate advanced to 11/40 with 29 missing units and zero
errors, and the supervisor backfilled the slot with DAPO seed 107. DAPO
seed 71 reached independently verified private checkpoint 30 at exact HF
commit `e6b10e441d945f82b2c36c20977242e6f0e520a2`; all six 11-file checkpoint
trees contain the resume-critical artifacts and report their expected trainer
global steps. Its held-out evaluation reached independently verified private
progress commit `e3463574664ba7af2b045f9e6a6483da3cfdba24` at 464/500 rows,
with 464 contiguous indices, 288 correct, checkpoint step 30, held-out size 500,
and the frozen unit fingerprint. Fresh DAPO seed 89 reached independently
verified private checkpoint 30 and progressed through 480/500 evaluation rows at exact HF commit
`d1c3a9d74c669f8a4d0fd072cd7fcd1a356bd155`; all six checkpoint trees contain
11 files and every resume-critical artifact, the evaluation indices are
contiguous with 300 correct, and the frozen unit and stack fingerprints remain
intact. Seed 71 then completed the full 500-row evaluation with 317 correct
answers (`0.6340`). Independent reconciliation accepted exact private HF commit
`59406ac609a1c7539b68891f1bb639f8e8893dc9`, finished W&B run `98b3f510`, all
six 11-file checkpoint trees with their exact trainer steps, the final adapter,
the byte-identical 500-row trace, and both frozen fingerprints. Its A100 was
released, the aggregate advanced to 12/40 with 28 missing units and zero
errors, and the supervisor backfilled DAPO seed 131 while seeds 89 and 107
continued. DAPO seed 107 then reached exact private HF commit
`cefd62d418d685b468441c99b7301df3c3963925`; its 11-file checkpoint-5,
checkpoint-10, and checkpoint-15 trees contain the adapter, optimizer,
scheduler, RNG state, trainer states at their exact global steps, and every
other resume-critical artifact, while W&B run `f56135ec` remains live beyond
step 15. DAPO seed 131 likewise reached exact private HF commit
`06d3e0951e1e37423ab0035a083e96de85159672`; its 11-file checkpoint-5 and
checkpoint-10 trees contain the same resume-critical state, its latest trainer
state reports exactly global step 10, and W&B run `95076dd5` remains live. The
focused suite now passes 42 tests; byte-compilation, Ruff, and diff checks pass.
Seed 89 then completed all 500
held-out rows with 318 correct (`0.6360`). Independent reconciliation verified
finished W&B run `c0e809d5`, exact private HF commit
`c290319779e45c6ca690d17fd666faa7a8e73813`, all six complete 11-file
checkpoint trees and exact trainer steps, the final adapter, identical
contiguous progress and manifest traces, and both frozen fingerprints. The
idle A100 and stale transport were released, the aggregate advanced to 13/40
with 27 missing units and zero errors, and the supervisor backfilled GSPO seed
11 to restore three active A100s. GSPO seed 11 subsequently reached exact
private-HF commit `d44a159ea7510c97887b5905a62e1da8eaa18ddd`; its
checkpoint-5 and checkpoint-10 trees each contain all 11 required resume files,
and its latest trainer state reports exactly global step 10.

At 2026-07-17 04:17 UTC, host inspection established that the Mac had rebooted
at 04:14:42 UTC, terminating the local tmux supervisor and runner transports.
This was a host reboot rather than a reported Colab credit or quota failure.
W&B marked DAPO seeds 107 and 131 and GSPO seed 11 crashed at global steps 19,
14, and 19, respectively. Their last complete private-HF recovery points remain
DAPO-107 checkpoint 15 at `cefd62d418d685b468441c99b7301df3c3963925`,
DAPO-131 checkpoint 10 at `06d3e0951e1e37423ab0035a083e96de85159672`,
and GSPO-11 checkpoint 15 at
`7ba66f44a5060e8f125caab76656c5c2cbcaf33d`; every tree contains all 11
resume files and the latest trainer states report their exact steps. Three
exact-source resume sessions were launched, all three received A100s and passed
the frozen environment check, and the persistent supervisor was restored with
three active and three remote assignments. The process scanner was also
hardened to ignore a tmux server process title that embeds its first window's
runner command, preventing a phantom fourth slot after recovery; the focused
suite remains 42/42 with Ruff and diff checks clean. GSPO seed 11 then completed
training at exact private-HF commit
`29364775fc310e75ad821a2ed8745b90f0aa4793`; all six checkpoint trees contain
the 11 required resume files, the checkpoint-30 trainer state reports exactly
global step 30, W&B remains live at step 30, and the 500-row held-out evaluation
reached 104/500 rows; exact private-HF progress commit
`279201f3eabe201f3ecca7189117f7eed63279f6` independently contains 96
contiguous traces covering indices 0--95 with 59 correct. DAPO seed 131
reached exact private-HF commit `4f5e8ca6c78cfcb758b58b16915359ebd8d31c8e`;
its checkpoint-15 tree contains all 11 resume files and its trainer state reports
exactly global step 15. DAPO seed 107 reached exact private-HF commit
`ef001e4caed28e3dd01cf1462cbb62e66f46eb20`; its checkpoint-20 tree contains
all 11 resume files, its trainer state reports exactly global step 20, and W&B
confirms step 20.

At 2026-07-17 05:18 UTC, all three remote assignments disappeared from the
authoritative Colab session list while their local CLI transports were still
alive. After the exact session names remained absent for three consecutive
successful probes, the supervisor terminated only the stale transports and
the wrappers recorded infrastructure failures. Replacement A100 probes at
05:20 UTC were rejected before VM creation with `Backend rejected accelerator
'A100'. You may not have quota or entitlement for this accelerator on your
account.` The three pre-assignment failures were credited back and did not
consume scientific attempts. Durable recovery remains independently present
in private HF: DAPO-107 through checkpoint 20, DAPO-131 through checkpoint 15,
and GSPO-11 through checkpoint 30 plus exact evaluation progress through row
112 at commit `19b2887e44b5fa4e38c75db539165274f7dc3863`. The supervisor
now ranks Hub-proven evaluation recovery ahead of partial-checkpoint recovery,
then untouched units, while pre-assignment failures retain the anti-pinning
rotation policy. Live selection against the current Hub state chooses GSPO-11,
DAPO-107, and DAPO-131 in that order for the next three available A100s. The
05:35 UTC guarded retry exercised that ordering end to end: GSPO-11 launched as
evaluation recovery, DAPO-107 and DAPO-131 launched as exact-source checkpoint
resumes, and Colab rejected all three A100s before VM creation with the same
quota-or-entitlement message. The supervisor refunded all three attempts and
entered another global cooldown. The focused suite passes 44/44;
byte-compilation, Ruff, and diff checks pass. The 15-minute task heartbeat
reports this quota/entitlement condition here and will report when capacity
returns.

At 2026-07-17 07:28 UTC, the user reloaded Colab credits while the supervisor
was still inside the prior rejection cooldown. A single exact GSPO-11 recovery
probe bypassed only that stale wait and received an A100; the two proven
checkpoint resumes then filled the remaining slots. All three sessions passed
the frozen environment check on NVIDIA A100-SXM4-40GB with the exact six package
pins. DAPO-107 restored checkpoint 20, DAPO-131 restored checkpoint 15, and
GSPO-11 restored checkpoint 30 plus the exact 112-row evaluation prefix. The
persistent supervisor independently recognizes all three local runners and all
three named remote sessions, reports three occupied slots, and therefore cannot
overfill the account while the recovered work proceeds.

At 2026-07-17 08:51 UTC, GSPO seed 11 completed evaluation-only recovery from
its immutable checkpoint 30 and exact 112-row prefix. Independent verification
confirmed 320/500 correct (`0.6400`) with contiguous indices, matching frozen
stack and unit fingerprints, finished W&B run `5f4fc0d3`, and exact private HF
commit `37a2793c2138940f1ece2950dd56df3e1cdf7ccc`. That revision is private,
contains the final adapter and all six complete 11-file checkpoint trees, and
its trainer states report exact global steps 5/10/15/20/25/30. The remote
manifest is byte-identical to the accepted local 500-row manifest. The wrapper
completed with return code zero, released its idle A100, and the fail-closed
aggregate advanced to 14/40 with 26 missing units, zero errors, and no verdict.
The supervisor immediately backfilled the slot with GSPO seed 71, restoring the
proven three-session ceiling while DAPO seeds 107 and 131 continue.

At 2026-07-17 10:22 UTC, DAPO seed 107 completed its exact-source recovery and
all 500 held-out examples with 317 correct (`0.6340`). Independent verification
confirmed finished W&B run `f56135ec`, exact private HF commit
`6e60484cfd40dd1d6b17ee92b18af93e12f7a011`, six complete 11-file checkpoint
trees with trainer states at 5/10/15/20/25/30, the final adapter, matching frozen
unit/stack/treatment fingerprints, and a byte-identical local/remote manifest.
The wrapper returned zero and released its A100; the fail-closed aggregate
advanced to 15/40 with 25 missing units, zero errors, and no verdict. The
supervisor backfilled the slot with GSPO seed 89. DAPO seed 131 and GSPO seed 71
were safely moved to evaluation recovery after stalled Colab transports; their
new sessions resumed from independently verified HF prefixes of 304 and 80 rows.

At 2026-07-17 11:05 UTC, DAPO seed 131 completed evaluation recovery with
316/500 correct (`0.6320`). Independent acceptance verified finished W&B run
`95076dd5`, the private HF repository, six exact 11-file checkpoint trees with
trainer states at steps 5/10/15/20/25/30, the final adapter, frozen unit and
stack fingerprints, and a contiguous 500-row trace. A post-acceptance review
found that the evaluation-only finalizer had hardcoded 480 rollouts instead of
reading DAPO's checkpoint telemetry. The finalizer now derives dynamic-arm
rollouts fail-closed and preserves the treatment configuration; the corrected
manifest records 1,728 rollouts, includes an explicit correction receipt, is
byte-identical locally and at private HF commit
`f34a67a1348a1f556e0f1ba78c2812cadf06e5ed`, and matches the corrected finished
W&B summary. The aggregate advanced to 16/40 with 24 missing units and zero
errors. The released A100 was immediately backfilled with GSPO seed 107.

At 2026-07-17 11:44 UTC, GSPO seed 71 completed evaluation recovery from its
immutable checkpoint 30 and exact 80-row prefix with 319/500 correct
(`0.6380`). Independent checks confirmed finished W&B run `10ef44ab`, a private
HF repository, six exact 11-file checkpoint trees with trainer states at steps
5/10/15/20/25/30, the final adapter, matching frozen stack, unit, and GSPO
treatment fingerprints, and a contiguous 500-row trace. Post-acceptance review
found that this older recovery finalizer omitted the frozen treatment fields
from `run_config`; it did not alter the already-correct rollout count, training,
or held-out evidence. The manifest now includes those fields and an explicit
correction receipt, is byte-identical locally and at private HF commit
`d3b74ccd365ba40f467c24e68b64bc597eb6746c`, and matches the finished W&B
summary. The fail-closed aggregate advanced to 17/40 with 23 missing units and
zero errors. The released A100 was immediately backfilled with GSPO seed 131,
keeping GSPO seeds 89, 107, and 131 active at the three-session ceiling.

At 2026-07-17 13:18 UTC, GSPO seed 89 completed its full frozen run with
323/500 correct (`0.6460`). Independent checks confirmed finished W&B run
`3ec7b7f9`, exact private HF commit
`40695b92925a2a11cf2a63fe6bfa3b65715004c6`, six 11-file checkpoint trees
whose trainer states report exact steps 5/10/15/20/25/30, the final adapter,
matching frozen stack, unit, and GSPO treatment fingerprints, and a contiguous
500-row trace with 500 unique completion hashes. The accepted local and remote
manifests are byte-identical. The fail-closed aggregate advanced to 18/40 with
22 missing units and zero errors; the released A100 was immediately backfilled
with Dr.GRPO seed 11, retaining the proven three-session ceiling alongside
GSPO seeds 107 and 131.

At 2026-07-17 14:02 UTC, GSPO seed 107 completed its full frozen run with
320/500 correct (`0.6400`). Independent reconciliation confirmed finished W&B
run `879174ca`, exact private HF commit
`2b521a728689e87afc7ac027108a4e3725468820`, six complete 11-file checkpoint
trees whose trainer states report exact steps 5/10/15/20/25/30, the final
adapter, matching frozen stack, unit, and GSPO treatment fingerprints, and a
contiguous 500-row trace with 500 unique completion hashes. The accepted local
and remote manifests are byte-identical. The aggregate advanced to 19/40 with
21 missing units and zero errors. A subsequent attempt to fill the third slot
with Dr.GRPO seed 23 was rejected before VM allocation because Colab reported
that the account might lack A100 quota or entitlement. That infrastructure
failure was credited back (`attempts[drgrpo:23]=0`); GSPO seed 131 retained all
six checkpoints and evaluation progress, Dr.GRPO seed 11 continued training,
and the supervisor entered guarded allocation backoff through 14:18 UTC.

At 2026-07-17 14:14 UTC, Colab reclaimed the two remaining A100 sessions. The
supervisor confirmed their absence across three independent session polls
before reaping the local wrappers. GSPO seed 131 preserved all six training
checkpoints plus a verified 368-row evaluation prefix at exact private HF
commit `b3d67a9710a3e031cadc2a97744851fae3edf21c` (221 correct, contiguous
indices, and 368 unique completion hashes). Dr.GRPO seed 11 preserved complete
checkpoints through step 25 at exact private HF commit
`ce0166fdc14a3d3d555ebe4a10c2afce40fb5f53`; its trainer state reports exactly
global step 25 of 30 with 25 log-history entries. At 14:18 UTC, guarded retry
selected the correct recovery modes---evaluation-only for GSPO-131,
exact-source checkpoint resume for Dr.GRPO-11, and a fresh Dr.GRPO-37 unit---but
Colab rejected all three A100 allocations before VM creation with the same
quota-or-entitlement error. The infrastructure failures were credited back,
leaving `attempts[gspo:131]=1`, `attempts[drgrpo:11]=1`, and
`attempts[drgrpo:37]=0`; no scientific attempt or preserved evidence was lost.
The supervisor remains live under guarded allocation backoff through 14:33:47
UTC.

At 2026-07-17 14:34 UTC, reloaded Colab credits restored all three A100
allocations on the first normal post-cooldown probe. The supervisor launched
GSPO seed 131 in evaluation-only recovery, Dr.GRPO seed 11 in exact-source
checkpoint resume, and a fresh Dr.GRPO seed 53 unit. Direct session inspection
confirmed three NVIDIA A100 assignments, and all three passed the frozen
environment check with CUDA 12.8, Torch 2.11.0+cu128, and the exact six package
pins. GSPO-131 restored checkpoint 30 plus its exact 368-row evaluation prefix;
Dr.GRPO-11 restored checkpoint 25; Dr.GRPO-53 entered the fresh Qwen3-8B
training path. The campaign supervisor reports three local wrappers, three
named remote sessions, and three occupied slots, so it cannot overfill the
account while these units proceed.

At 2026-07-17 14:42 UTC, independent inspection of GSPO-131's first recovered
progress commit found that the standalone checkpoint evaluator had omitted
`completion_sha256` from newly appended rows, even though the original 368-row
prefix retained complete hashes. The recovery session was stopped before
acceptance, the evaluator was corrected to hash every decoded completion, and
its validator now rejects final traces with missing or malformed hashes. A
rewind helper truncates only the earliest unhashed suffix and records a receipt
before deterministic replay. Rows 368--399 from affected commit
`793083965c5b89d6d59cd85829466c194eb4f873` were removed; private HF commit
`8e3b43df25f842a02ecfb27d33c938a59a2025fe` now contains exactly the verified
368-row boundary with contiguous indices, 368 valid and unique hashes, and 221
recomputed correct answers. The focused campaign/recovery suite passes 42/42;
the complete E1 audit suite passes 50/50, with byte-compilation, Ruff, and diff
checks clean. GSPO-131 remains queued for
exact replay from row 368 after its per-unit retry cooldown; the supervisor
backfilled the temporarily free slot with fresh Dr.GRPO seed 71, retaining three
active A100 sessions without admitting the unhashed suffix.

At 2026-07-17 16:36 UTC, Dr.GRPO seed 11 completed exact-source checkpoint
resume and the full held-out evaluation with 321/500 correct (`0.6420`).
Independent reconciliation verified finished W&B run `52417d14`, exact private
HF commit `dbcd0740bc95d0500713dfbcd19e667cdd8555a3`, the final adapter, and six
complete 11-file checkpoint trees whose trainer states report exact steps
5/10/15/20/25/30. The frozen stack, unit, and treatment fingerprints match;
the 500 indices are contiguous; all completion hashes are valid and unique;
and the remote progress trace is identical to the final manifest. The aggregate
advanced to 20/40 with 20 missing units and zero errors. Its released A100 was
immediately backfilled with GSPO seed 131 evaluation recovery from the repaired
368-row boundary, restoring the proven three-session ceiling alongside Dr.GRPO
seeds 53 and 71.

At 2026-07-17 17:07 UTC, GSPO seed 131 completed deterministic evaluation
recovery from the repaired 368-row boundary with 314/500 correct (`0.6280`).
Independent reconciliation verified finished W&B run `726607de`, exact private
HF commit `5753775e1976d3441f8643fff2dc852a53e16da3`, the final adapter, and six
complete 11-file checkpoint trees at trainer steps 5/10/15/20/25/30. The
frozen stack, unit, and GSPO treatment fingerprints match; the final 500-row
trace has contiguous indices and valid unique hashes; the original 368-row
prefix is preserved byte for byte; and the local and remote manifests have the
same SHA-256 digest. The aggregate advanced to 21/40 with 19 missing units and
zero errors. Its released A100 was immediately backfilled with fresh Dr.GRPO
seed 89, restoring three active sessions alongside Dr.GRPO seeds 53 and 71.

At 2026-07-17 17:26 UTC, Dr.GRPO seed 53 completed its fresh frozen run with
318/500 held-out answers correct (`0.6360`). Independent reconciliation
verified finished W&B run `1215e9cb`, exact private HF commit
`5bef04e01ba9345493e014b5ea301f295122d378`, the final adapter, and six
complete 11-file checkpoint trees whose trainer states report exact steps
5/10/15/20/25/30. The frozen stack fingerprint, treatment-spec SHA, and unit
fingerprint match; the 500 evaluation indices are contiguous; every completion
hash is valid and unique; and local and remote manifests share SHA-256
`affd54d90c9554e8c48b23aed0a4f9b56cc8cbb695c5d0fbff88d2d5be3438ba`.
The aggregate advanced to 22/40 with 18 missing units and zero errors. Its
released A100 was immediately backfilled with fresh Dr.GRPO seed 107, retaining
three live sessions alongside Dr.GRPO seed 89 and seed 71 evaluation recovery.

At 2026-07-17 17:44 UTC, Dr.GRPO seed 71 completed evaluation recovery from its
exact checkpoint-30 source and original 240-row prefix with 310/500 held-out
answers correct (`0.6200`). Independent reconciliation verified finished W&B
run `918b9a8a`, exact private HF commit
`a8cd65ea26f03909ab84b2f5f3601b64ba0ae0c5`, the final adapter, and six
complete 11-file checkpoints at exact trainer steps 5/10/15/20/25/30. The
frozen stack, treatment-spec, source-unit, and recovery fingerprints match; the
500 indices are contiguous; all completion hashes are valid and unique; the
original 240-row trace prefix is byte-identical; and local and remote manifests
share SHA-256
`f14913a185c50ec86ce553437a4b5425a494072c8de0a6c1202c090b346ac920`.
The aggregate advanced to 23/40 with 17 missing units and zero errors. Its
released A100 was immediately backfilled with fresh Dr.GRPO seed 131, retaining
three live sessions alongside Dr.GRPO seeds 89 and 107.

Between 2026-07-17 19:58 and 20:35 UTC, fresh Dr.GRPO seeds 89, 107, and 131
completed with 314/500 (`0.6280`), 314/500 (`0.6280`), and 323/500 (`0.6460`)
held-out answers correct. Independent reconciliation verified finished W&B runs
`faea790b`, `e033d448`, and `31ef6f34`; exact private HF commits
`1977a08e30b30ca0db3a48f3bafaebfc2beb22cc`,
`8978e087b78f65f607f809f67bd7cffb89a5f381`, and
`4b05547ae98a97a5cf069249e3a642e9b600e057`; final adapters; and six complete
11-file checkpoint trees per run at exact trainer steps 5/10/15/20/25/30. All
three frozen stack, treatment-spec, and unit fingerprints match. Every final
trace contains 500 contiguous rows with valid unique completion hashes, and
each local manifest is byte-identical to its exact remote revision. The
aggregate advanced to 26/40 with 14 missing units and zero errors.

The three released A100 slots were backfilled with fresh AERO seeds 11, 23,
and 37, each of which reached and privately uploaded exact checkpoint 5 before
its first wrapper stopped. The durable HF commits are respectively
`54c5f3edf7038e56343e47d9666d313572be9fc5`,
`c24de8d96afaca8b2c1e0aca0a92b181fcae78ec`, and
`003039f4d2456e7925c4397f1b3362445cdae0e8`; their W&B runs are correctly
marked `crashed`, and none has held-out rows. Exact-source checkpoint recovery
then repeatedly received Colab's explicit allocation error: `Backend rejected
accelerator 'A100'. You may not have quota or entitlement for this accelerator
on your account.` These infrastructure failures are not scientific attempts.
After the local supervisor process disappeared, it was restored at 2026-07-18
03:32 UTC, reproduced the same allocation error for all three units, and
returned to guarded global backoff with no Colab session left allocated.

After the user reloaded Colab credits, exact-source recovery admitted all three
AERO units on A100s. Seeds 11, 23, and 37 each advanced through optimizer step
20, with live W&B telemetry, and independently durable private checkpoint-15
commits `259f061364acb8586fbd2d2efb928bab0808b68f`,
`d8cdbef89af1726f2215653ff2ddfd5bb8aa99b6`, and
`3e8a87345d03cc643143ee0a3969ee4b581b42e8`. Each checkpoint-15 tree has
exactly 11 files and a `trainer_state.json` proving `global_step=15` and
`max_steps=30`.

At step 20, Hugging Face rejected the new private commits with the exact error
`Private repository storage limit reached, please upgrade your plan to increase
your private storage limit`. No checkpoint-20 commit exists for any of the
three units, so checkpoint 15 remains the authoritative recovery point and no
unit is counted complete. An inventory found 37 private E1 repositories using
89,870,440,367 bytes in total before accounting for other account repositories.
Five older preflight repositories use 2,735,669,033 bytes and three are empty,
but they were not deleted: even reclaiming all of them would be insufficient for
the 14 remaining private confirmatory units and would remove recorded preflight
provenance. Completing the frozen private-HF contract therefore requires an HF
storage-plan upgrade or equivalent additional private storage capacity.
Using the live full-unit footprint (3,168,536,610 bytes) and checkpoint-15
footprint (1,584,272,134 bytes), the remaining three partial plus eleven fresh
repositories require approximately 39,606,696,138 additional bytes (39.61 GB,
36.89 GiB), before safety margin.

The campaign supervisor was deliberately stopped, the automatic monitor was
changed to prohibit relaunch until private HF capacity is restored, and all
three Colab sessions were released. This prevents further Colab-credit spend
against a known artifact-storage failure. The frozen aggregate remains
fail-closed at 26/40 validated, 14 missing, zero errors, and no verdict.

At 2026-07-18 08:10 UTC, the user's Hugging Face upgrade was independently
confirmed by the authenticated account API (`isPro=true`). The existing
three-slot supervisor was restarted and selected exact-source checkpoint-15
recovery for AERO seeds 11, 23, and 37. Seeds 11 and 23 received A100s, passed
the frozen environment check, downloaded their private checkpoint-15 trees,
and entered trainer reconstruction. Seed 37 lost its Colab connection during
package installation; the supervisor classified this as transient provider
infrastructure, restored its attempt credit to one, released the failed
session, and scheduled a guarded retry after 2026-07-18 08:27:32 UTC. The
aggregate remains 26/40 until a unit completes full training, 500-row held-out
evaluation, W&B finalization, and remote HF verification.

At 2026-07-18 08:27:49 UTC, the backoff expired and seed 37 was admitted on a
fresh A100. It passed the same frozen environment check and began restoring
the exact private checkpoint-15 source. The supervisor now independently sees
three local runners, three named remote A100 sessions, and all three occupied
slots. Seeds 11 and 23 have advanced to training steps 18 and 17, respectively.

At 2026-07-18 08:55 UTC, the upgraded Hugging Face capacity passed the first
artifact-write gate. AERO seeds 11 and 23 privately committed complete
checkpoint-20 trees at exact revisions
`6d407834b53939792e00a9a7389e665854023805` and
`f8a44fa5b71d0360d7e5c6fc353dd18adb0cb42a`. Independent downloads confirmed
all 11 checkpoint files, including the adapter, optimizer, scheduler, RNG
state, trainer state, and training arguments. Both trainer states report
`global_step=20` and `max_steps=30`; the adapter SHA-256 values are distinct
(`71e1ef266c713e31e277a21ca01622be6df5df8cc2a8eb343c1ccff9447221d4` and
`7229767bf266ea06845e45060cf067b23b2eb18ec56060b26701e16f3642dea2`).
Their W&B runs remain live beyond step 20. Seed 37 subsequently committed its
own complete checkpoint-20 tree at exact revision
`6f504b1e797263485243f4423c873e6110f9ed22`. An independent download verified
the same six resume-critical files, `global_step=20`, `max_steps=30`, and
adapter SHA-256
`aa4206feddfb5a4c0d80df835f09d7ea89066bfef6765629be01adfa1b0cf1be`.
All three active AERO units therefore have durable private checkpoint-20
recovery points. The aggregate correctly remains 26/40 until full training,
the 500-row held-out pass, W&B finalization, and strict remote reconciliation
complete.

At 2026-07-18 09:41 UTC, AERO seed 11 crossed the next durable boundary and
privately committed its complete checkpoint-25 tree at exact revision
`9c9192a194f55af9c41c0794d856e05f39e7d637`. The repository remains private
and now contains 56 files across checkpoints 5/10/15/20/25. A fresh independent
download verified the adapter, optimizer, scheduler, RNG state, trainer state,
and training arguments; `trainer_state.json` reports `global_step=25` and
`max_steps=30`, and the checkpoint-25 adapter SHA-256 is
`adf36e6145fb45ba18e407c63ea455f1aa7919c4393c737257e909e25da5bc2f`.
Its W&B run remains live beyond the save. Seeds 23 and 37 simultaneously
advanced to completed steps 24 and 22, respectively, with live W&B telemetry
and their independently verified checkpoint-20 recovery trees unchanged.
No partial AERO unit is counted in the 26/40 aggregate.

At 2026-07-18 09:47 UTC, AERO seed 23 also privately committed a complete
checkpoint-25 tree at exact revision
`588a35a656323bcbf7494e0807156ef0e6445907`. Its private repository now has
56 files across checkpoints 5/10/15/20/25. A fresh independent download
verified the six resume-critical files; `trainer_state.json` reports
`global_step=25` and `max_steps=30`, and the adapter SHA-256 is
`1ef6c121468aaffaef69f7c9ba860d345ec0da35fdf38466629836b805e24541`.
The W&B run remains live after the save. Seed 11 has advanced to completed step
26, while seed 37 remains live beyond its verified checkpoint-20 recovery
point. The aggregate remains fail-closed at 26/40.

At 2026-07-18 10:00 UTC, AERO seed 37 completed the checkpoint-25 set at exact
private revision `1ec05bd995447e56d93594b65af76a0576f9599f`. Its repository
now also has 56 files across checkpoints 5/10/15/20/25. A fresh independent
download verified the adapter, optimizer, scheduler, RNG state, trainer state,
and training arguments. The trainer state is exactly `global_step=25` with
`max_steps=30`, and the adapter SHA-256 is
`46da25bd0605cc314a9b858c048801891be3945fafa57a4d9a352d1d16d8274d`.
All three active AERO seeds therefore have independently audited checkpoint-25
recovery points. Seed 11 has advanced to completed step 28; all three W&B runs
remain live. No AERO unit is counted before checkpoint 30 and the complete
held-out evaluation pass.

At 2026-07-18 10:24 UTC, AERO seed 23 crossed the final training boundary.
Checkpoint 30 was committed to the private repository at exact revision
`7c3f1493880c96b35d9a0577893daab28ccaafc7`, bringing the repository to 67
files across all required checkpoints 5/10/15/20/25/30. A fresh independent
download verified the adapter, optimizer, scheduler, RNG state, trainer state,
and training arguments. The trainer state is exactly `global_step=30` with
`max_steps=30`; the checkpoint-30 adapter SHA-256 is
`75c062f1bd029f4f40da4180e2f4641173507db8068711d0a15f610f76f24839`.
The held-out evaluation then started and durably committed progress 16/500 at
revision `c3aa51017639cbe198781720cc317aa18ef6c48c`. W&B remains live. This is a
verified recovery and evaluation boundary, not yet a completed unit; the
aggregate remains fail-closed at 26/40 until all 500 rows, final artifacts,
W&B finalization, and strict reconciliation pass.

At 2026-07-18 10:28 UTC, AERO seed 11 also committed its complete
checkpoint-30 tree at exact private revision
`ca0d9556d1d56b98edf51d6dd9dcc6f6f9143298`. A fresh independent download
verified all six resume-critical artifacts. The trainer state is exactly
`global_step=30` with `max_steps=30`, and the checkpoint-30 adapter SHA-256 is
`d34a2608a49cce40cdeb1c427823357ffcada6cd57a9dc38984c0f4a9f2123d6`.
Held-out evaluation immediately committed progress 16/500 at revision
`96cc983d98dfa88ff3fd59670a06b2cedd1e92ce`. In the same poll, seed 23 had
advanced to a durable 32/500 evaluation commit
`04516d5770325de7afccb73ba4ed87c88345e327`. Both W&B runs remain live, seed
37 remains live at completed step 27, and the aggregate remains 26/40 pending
full evaluation and strict final reconciliation.

At 2026-07-18 10:52 UTC, AERO seed 37 completed the final training boundary
and committed checkpoint 30 at exact private revision
`12ac8fa85a62ba2febda7651ed1e278300c624c8`. Its repository contains all six
required checkpoint trees 5/10/15/20/25/30 and 67 files. A fresh independent
download verified the adapter, optimizer, scheduler, RNG state, trainer state,
and training arguments. The trainer state is exactly `global_step=30` with
`max_steps=30`, and the adapter SHA-256 is
`3e7a7c06e5d9bfdf5a67053aad616e80707a7a503bd21aa8de3e76b1fff757c1`.
Evaluation then durably reached 16/500 at revision
`d6f13ad86cfae44c1bd3337b272c999c2044ef30`. Seeds 11 and 23 concurrently
advanced to 112/500 and 128/500, respectively. All three active AERO units are
now in held-out evaluation with live W&B provenance. The aggregate remains
26/40 pending complete 500-row records, final artifacts, finished W&B runs,
and strict reconciliation.

At 2026-07-18 12:11--12:19 UTC, AERO seeds 11 and 23 completed all 500 held-out
examples and passed strict independent reconciliation. Their final private Hub
commits are `2affa816a79574e2c6e9b38915b2f58daf5fd9cf` and
`92123d35e401ff9c6c42494a383116cfbd91479e`; their finished W&B runs are
`0d4c16f9` and `3573a421`. Fresh downloads verified contiguous 500-row traces,
all six Trainer checkpoint trees, the frozen stack and unit fingerprints, and
the final adapter hashes. The seed-23 local transport remained blocked after
remote completion, so its already-immutable final manifest was reconciled with
the allocation-free recovery path and the completed local record was written
atomically. The aggregate now validates 28/40 with 12 missing units, zero
errors, and no verdict. Seed 37 subsequently reached a durable 448/500 trace at
private commit `e6b584d4bcfbdada58c19b46efa0b063604fc280` before Colab reclaimed
its session. A fresh download verified trainer state 30/30, 448 contiguous
indices, and 280 correct rows under unit fingerprint
`7547ba194cc459fda76781723343875d28cb0d9549ac81c5a83fd2a8a3a996b9`;
checkpoint 30 and all 448 rows remain recoverable. The attempted
AERO seed-53 backfill was rejected
before VM assignment with `Backend rejected accelerator 'A100'. You may not
have quota or entitlement for this accelerator on your account.` The attempt
was refunded, as was the seed-37 session loss, and the supervisor entered its
guarded provider backoff without altering scientific state. No Colab session is
currently allocated. The 2026-07-18 18:10 IST retry selected seed 37's exact
evaluation recovery plus fresh AERO seeds 71 and 89, but Colab rejected all
three A100 assignments before VM creation with the same quota-or-entitlement
message. Every attempt was refunded. At 18:26 IST, the user's Hugging Face
upgrade was independently proven with a successful private-repository create,
upload, read-back, and cleanup probe. The expired Colab backoff then admitted
seed 37 in exact evaluation recovery plus fresh AERO seeds 107 and 131, but
Colab again rejected all three A100 assignments before VM creation with the
same quota-or-entitlement message. The supervisor refunded all three attempts,
left seed 37's checkpoint-30 and 448-row recovery state unchanged, and entered
guarded backoff until 2026-07-18 18:41:13 IST. At 18:41 IST, the next guarded
retry selected seed 37 evaluation recovery plus fresh DAPO seed 37 and GSPO
seed 23, but Colab rejected every A100 request before VM creation with the
same quota-or-entitlement error. The three attempts were refunded, no remote
session was created, and the next guarded retry was 2026-07-18 18:56:44 IST.
That retry selected seed 37 evaluation recovery plus fresh GSPO seeds 37 and
53; Colab again rejected all three A100 assignments before VM creation with
the same error. The supervisor refunded every attempt and entered guarded
backoff until 2026-07-18 19:12:17 IST. A non-allocating OAuth diagnostic now
proves that the CLI is authenticated as the verified Google identity
`arvindcr4@gmail.com` and that Colab's A100 eligibility GET succeeds; the
failure is isolated to the provider's POST allocation gate. Google's
[official Colab FAQ](https://research.google.com/colaboratory/faq.html) states
that paid plans still have dynamically changing resource availability and do
not guarantee a particular premium GPU. A Google Colab maintainer has also
previously attributed paid-user A100 allocation failures to
[provider capacity](https://github.com/googlecolab/colabtools/issues/5342#issuecomment-2902631046).
That historical incident is supporting evidence for the current diagnosis,
not proof of the present backend's undisclosed reason.
At 19:12 IST, the next guarded retry selected seed 37 evaluation recovery plus
fresh Dr.GRPO seeds 23 and 37. Colab rejected all three A100 POST allocations
before VM creation with the same error; the supervisor refunded every attempt
and entered guarded backoff until 2026-07-18 19:27:28 IST.
At 19:27 IST, the next retry selected seed 37 evaluation recovery plus fresh
AERO seeds 53 and 71. Colab again rejected all three POST allocations before
VM creation. Every attempt was refunded, and the next guarded retry is
2026-07-18 19:43:09 IST.
At 19:43 IST, the next retry selected seed 37 evaluation recovery plus fresh
AERO seeds 89 and 107. Colab again rejected all three A100 POST allocations
before VM creation with the same quota-or-entitlement error. No Colab session
was created, every pre-assignment attempt was refunded, and the next guarded
retry is 2026-07-18 19:58:50 IST.
At 19:58 IST, the next retry selected seed 37 evaluation recovery plus fresh
AERO seed 131 and DAPO seed 37. Colab again rejected all three A100 POST
allocations before VM creation with the same quota-or-entitlement error. No
Colab session was created, every pre-assignment attempt was refunded, and the
next guarded retry is 2026-07-18 20:14:21 IST.
At 20:14 IST, A100 capacity returned. Colab admitted all three requested
sessions: AERO seed 37 exact-source evaluation recovery, plus fresh GSPO seeds
23 and 37. The authoritative session list reports an A100 for each session;
all three completed environment verification on CUDA 12.8 with the frozen
training stack and entered their recovery or training paths. The supervisor is
again at the proven three-session ceiling and will backfill released capacity.
GSPO seed 23 W&B run `1c16ac07` and GSPO seed 37 W&B run `60e5b791` are both
live; their corresponding private HF repositories exist with initialization
commits. Mandatory checkpoint verification begins at step 5.
The resumed AERO seed-37 evaluator completed the remaining 52 examples and
durably reached 500/500. Finalization initially rejected its immutable
`aero/rollouts_cumulative=436` telemetry because the recovery finalizer applied
the fixed-arm minimum of 480. AERO's frozen three-initial-plus-one-rescue design
generates 12--16 real rollouts per step, so its valid 30-step range is
360--480. The arm-specific bound and shared-log failure classifier were
corrected under tests; a second allocation-free finalization pass reused the
500/500 remote trace. Independent verification accepted W&B run `7547ba19`,
private HF commit `613d3f70cc0fb89eae5775a398d6f66136d14f49`, checkpoints
5/10/15/20/25/30, the byte-identical local/remote manifest, all frozen
fingerprints, and 500 contiguous hashed rows with 318 correct. The aggregate
now validates 29/40 with 11 missing units, zero errors, and no verdict. The
freed A100 slot was backfilled with fresh GSPO seed 53.

At 2026-07-18 17:36 UTC, GSPO seed 23 completed its frozen run. Independent
reconciliation accepted finished W&B run `1c16ac07`, private HF commit
`d1a6f1879bdee3eb41f65cdfc1ce34606641d828`, checkpoints
5/10/15/20/25/30, matching stack and treatment fingerprints, and a
byte-identical local/remote manifest containing 500 contiguous completion
hashes with 319 correct predictions. The manifest SHA-256 is
`2eff8815f458788ba09895d40012b6c9a0ff9877be8791f6649e8f216d88cea2`.
The fail-closed aggregate advanced to 30/40 with 10 missing units, zero errors,
and no verdict. GSPO seeds 37 and 53 remain live while the supervisor releases
and backfills the completed seed-23 session.

At 2026-07-18 17:43 UTC, GSPO seed 37 completed and independently reconciled.
Finished W&B run `60e5b791`, private HF commit
`a9c9ee88756d4c10e67925964f7b4a3662e354ad`, all six checkpoint trainer
states, the frozen fingerprints, and a byte-identical local/remote manifest
agree on 321 correct predictions across 500 contiguous hashed rows. The
manifest SHA-256 is
`73ff2239a2ac7fe3f7618afca53e937be018222014162a91a6d4b5d2395204d9`.
The aggregate advanced to 31/40 with 9 missing units, zero errors, and no
verdict. Its released A100 was immediately backfilled with Dr.GRPO seed 37;
the live lineup is GSPO seed 53 plus Dr.GRPO seeds 23 and 37.

At 2026-07-18 17:58 UTC, GSPO seed 53 completed, closing the eight-seed GSPO
arm. Independent reconciliation accepted finished W&B run `ada5a2c2`, private
HF commit `6b58a60fb73d0ac1f34e77e52aac9083877fb7b7`, all six checkpoint
trainer states, frozen fingerprints, and a byte-identical 500-row manifest with
316 correct predictions. The manifest SHA-256 is
`ad6de89244fde05c7b69975bf81ae65459c0580e8980674e82ec3b4b7729133f`.
The aggregate advanced to 32/40 with 8 missing units, zero errors, and no
verdict. Dr.GRPO seed 37 remains live; Dr.GRPO seed 23 and the next missing unit
are queued behind the transient-provider backoff after a pre-training Colab
transport loss.

At 2026-07-18 20:40 UTC, Dr.GRPO seed 37 completed and passed independent
reconciliation. Finished W&B run `5610e4f2`, private HF commit
`26e36e5e915f05922d9792014e2afecd18f6f364`, all six exact checkpoint
trainer states, and the frozen stack and Dr.GRPO treatment fingerprints agree.
The remote progress trace and remote/local manifests contain the same 500
contiguous completion hashes with 311 correct predictions (`0.6220`) and
canonical trace SHA-256
`cf82a9e27c1cfd322e6fe4dbd901d59cd2ac4b0d5d6bb4858fa1b3bffe7cca54`.
The aggregate advanced to 33/40 with 7 missing units, zero errors, and no
verdict. Its released A100 was immediately backfilled with AERO seed 89 while
AERO seeds 53 and 71 continued from independently verified checkpoint 15.

At 2026-07-18 21:44 UTC, Colab stopped listing all three active AERO sessions
for three consecutive supervisor polls. The stale local transports were
terminated without deleting remote evidence or consuming a scientific attempt.
AERO seed 53 had reached W&B global step 23 and remains recoverable from its
independently verified private checkpoint 20 at commit
`7f044084d43b5632204032b39ed5d1856b930a2e`; seed 71 had reached global step
25 and its complete 11-file checkpoint-25 tree was independently read back at
commit `ce3c740552eb17980e3cdd504239407c80598ff4`; seed 89 had reached global
step 7 and remains recoverable from its verified checkpoint 5 at commit
`c45d78465df340f1ca910411b3e06bff4ecc964e`. Guarded exact-source resume
waves at 21:44 and 21:59 UTC were rejected before VM creation with the exact
Colab error `Backend rejected accelerator 'A100'. You may not have quota or
entitlement for this accelerator on your account.` All three pre-assignment
failures were refunded, no remote Colab session remains allocated, and the
persistent supervisor is waiting under provider backoff for A100 capacity to
return. The aggregate therefore remains 33/40 rather than crediting any partial
run.

At 2026-07-19 04:21 UTC (09:51 IST), the user-reloaded Colab account again
admitted an A100 capacity probe. The probe was immediately released, and the
three-slot supervisor was restarted in persistent terminal session `7348`.
Exact-source AERO resumes 53, 71, and 89 all received NVIDIA A100-SXM4-40GB
sessions, passed the frozen environment check, downloaded authoritative private
checkpoints 20, 25, and 5, and reconnected to the original W&B run IDs
`bd8d23f9`, `c73138a3`, and `201279c6`. Reconstruction then completed and the
three W&B runs reported post-resume optimizer steps 21, 26, and 6,
respectively, proving continuation from authoritative checkpoint trainer states
20/25/5. The aggregate remains 33/40 until full runs pass the normal final
training and held-out gates.

At 2026-07-19 04:53 UTC, AERO seed 53 uploaded and independently verified a
stronger private recovery point: checkpoint 25 at exact HF commit
`50a57e24a8db17f93cc1f00b5ccbe4d040e3a210`. The repository is private, the
commit resolves exactly, and the downloaded trainer state records
`global_step=25` with `max_steps=30` and intact AERO metrics. The three live
trainers are computing steps 26, 29, and 9 for seeds 53, 71, and 89,
respectively; their authoritative recovery points are now checkpoints 25, 25,
and 5. No partial unit enters the aggregate before checkpoint 30 and the
500-row held-out audit complete.

At 2026-07-19 05:06 UTC, two more private recovery boundaries passed
independent exact-revision verification. AERO seed 71 completed training at
checkpoint 30, commit `39deaf8e358edc7d19cf9548276d0253e5a77571`, with
`global_step=30`, `max_steps=30`, and intact AERO metrics; its held-out
evaluation then reached 48/500. The exact private 48/500 snapshot at commit
`97c037dc58b55c33714c566e72c7a6a7b27feded` independently verifies contiguous
indices 0--47, valid completion hashes, 29/48 correct, and the frozen unit
fingerprint. AERO seed 89 committed checkpoint 10
at `cf36173bcfa9373dcae7b7342757766132b861f8`, with `global_step=10`,
`max_steps=30`, and intact AERO metrics. The authoritative private recovery
points are now checkpoints 25, 30, and 10 for seeds 53, 71, and 89. Seed 71 is
not accepted until its evaluation reaches 500/500 and its final adapter,
manifest, W&B finish, and frozen fingerprints all verify.

At 2026-07-19 06:50 UTC, AERO seed 71 passed every final evidence gate and was
accepted as unit 34/40. Its finished W&B run is `c73138a3`; exact private HF
commit `5168da9f98c37cd432e60b14443436daaaed6951` contains checkpoints 5, 10,
15, 20, 25, and 30, the final adapter and manifest, and a 500-row held-out
trace with contiguous indices, valid completion hashes, and 316/500 correct
(0.6320). The local and remote manifests match, including frozen stack
fingerprint `4737be74fdc97dd400b846e93c6b0c03f443ee65da2cc89697bf5b078d16aa0d`
and unit fingerprint
`c73138a35898430abdedf1226065576bda48e03558c74ff7e62f8a8a9ec6b7e8`.
Its released A100 was immediately backfilled with AERO seed 107. AERO seed 53
has completed training and independently verified checkpoint 30; its current
exact private HF commit `6a2d3e706d38d7dea596b67f88bce5d9de4da932` contains a contiguous,
hash-valid 320/500 trace with 193 correct (0.6031). AERO seed 89 remains
recoverable from verified checkpoint 20 at
`b001dd2357bde9d4927fdf11f48871330a42631a`. AERO seed 107 is running under
W&B `853facf3` in private repository
`arvindcr4/tinker-rl-lab-e1-aero-s107-853facf3`. All three A100 sessions and
the supervisor are healthy; the fail-closed aggregate is 34/40 with six units
missing and no verdict emitted.

At 2026-07-19 07:36 UTC, AERO seed 53 passed every final evidence gate and was
accepted as unit 35/40. Finished W&B run `bd8d23f9` and exact private HF commit
`448707a7a1dce3f705204d90d06598a971cfbed2` reconcile checkpoints 5, 10, 15,
20, 25, and 30, the final adapter and manifest, and 500 contiguous held-out
rows with valid completion hashes and 316/500 correct (`0.6320`). Exact trainer
states report the expected `global_step` at every checkpoint and
`max_steps=30`; the remote and local manifests have identical SHA-256 digest
`c011fab69d3277d9bdfd5073ce1247ebef73261ec1879498cf0340b20c9c7d87`.
The aggregate advanced to 35/40 with five missing units, zero errors, and no
premature verdict. The released A100 was immediately backfilled with fresh
AERO seed 131, leaving AERO seeds 89, 107, and 131 active on three verified
A100 sessions.

At 2026-07-19 10:08 UTC, AERO seed 89 passed every final evidence gate and was
accepted as unit 36/40. Finished W&B run `201279c6` and exact private HF commit
`43421d47afa9a82453e883508a634f55ff770f14` contain checkpoints 5, 10, 15,
20, 25, and 30, the final adapter and manifest, and 500 contiguous held-out
rows with 315/500 correct (`0.6300`) and 500 unique valid completion hashes.
Every trainer state reports the expected `global_step`, `max_steps=30`, and a
contiguous training history; the remote and accepted local manifests are
byte-identical with SHA-256
`1ecc0af8f374cf430375ae5db3ae6422678fd413b01c59143c6e44a6d59713d9`.
The aggregate advanced to 36/40 with exactly four missing units, zero errors,
and no premature verdict. Its released A100 was immediately backfilled with
fresh DAPO seed 37, whose environment preflight verified an A100-SXM4-40GB.
AERO seed 107 independently reached private checkpoint 20 at commit
`f412a1e4d81ca7473a5036411d9a7fb58268c557`; AERO seed 131 independently
reached private checkpoint 15 at commit
`c4a68af97c99679cc37a37fcba533873d6ecd25c`. Both exact trainer states have
the required global step, `max_steps=30`, contiguous histories, and intact
AERO treatment metrics. The three-session supervisor remains healthy.

At 2026-07-19 13:23 UTC, AERO seed 107 completed its exact-source recovery and
passed every final evidence gate. Finished W&B run `853facf3` and exact private
HF commit `42980267a519c028dd0e5a0592c0a106933fb0b7` contain checkpoints 5, 10,
15, 20, 25, and 30, the final adapter and manifest, and 500 contiguous held-out
rows with 315/500 correct (`0.6300`) and 500 unique valid completion hashes.
The accepted local and remote manifests are byte-identical with SHA-256
`8da9d6a49ae2961bcc291ad2c12164b0c83a69ff0bed7564431d04f4347a7632`.
The aggregate advanced to 37/40 with exactly three missing units, zero errors,
and no premature verdict. Its released A100 was immediately backfilled with
fresh Dr.GRPO seed 23, which passed the frozen A100 environment check; AERO
seed 131 and DAPO seed 37 continue held-out evaluation from independently
verified checkpoint-30 revisions.

At 2026-07-19 15:00 UTC, DAPO seed 37 passed every final evidence gate.
Finished W&B run `605e7589` and exact private HF commit
`bb7fb354be9f02461e94f69d226b3566e8417368` contain all six checkpoint trees,
the final adapter and manifest, and 500 contiguous held-out rows with 314/500
correct (`0.6280`) and 500 unique valid completion hashes. The accepted local
and remote manifests are byte-identical with SHA-256
`15448dfcde4a095d782b82fa259bb2fcc318a66e780389a0fa987e10d5b1038e`.
The aggregate advanced to 39/40 with only Dr.GRPO seed 23 missing, zero errors,
and no premature verdict. Dr.GRPO seed 23 remains live on its original A100 and
continues held-out evaluation from its verified checkpoint-30 revision.

At 2026-07-19 16:17 UTC, Dr.GRPO seed 23 completed and passed every final
evidence gate. Finished W&B run `63941052` and exact private HF commit
`4318ffd9026b64749561c033ab7c0f3e3841cb84` contain all six checkpoint trees,
the final adapter and manifest, and 500 contiguous held-out rows with 313/500
correct (`0.6260`) and 500 unique valid completion hashes. The accepted local
and remote manifests are byte-identical with SHA-256
`3057541e3364213fa2ee36dfdfde995cf78b59c49fecfa50665739dcdb8a07de`.
Every checkpoint trainer state reports its exact step, `max_steps=30`, and a
contiguous history. The fail-closed aggregate now accepts 40/40 units and emits
the preregistered verdicts: DAPO `DISAPPEARS`; GSPO, Dr.GRPO, and AERO
`INCONCLUSIVE`. The campaign supervisor exited cleanly and released the last
Colab A100 session.

This was the 2026-07-19 16:17 UTC acceptance state before the subsequent
campaign-wide completion-hash audit. That audit found six legacy unhashed
prefixes, hardened both the campaign and aggregate validators, and correctly
reopened E1 at 34/40 with all verdicts withheld. The original records are
preserved under `results/full/legacy-unhashed-2026-07-19`; exact checkpoint-30
evaluation-only repair is active for GRPO seeds 11, 89, and 107, DAPO seed 131,
and GSPO seeds 11 and 71.

At 2026-07-19 18:19 UTC, GRPO seed 107 completed exact checkpoint-30
evaluation repair and passed the independent campaign verifier. Finished W&B
run `5f0a7bf0` and exact private HF commit
`ffcbfffd322a181e82c0d3a552ec611432dce471` reconcile all six trainer states,
the final adapter, frozen fingerprints, and a byte-identical manifest with
SHA-256 `792e11fff9154c6d421322d065ef1c3be296054426fe5ed3c4a212cb97103dc0`.
Its 500 contiguous held-out rows contain 500 unique valid completion hashes and
317 correct answers (`0.6340`). The fail-closed aggregate advanced to 37/40
with three missing units and no verdict; the released A100 was immediately
backfilled with GSPO seed 71 evaluation repair.

At 2026-07-19 18:30 UTC, GRPO seed 11 completed exact checkpoint-30
evaluation repair and passed the independent campaign verifier. Finished W&B
run `78393a67` and exact private HF commit
`b39028ea32b042247e7ecd3ee228b8b302c55226` reconcile all six trainer states,
the final adapter, frozen fingerprints, and a byte-identical manifest with
SHA-256 `b08c10604b0222b5296fb4345931401d9e5ac207cfc2260108786d3904a3f343`.
Its 500 contiguous held-out rows contain 500 unique valid completion hashes and
325 correct answers (`0.6500`). The fail-closed aggregate advanced to 38/40
with only GSPO seeds 11 and 71 missing and no verdict.

At 2026-07-19 18:55 UTC, GSPO seed 11 completed exact checkpoint-30
evaluation repair and passed the independent campaign verifier. Finished W&B
run `5f4fc0d3` and exact private HF commit
`f064274b7c79372fe3fa1501737ae8a5398bec07` reconcile all six trainer states,
the final adapter, the frozen GSPO treatment fingerprint, and a byte-identical
manifest with SHA-256
`bf40c2437315d306ff7d22737d3d750e7950969b9c8ff9612d7f5add4acdc592`.
Its 500 contiguous held-out rows contain 500 unique valid completion hashes and
320 correct answers (`0.6400`). The fail-closed aggregate advanced to 39/40
with only GSPO seed 71 missing and no verdict; its completed A100 session was
released.

At 2026-07-20 13:19 IST, Colab again admitted the frozen A100 recovery for
GSPO seed 71. The evaluator reconstructed exact checkpoint 30, resumed at the
verified `next_index=416` boundary, and completed the final 84 held-out
examples. Finished W&B run `10ef44ab`, exact private HF commit
`c0c0b968a61a9ade251b5b7e6ece3119197dc1b1`, all six checkpoint trees, the
final adapter, frozen fingerprints, and the byte-identical manifest (SHA-256
`0243e256a3cdae62d30cad889f8bdf19bfe6d7cc2862edc60edffc23586ac3ed`)
reconcile with the accepted local record. The final trace contains indices
0--499, 500 valid unique completion hashes, and 319 correct answers (`0.6380`).
The full verifier accepted all 40 units with zero errors, and the frozen
aggregate emitted the final verdicts: DAPO `DISAPPEARS`; GSPO, Dr.GRPO, and
AERO `INCONCLUSIVE`.

## Preflight and recovery record

The one-step A100 preflight is explicitly `preflight-not-evidence`. It logged to
finished W&B run
[`4bb501e5`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab/runs/4bb501e5)
and stored its checkpoint and manifest in private HF repository
[`arvindcr4/tinker-rl-lab-e1-grpo-s11-4bb501e5`](https://huggingface.co/arvindcr4/tinker-rl-lab-e1-grpo-s11-4bb501e5/tree/07281c538e72efb09ee3edfc636c35b5fa1cf161).
It verified the A100, Qwen3-8B LoRA path, pinned data/model revisions, W&B, HF,
and result schema before the full run.

Training uploaded a resumable HF checkpoint every five optimizer steps. The
initial held-out pass later stalled after 192 examples because the training
configuration retained `use_cache=False`. Training was not rerun. The recovery
loader downloaded checkpoint 30, disabled gradient checkpointing, enabled a
dynamic generation cache, and resumed evaluation from an HF-synced progress
file saved every 16 examples. The first 16 recovery examples used batch size 4;
the remaining 484 used batch size 8. The remote manifest records the final
recovery batch size, while the two local recovery request files preserve both
phases.

The post-run review improved the workspace scripts, so their current hashes no
longer equal the executed hashes. A byte-level in-memory reversal of only those
review edits reproduces the executed training SHA, evaluator SHA, and both
recorded recovery fingerprints; the derivation is stored in
`results/colab-e1-confirmatory/provenance/grpo-seed-11-executed-source.json`.
Future launcher requests store one content-addressed source snapshot per SHA,
reusing identical snapshots rather than duplicating them per run.

Trackio was tested as an optional third telemetry sink, but hosted Space
creation returned HTTP 402 because it requires HF PRO. It was removed from the
required contract before the confirmatory run. The two user-required sinks,
W&B and private HF checkpoints, were both verified after completion.

Resume/finalize the same scientific unit, if its local wrapper ever needs to be
reconstructed from remote state, with:

```bash
python3 zvf-program/audit/run_colab_e1_evaluation.py \
  --source-request \
  zvf-program/audit/results/colab-e1-confirmatory/requests/e1__grpo__s11__78393a67980d.json \
  --eval-batch-size 8
```

The evaluator discovers `evaluation/progress.json`; at 500/500 it performs
finalization and remote verification without regenerating completed examples.

## Earlier T4 pilot campaign

Four seed-11 open-trainer pilots remain useful infrastructure evidence but are
not confirmatory: GRPO (`plu9uoil`), Dr.GRPO (`gr89mjo2`), DAPO (`3hr07bss`),
and Adaptive-G (`3u64sgr3`). They used Qwen2.5-0.5B-Instruct and synthetic
addition. Their private HF artifacts are in
[`arvindcr4/tinker-rl-lab-colab-obligations`](https://huggingface.co/arvindcr4/tinker-rl-lab-colab-obligations).
They do not enter `results/full` and cannot satisfy any frozen E1 arm.

## Remaining execution obligations

- E1 is closed: all 40 frozen units pass strict local, W&B, private-HF
  checkpoint, stack-fingerprint, treatment, manifest, and 500-row unique-hash
  verification. `campaign-verification.json` and `audit.json` both report
  `COMPLETE`; DAPO `DISAPPEARS`, while GSPO, Dr.GRPO, and AERO are
  `INCONCLUSIVE`.
- E8 flagship conformance screening is stopped at the infrastructure gate.
  The balanced-regime seed-11 corpus exhausted its three guarded attempts on
  2026-07-21 without any scientific unit running: attempt 1 (W&B `ujryg527`)
  lost its VM at the group-80 frozen profiler boundary (last committed row
  group 78, 309,364 cumulative charged tokens); attempt 2 (`lwjtk9dk`) lost
  its VM during the group-100 frozen profiler (last committed row group 99,
  393,714 charged tokens); attempt 3 (`hge0xhav`) lost its VM at the
  identical point (group 99, 393,714 charged tokens), confirming
  deterministic replay. Every loss occurred roughly 2h20m into the session,
  consistent with provider-side runtime reclamation rather than credit
  exhaustion. The frozen corpus design carries no partial checkpoint, so
  each loss forces full regeneration. All six corpus jobs are
  `failed_infrastructure` at their attempt caps; all 24 scientific units
  remain pending and fail-closed; all Colab sessions are released and no
  version-1 flagship process remains active.
- On 2026-07-22 the user authorized versioned amendment
  `A1-corpus-intermediate-persistence`. The amended protocol preserves every
  scientific field and all version-1 failures, but adds atomic private-Hub
  corpus-prefix commits at groups 20/40/60/80, strict protocol/source/order/
  runtime/artifact/token/FLOP validation before resume, a three-VM cap per
  version-2 corpus job, and one-corpus-at-a-time scheduling. The final verifier
  independently validates the referenced group-80 commit as well as the
  complete 100-group corpus. The exact pinned Python 3.12 suite passes 106/106;
  the focused amendment suite passes 55/55 under repository Python 3.12.
  A first version-2 launch exposed a missing `runtime_install.py` in the source
  archive before W&B or group generation. Implementation revision 2 binds the
  complete source bundle, preserves attempt logs before retry, and retries only
  recognized provider failures. Its separate source-bound A100 smoke is now
  accepted, and balanced-equal-length seed 11 completed all 100 groups as W&B
  run `b8eoqd09` from `launch-v2-corpus-resume-r1/`. The first real group-20
  prefix is durable at private-Hub commit
  `46030fba999dccbabc40567ab8f605589aa6a50a`; an independent local verifier
  downloaded and re-hashed the source manifest and every group artifact,
  accepting its exact fingerprint and 80,081-token prefix ledger. The same
  independent verifier accepted the replacement group-40 prefix at exact
  private-Hub commit `55091520f883bec456fe3f3334edf68dbc770013`, with all
  40 groups and its 160,423-token prefix ledger intact. It then accepted the
  group-60 replacement at exact private-Hub commit
  `4776e185ee8a91e924672179062380fb9423bddb`, with all 60 groups and its
  236,615-token prefix ledger intact. The final resumable group-80 prefix was
  independently accepted at exact private-Hub commit
  `2faf00b02c5c81fcdcd2c4ed9e97e5fa8b721101`, with all 80 groups and its
  317,482-token prefix ledger intact. The final verifier then independently
  re-hashed the complete 100-group corpus, source manifest, token/FLOP ledger,
  and exact group-80 commit. It accepted corpus fingerprint
  `8b24a0520a97f0d5101c2662a1e3e369e8342c1759c9963a0ccb909b01525589`
  at private-Hub commit `91ec135ce5ffd562d991e535a16cae28c6552389`,
  reconciling 396,672 charged generated tokens and zero resumes with finished
  W&B run `b8eoqd09`. The foreground controller exited while the remote run
  survived; its stale PID and verified no-duplicate adoption are preserved in
  a recovery receipt. The completed session was released and a launchd-owned
  supervisor now continues the remaining DAG. One of six corpora is accepted;
  no scientific unit is accepted yet.
- Later on 2026-07-22, the first two r1 scientific units failed before
  optimizer step 1 when Qwen3 gradient-checkpoint recomputation escaped the
  deterministic SDPA context. The corrected r2 non-scientific A100 smoke then
  exposed a second fail-closed defect: cosine receipts
  `1.00221848487854` and `1.0022610425949097`, while the verifier enforced
  finiteness but not the mathematical `[-1,1]` range. The queued r2 corpus was
  stopped during runtime installation before W&B, private-Hub, group, or
  scientific initialization. Implementation revision 4 preserves and excludes
  all r1/r2 artifacts, computes diagnostics in float64, and validates cosine,
  relative-L2, and norm bounds in preflight and full records. Its exact pinned
  gates pass 109/109 and 55/55. Fresh r3 protocol SHA-256 is
  `04d20f712f652f80754fa4c8c0a3f48d4d2f1c5d716b3981746322c938b21970`;
  LaunchAgent `ai.openai.codex.flagship-pilot-v2-r3` is active with only the
  fresh non-scientific smoke initially launched. Attempt 1 is independently
  accepted on `NVIDIA A100-SXM4-40GB` with gradient cosine
  `0.999957795529626`, selected-vs-intended cosine
  `0.9999999999997982`, positive norms/FLOPs, exact runtime pins, and exact
  source hashes. Its session was deleted; balanced-equal-length seed 11 corpus
  attempt 1 is the only running downstream job. The current eligible count
  remains zero r3 corpora and zero scientific units.
- The same r3 balanced-seed-11 run published its first immutable prefix at
  group 20. Independent local verification downloaded and re-hashed the source
  manifest and all 20 group artifacts, accepting exact private-Hub commit
  `7c6d13ee7b22ef1a9ca83f2a550a43fbcff8a7e9`, fingerprint
  `a054b9c6f1ce9a69424677f201c46c242c805bb22674e8744fedb381e3fe556b`,
  80,081 charged tokens, profiler steps 1/20, one W&B attempt (`3jpcepfy`),
  and zero resumes. The same A100/controller remain live. This prefix proves
  restartability but is not a complete corpus or scientific observation, so
  the eligible counts remain zero and all downstream units stay gated.
- The running r3 corpus then published and independently passed its group-40
  replacement at exact private-Hub commit
  `b23d1da97dc5dadd3da6d133ba3ffb048d055af0`, fingerprint
  `5c1a6cf763737d63efa116e1bac67a5061e06f34dbd360ae6e1fefd7b42dda3b`.
  All 40 group artifacts and the source manifest re-hash exactly, reconciling
  160,423 charged tokens, profiler steps 1/20/40, one W&B attempt
  (`3jpcepfy`), and zero resumes under the exact A100/runtime bindings. The
  same controller continues toward group 60; the prefix is restartable
  infrastructure, not an accepted corpus, so downstream units remain gated.
- The group-60 replacement then independently passed at exact private-Hub
  commit `a0c83171731c497ce13ae1dcc14b48b045c72956`, fingerprint
  `dd7caf181a7463196d86d404ea21ff2fe5b88e8878f388757b70b8a268ff5790`.
  Independent download and re-hash covered all 60 group files, the checkpoint
  manifest, and all 14 source entries, reconciling 236,615 charged tokens,
  profiler steps 1/20/40/60, 15,644 profiled tokens, one W&B attempt
  (`3jpcepfy`), zero resumes, and exact A100/runtime/source/order bindings.
  W&B exposes the same commit/fingerprint and the live attempt has continued
  through group 61 toward group 80. The eligible corpus and unit counts remain
  zero until the 100-group final verifier accepts the corpus.
- The final resumable group-80 replacement then independently passed at exact
  private-Hub commit `ba2a67680eee15e956f406fd9caebc83326967cf`,
  fingerprint
  `c50c78dda0978525d7bf32247087850436e844b43825234d572dc5a2ed3e4b12`.
  Independent download and re-hash covered all 80 group files, the checkpoint
  manifest, and all 14 source entries, reconciling 317,482 charged tokens,
  profiler steps 1/20/40/60/80, 19,740 profiled tokens, one W&B attempt
  (`3jpcepfy`), zero resumes, and exact A100/runtime/source/order bindings.
  W&B exposes the same commit/fingerprint and token ledger. The same attempt
  continues toward group 100; no corpus or scientific unit is eligible until
  the final record passes its independent verifier.
- The 100-group final record passed the independent full-corpus verifier at
  exact private-Hub commit `2735a27d5f18bbdaaae76494a2047b39a4318e22`,
  corpus fingerprint
  `b09c72247b168297e73ce5edf2aad59e4496e7d78257beb252e864dd1a9587f1`.
  The verifier downloaded all 185 remote files and re-hashed 100 group
  artifacts plus the manifest and all 14 source entries, reconciling 396,672
  charged tokens, profiler steps 1/20/40/60/80/100, 22,698 profiled tokens,
  exact group-80 checkpoint lineage, one finished W&B run (`3jpcepfy`), one
  attempt, zero resumes, and exact A100/runtime/source/order bindings. The
  supervisor accepted balanced seed 11 and released its Colab session. The
  eligible count is now one r3 corpus and zero scientific units. Balanced seed
  23 is the only running corpus builder; intended/native balanced-seed-11
  units are concurrently running on separate A100s.
- The first r3 unit wave failed closed before optimizer step 1. Intended W&B
  run `22107a6b` and native run `07c23895` both completed the identical step-0
  evaluation (`accuracy=0.15625`, 64,038 generated tokens) and then failed.
  Native's preserved traceback terminates at `TrainingContractError: intended
  gradient norm is non-positive or non-finite`. Immutable replay group 1 has
  eight zero rewards, so all objective advantages and gradients are correctly
  zero. A full corpus audit found 59 all-zero groups, 3 all-one groups, and 38
  mixed groups. The current positive-norm/cosine-only receipt contract cannot
  represent 62 genuine variance-starvation steps, yet the gate requires all
  100 receipts and 95 balanced-equivalence steps. Neither failed unit produced
  an optimizer step, checkpoint, final record, or eligible observation.
- The supervisor marked native `failed_validation` and exited. Its crash-only
  launch agent was unloaded to stop the persistent restart loop; both unit
  sessions were released and no replacement unit launched. The detached
  balanced-seed-23 corpus continued only through its first atomic recovery
  point, which independently passed at private-Hub commit
  `b1d897a968470898848ddb85ba24a334c3d59237`, fingerprint
  `67d51945e773e9e6aa50a88f8d72a182230c2452bd0285caf00be554b1aa1764`,
  with 80,988 charged tokens, profiler steps 1/20, one W&B attempt
  (`ge121gt6`), and zero resumes. The stop propagated after W&B row 22 (86,052
  charged tokens), but no later Hub commit exists and the orphaned run remains
  stale `running`; only group 20 is recoverable. No Colab session, supervisor,
  or launcher remains. Replacement execution requires an explicit joint-zero
  receipt/scoring and corpus-source-reuse amendment.
- E2: 50 PPO/SAO jobs still require an LLM PPO stack, SWE-Bench agentic SAO
  environment, matched budgets, and W&B/HF provenance.
- E3: 15 M-GRPO jobs still require planner/sub-agent training and a tool
  sandbox.
- E4--E6 remain compute campaigns; E7 still requires approved real fraud data
  and privacy/ethics authority.

### A1-R4 authorization and revision-6 release gate (2026-07-22)

- The user authorized A1-R4 with corpus reuse. Revision 5 now records explicit
  `nonzero`, `joint_zero`, and named one-sided-zero relations. Zero-vector
  cosine/relative-L2 values are null; joint-zero is equivalence and zero
  effect; one-sided-zero is maximal divergence; nonzero thresholds are frozen.
  A selected zero gradient is a true AdamW no-op while the scheduler advances.
- Frozen r3 corpus generation and revision-6 unit training are separately
  bound. The full frozen corpus control archive remains exactly
  `f04aff3fb8ef87be2bc885263750c2cc0b6be6bd71fcc8b02ab5be8f116fac31`.
  Live verification accepts balanced seed 11 at final commit
  `2735a27d5f18bbdaaae76494a2047b39a4318e22` and balanced seed 23 at exact
  group-20 commit `b1d897a968470898848ddb85ba24a334c3d59237`.
- The revision-6 local release gate passes 115/115 exact pinned tests, the
  focused correction gate passes 69/69, and Ruff check/format is clean. The
  protocol SHA-256 is
  `1b001a920a042ee2a41f232175066483b4b28e5e37db2e7e9ebf48d0a561007a`.
  The next allowed external action is the fresh revision-6 A100 smoke; all
  confirmatory work remains forbidden until a screening GO verdict.
- The first revision-5 smoke installed the exact pins and then failed closed
  before model loading because its unit source bundle omitted the frozen
  archive files required by remote protocol validation. No W&B/HF/scientific
  state exists. `A1-R4.1` packages both already-hashed archives into revision
  6, preserves the r4 failure log, and moves execution to fresh r4-1 identities.
- The revision-6 smoke is independently accepted on an A100 with exact pins,
  positive required phase FLOPs/norms, intended/native cosine
  `0.999957795529626`, selected/intended cosine `0.9999999999997982`, and an
  applied optimizer update. Its session was released. Frozen balanced seed 11
  was then reaccepted at its exact pinned commit/fingerprint, unlocking the
  balanced seed-23 resume and revision-6 intended/native seed-11 units.
- Balanced seed 23 is now live as W&B `ncpafe25`, with config proving
  `corpus_start_group=20`, `corpus_resume_count=1`, and the exact bound
  group-20 commit/fingerprint. It has advanced through group 23 at 90,148
  charged tokens. Two capacity-only `TooManyAssignmentsError` exits preceded
  the live third local attempt. Intended/native seed-11 W&B runs `a0a67b52`
  and `87ba3535` are live against the pinned corpus; their step-0 evaluations
  are still running, so no revision-6 scientific receipt is counted yet.

### A1-R4.2 revision-7 correction and live release (2026-07-22)

- Revision-6 intended W&B `a0a67b52` passed a real step-1 `joint_zero` no-op
  and a nonzero step-2 intended/native comparison, then failed closed because
  byte-identical selected/intended vectors produced the reduction-roundoff
  cosine `1.000000000002599`. Native W&B `87ba3535` corroborated the expected
  joint-zero/nonzero sequence before its superseded session was released.
- `A1-R4.2-exact-identical-gradient-diagnostics` makes exact vector equality a
  direct cosine `1.0` / relative-L2 `0.0` receipt in implementation revision 7.
  The local release gate is 116/116 exact-pinned tests, 70/70 focused tests,
  byte-identical regenerated manifests, and clean Ruff check/format across all
  22 changed Python files. Protocol SHA-256 is
  `87d929d0a3af789d3ba3ee10a1f4c3e83572ecec7cc4efa28ca032008f88fbc4`;
  unit source binding is `005d3f8242b992cf70af2944c2b3f63351f5d3e00e95cdc5caeb40d1261b0918`.
- The revision-7 A100 smoke is independently accepted with intended/native
  cosine `0.999957795529626`, relative L2 `0.009205099545490102`, positive
  required FLOPs/norms, an applied optimizer update, and exact-equality
  selected/intended cosine `1.0`. Its A100 was released. Balanced seed 11 was
  reaccepted at commit `2735a27d5f18bbdaaae76494a2047b39a4318e22`,
  fingerprint `b09c72247b168297e73ce5edf2aad59e4496e7d78257beb252e864dd1a9587f1`.
- Surviving corpus W&B `ncpafe25` independently passed group 40 at private-Hub
  commit `b45dc64a59a8cd7fb068d0f2182c507c34db8aec`, fingerprint
  `1d7e72efb8df8e22beb15a9756d8255aa6b44f4f4a9f4af3d53b547143138c37`,
  158,590 charged tokens, and is live through group 58 / 230,855 tokens. Fresh
  r4-2 intended/native seed-11 A100 sessions use the `87d9005d` identity and
  are live on final allocation attempt 3. Attempt 1 failed before allocation
  due only to the LaunchAgent PATH. Attempt 2 allocated, but a host reboot at
  18:52:18 auto-reran the launchers; their duplicate-name 412 recovery failed
  and cleanup stopped the surviving sessions before W&B/HF state. The agents
  are now non-RunAtLoad, so another reboot cannot duplicate or stop remote work.
  Exactly three A100s and one corpus builder are active. Confirmatory work
  remains locked.

E1 paired confidence intervals and arm-level verdicts are now released from the
frozen aggregate. E2--E7 remain separate obligations and must not be inferred
from E1.
