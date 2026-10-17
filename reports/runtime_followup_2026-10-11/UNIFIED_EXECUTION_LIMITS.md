# Unified launch inspection is not canonical execution evidence

The local/Colab unified Tinker and SkyRL execution routes now stop before importing a provider or starting a subprocess. This leaves the frozen canonical specification and benchmark defaults unchanged.

- **Tinker:** the direct baseline implements its existing token-sum group-relative policy-gradient objective. It does not enforce the canonical gradient-accumulation, token-ratio clipping, KL, LoRA-alpha or target-module contract. Forwarding model, learning rate and group size alone did not make the algorithms equivalent. A subprocess exit also did not establish an observed optimizer-step count.
- **SkyRL:** the previous unified route launched `skyrl.tinker.api`, not a training recipe, but reported the requested training-step count when that server process exited. A separately configured server/recipe is not a verified canonical completion.
- **Unknown completion:** a returned result with `final_step=None` now prints that optimizer-step completion is unverified. In particular, successful native TRL script execution does not by itself attest a step receipt.

## Supported migration and inspection

To inspect the direct, independent Tinker baseline without credentials, downloads or provider work:

```sh
python -m platform_tinker.tinkerrl.grpo_cli --preset gsm8k --check-config
```

Its existing direct execution interface remains available for separately authorized runs. Removing `--check-config` can contact external services and consume provider resources; no such run was performed for these changes. This baseline must not be labeled a completed canonical clipped-PPO comparison.

Read-only matrix inspection remains available:

```sh
python -m platform_local.unified --framework tinker --dry-run
python -m platform_local.unified --framework skyrl --dry-run
```

A resolved plan and existing driver file demonstrate routing only. They do not prove that every canonical field is implemented, that a provider is available, or that training completed. Other backend-specific drivers retain their own constraints; this patch does not certify their equivalence.

A future canonical implementation needs explicit field mapping, objective-level parity tests, and an independently validated optimizer-step/result receipt before these guarded routes should execute.
