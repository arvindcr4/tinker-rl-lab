# S1 objective differential harness

`reference.py` is the framework-neutral Trace contract. `trl_adapter.py` and
`verl_adapter.py` evaluate frozen fixtures against exact external framework
distributions on CPU without loading a model, generating completions, or
running an optimizer. Both adapters record package versions, source paths, and
source hashes; the TRL adapter additionally verifies the locked wheel hash.

Run the common and TRL tests in their isolated environment.  Do not use test
discovery across both adapters: the pinned TRL and verl distributions require
different Python, Torch, and Transformers versions.  The repository root must
contain the bundled `uv.lock`, whose TRL wheel hash is checked at runtime.

```bash
cd /absolute/path/to/review_bundle/repository
PYTHONPATH=zvf-program uv run --isolated --no-project --python 3.12 \
  --with trl==1.2.0 \
  --with transformers==5.5.4 \
  python -m unittest \
  flagship.s1.test_reference \
  flagship.s1.test_receipts \
  flagship.s1.test_trl_adapter -v
```

Emit a machine-readable differential for one fixture/arm:

```bash
PYTHONPATH=zvf-program uv run --isolated --no-project --python 3.12 \
  --with trl==1.2.0 \
  --with transformers==5.5.4 \
  python -m flagship.s1.trl_adapter --fixture base --arm grpo
```

Run the external verl test separately and from outside the repository so Python
cannot resolve the repository-local `verl/` wrapper.  Use a fresh temporary
environment for this command:

```bash
uv venv --python 3.11 /tmp/tinker-s1-verl-py311
uv pip install --python /tmp/tinker-s1-verl-py311/bin/python \
  verl==0.3.0.post1 torch==2.4.0 transformers==4.45.2
cd /tmp
PYTHONPATH=/absolute/path/to/review_bundle/repository/zvf-program/flagship \
  /tmp/tinker-s1-verl-py311/bin/python -m unittest \
  s1.test_verl_adapter -v
```

Clean-extraction baseline on 2026-07-27: 35/35 common+TRL tests and 10/10 verl
tests passed with the commands above.

The harness intentionally reports rather than hides native formula differences.
TRL uses `sample_std + 1e-4`, DAPO global-active-token reduction, and DrGRPO
`batch_size * max_completion_length` reduction. Native verl GRPO uses
`sample_std + 1e-6` and one global masked-token mean; stable Python group IDs
are required because tensor scalar IDs fail its dictionary lookup. TRL has no
native AERO posterior-advantage API, and verl 0.3.0.post1 has no native mapping
for DAPO, GSPO, DrGRPO, or AERO. These native results are therefore
`MATERIAL_DIFFERENCE` or `NOT_TESTED`, not conformance. The intended stack
integrations now pass the canonical differential on both pinned stacks. The
fail-closed combined receipt is `results/implementation_freeze.json`, which
records `S1_PASS`, source hashes, 14 intended cases per stack, 36 controller
cases, and the preserved native verdict vectors. This closes S1 only; it does
not authorize GPU screening. See `S1_AMENDMENT.md` and the active execution
notes for the next gate.
