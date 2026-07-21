# S1 objective differential harness

`reference.py` is the framework-neutral Trace contract. `trl_adapter.py` and
`verl_adapter.py` evaluate frozen fixtures against exact external framework
distributions on CPU without loading a model, generating completions, or
running an optimizer. Both adapters record package versions, source paths, and
source hashes; the TRL adapter additionally verifies the locked wheel hash.

Run the focused suite without changing the project environment:

```bash
PYTHONPATH=zvf-program uv run --no-sync \
  --with trl==1.2.0 \
  --with transformers==5.5.4 \
  python -m unittest discover \
  -s zvf-program/flagship/s1 -t zvf-program -p 'test_*.py' -v
```

Emit a machine-readable differential for one fixture/arm:

```bash
PYTHONPATH=zvf-program uv run --no-sync \
  --with trl==1.2.0 \
  --with transformers==5.5.4 \
  python -m flagship.s1.trl_adapter --fixture base --arm grpo
```

Run the external verl differential from outside the repository so Python
cannot resolve the repository-local `verl/` wrapper:

```bash
uv venv --python 3.11 /tmp/tinker-s1-verl-py311
uv pip install --python /tmp/tinker-s1-verl-py311/bin/python \
  verl==0.3.0.post1 torch==2.4.0 transformers==4.45.2
cd /tmp
PYTHONPATH=/absolute/path/to/tinker-rl-lab/zvf-program/flagship \
  /tmp/tinker-s1-verl-py311/bin/python -m unittest \
  s1.test_reference s1.test_verl_adapter
```

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
