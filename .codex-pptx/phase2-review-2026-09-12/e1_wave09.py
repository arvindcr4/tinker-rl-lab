import importlib.util,sys,hashlib
from pathlib import Path
ROOT=Path('/Users/arvind/Developer/tinker-rl-lab')
p=ROOT/'.codex-run/public_native_improve_v2/e1/metadata_wave08.py'
s=importlib.util.spec_from_file_location('wave',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.PRIOR=ROOT/'outputs/public_portfolio_2026-09-05/swe_multilingual_setup/registry_preflight08/registry_receipt.json'
m.PRIOR_SHA=hashlib.sha256(m.PRIOR.read_bytes()).hexdigest()
m.OUTPUT=ROOT/'outputs/PES_Phase2_Review_2026-09-12/e1_registry_wave09'
# Invoke the unchanged child in a fresh process under its frozen 120-second network deadline.
m.child()
