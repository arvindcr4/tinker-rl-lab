"""Fidelity check: re-collect previously recorded source contexts and compare exactly."""
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e1_runtime as R  # noqa: E402

F = R.REPO / 'outputs/PES_Phase2_Review_2026-09-12/finish/e1_completion'
rows = {json.loads(l)['instance_id']: json.loads(l) for l in open(
    R.REPO / 'outputs/public_portfolio_2026-09-05/swe_multilingual_setup/prepared/native_dataset.jsonl')}
f = R.pro_flagship()
sb = R.open_sandbox(timeout=1800)
out = {}
try:
    print(R.sh(sb, 'df -h / /var/lib/docker | tail -2; nproc; free -g | head -2')[1])
    for iid in sys.argv[1:]:
        rec = json.load(open(glob.glob(str(F / f'**/attempts/{iid}/source_context.json'), recursive=True)[0]))
        t = time.time()
        digest, _ = R.pull(sb, rows[iid]['image'])
        tp = time.time() - t
        got = R.collect_sources(sb, rows[iid], f)
        diff = [p for p in set(got['files']) | set(rec['files']) if got['files'].get(p) != rec['files'].get(p)]
        out[iid] = {'digest': digest, 'pull_s': round(tp, 1), 'exact_match': got['files'] == rec['files'],
                    'same_file_set': set(got['files']) == set(rec['files']),
                    'differing_paths': diff[:10], 'terms': got['receipt']['search_terms']}
        print(iid, out[iid], flush=True)
        R.sh(sb, f"docker rmi -f {rows[iid]['image']} >/dev/null 2>&1")
finally:
    sb.terminate()
(R.REPO / 'outputs/finish_pending_2026-09-27/E1/validation').mkdir(parents=True, exist_ok=True)
json.dump(out, open(R.REPO / 'outputs/finish_pending_2026-09-27/E1/validation/source_fidelity.json', 'w'), indent=2)
