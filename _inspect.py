import json, glob, os
files = sorted(glob.glob('registry/entries/delta_*.json'))
print(f'Total delta entries: {len(files)}')
for f in files[:5]:
    d = json.load(open(f))
    rid = d.get('id', '?')
    meas = d.get('measured', [])
    print(f'\n=== {os.path.basename(f)} ({rid}) ===')
    print(f'  measured count: {len(meas) if isinstance(meas, list) else meas}')
    for m in (meas or [])[:2]:
        src = m.get('source', '')
        print(f'    metric={m.get("metric")}, panel={m.get("panel")}, delta={m.get("delta")}, ci=[{m.get("ci_low")}, {m.get("ci_high")}], sig={m.get("significant")}, source={src[-60:]!r}')