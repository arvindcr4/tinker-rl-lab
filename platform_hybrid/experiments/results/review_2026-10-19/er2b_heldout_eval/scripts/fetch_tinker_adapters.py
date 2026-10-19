"""Read-only: fetch the four E-R2b Tinker sampler checkpoints (final) via signed archive URL
(kept in memory only, never printed/written), extract, record file list + sha256."""
import hashlib, io, json, os, sys, tarfile, urllib.request
import tinker

OUT = sys.argv[1]
PATHS = {
    "er2b_g2_s123": "tinker://8ce8cdc4-9d71-54a8-b05d-5905d50d1df0:train:0/sampler_weights/final",
    "er2b_g2_s456": "tinker://105c11dd-8b93-5f92-afed-143640e612dd:train:0/sampler_weights/final",
    "er2b_g16_s123": "tinker://2af35713-c50d-5215-b1b5-68e5ebcec0db:train:0/sampler_weights/final",
    "er2b_g16_s456": "tinker://681b66a6-d58c-5e8f-aa5c-f8f060fa2252:train:0/sampler_weights/final",
}
rc = tinker.ServiceClient().create_rest_client()
meta = {}
for name, tp in PATHS.items():
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    r = rc.get_checkpoint_archive_url_from_tinker_path(tp).result()
    url = getattr(r, "url", None) or r
    data = urllib.request.urlopen(url, timeout=900).read()
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            base = os.path.basename(m.name)
            if m.isfile() and base and ".." not in m.name:
                m.name = base
                tf.extract(m, d)
    meta[name] = {"tinker_path": tp, "archive_sha256": hashlib.sha256(data).hexdigest(),
                  "archive_bytes": len(data), "files": sorted(os.listdir(d))}
    print(name, meta[name]["files"], len(data), flush=True)
json.dump(meta, open(os.path.join(OUT, "tinker_adapters_meta.json"), "w"), indent=1)
