"""Read-only: fetch matched-run Tinker sampler checkpoints (m42/m123/m456_final) via signed archive URL
(kept in memory only, never printed/written), extract, record file list + sha256."""
import hashlib, io, json, os, sys, tarfile, urllib.request
import tinker

OUT = sys.argv[1]
PATHS = {
    "tinker_s42": "tinker://932c7b50-f12c-5d1c-b3b9-9192739cb079:train:0/sampler_weights/m42_final",
    "tinker_s123": "tinker://72ce705d-c863-543f-9e47-7ca111ba7978:train:0/sampler_weights/m123_final",
    "tinker_s456": "tinker://47b2d7cd-36af-5b20-a719-170ccb8252c6:train:0/sampler_weights/m456_final",
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
