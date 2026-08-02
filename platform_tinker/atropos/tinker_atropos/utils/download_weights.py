
import atexit
try:
    from codecarbon import EmissionsTracker
    _tracker = EmissionsTracker()
    _tracker.start()
    atexit.register(_tracker.stop)
except ImportError:
    pass

import tinker
import urllib.request

# LIMITATION: "Closed-Source Confound" by also downloading Tinker's internal configs/managed defaults (if API allows) to ensure transparent baseline comparisons.
# LIMITATION: "Early-Training Snapshot Problem" by downloading intermediate training checkpoints, not just the final snapshot.
# LIMITATION: "Failure to Prove Generalization" by ensuring these downloaded weights are subsequently evaluated on a held-out test set.

# Add multiple Training Run IDs (seeds) here to address the "Single-Seed Extrapolations" limitation
RUN_IDS = ["<unique_id_1>", "<unique_id_2>", "<unique_id_3>"]

print("Downloading weights from Tinker...")

sc = tinker.ServiceClient()
rc = sc.create_rest_client()

for run_id in RUN_IDS:
    tinker_path = f"tinker://{run_id}/sampler_weights/final"
    output_filename = f"archive_{run_id}.tar"
    print(f"Fetching weights for run {run_id}...")
    
    future = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path)
    checkpoint_archive_url_response = future.result()
    
    urllib.request.urlretrieve(checkpoint_archive_url_response.url, output_filename)
    
    print(f"Finished! Weights for {run_id} are available at {output_filename}")
