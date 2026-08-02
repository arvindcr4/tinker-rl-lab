# Colab, Hugging Face, and W&B hardening

The authorized A100 preflight completed remotely, but the original Colab wrapper
lost the child process result marker after treating a successful `SystemExit(0)`
as an IPython exception. The retained implementation streams child output into
the parent kernel and lets a successful wrapper cell return normally.

If the transport marker is still absent, recovery is fail-closed. It derives the
unique repository from the frozen request fingerprint, requires a private
Hugging Face repository containing both the run manifest and final adapter,
downloads the manifest at the resolved commit, re-runs the full semantic
validator, and requires the referenced W&B run to be finished.

The repaired local receipt validates private Hugging Face commit
`3cfeca29d02cbda308376f83de4b5911ac865607` and finished W&B run `5573d4a7`.
Colab cleanup was verified both in the receipt and by a fresh server-side session
enumeration.

The scientific boundary is unchanged: this is `preflight-not-evidence`. The two
sampled groups were homogeneous and produced zero parameter updates, so the run
does not validate the mixed-reward gradient path or support any conference
performance claim.
