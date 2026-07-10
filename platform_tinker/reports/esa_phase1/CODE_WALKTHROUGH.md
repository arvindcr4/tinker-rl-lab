# Phase-1 defense demo runbook

Audience: M.Tech Phase-1 examiners. Goal: demonstrate the ZVF mechanism and verify
one committed result artifact without depending on Tinker, W&B, a GPU, or network
access.

The authoritative demo is `submission/demo/`. The older live-Tinker and Hugging
Face Space instructions are retired because they depended on stale checkout paths,
missing secrets, and inherited Semester-3 content.

## Pre-viva check

From the repository root:

```bash
./submission/demo/demo.sh
python3 -m unittest discover -s submission/demo/tests -v
```

Expected headline output:

```text
Mechanism fixture: PASS (4 groups, ZVF=0.500, GU=0.500)
Recorded artifact: PASS (80 rewards, mean=0.6875, ZVF=0.3000)
DEMO STATUS: PASS
```

The recorded reward mean is an arithmetic check of a committed artifact. It is not
presented as held-out accuracy or new scientific evidence.

## 90-second defense sequence

1. Run `./submission/demo/demo.sh`.
2. Explain that two of four synthetic groups have zero reward variance, so
   `ZVF=0.5` and reward-relative gradient utilisation is `0.5`; KL-only updates are
   outside this metric.
3. Point out the SHA-256 verification before the committed JSON is analysed.
4. Open `submission/demo/output/demo_report.html` if the panel wants a visual view.
5. Close with the boundary: the demo proves the mechanism and artifact arithmetic;
   the thesis claims retain their stated seed, schedule, and CV limitations.

For a longer presenter script and Q&A boundaries, use
`submission/demo/DEFENSE_RUNBOOK.md`.

## Optional local visual mode

```bash
./submission/demo/demo.sh --serve --port 8765
```

Open `http://127.0.0.1:8765/demo_report.html`. This remains offline and serves only generated local
files. The optional Groq/Kimi schema smoke in `submission/demo/README.md` is not part
of the primary defense path.
