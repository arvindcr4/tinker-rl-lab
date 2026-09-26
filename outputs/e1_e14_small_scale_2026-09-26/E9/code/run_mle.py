"""E9 small-scale: few-step MLE-bench agent. Base actor (via local Tinker shim) writes a solution script;
the script runs on a Colab CPU VM (session e9-mle); submission.csv is downloaded and graded locally with the official mlebench grader.
Usage: run_mle.py <data_dir> <raw_out_dir> <comp_id> [<comp_id> ...]"""
import json, os, re, subprocess, sys, tarfile, time, urllib.request

SHIM = os.environ.get("SHIM_URL", "http://127.0.0.1:18770/chat")
SESSION = os.environ.get("COLAB_SESSION", "e9-mle")
MAX_ATTEMPTS, MAX_TOKENS, RUN_TIMEOUT = 3, 4096, 1200
DATA, OUT, COMPS = sys.argv[1], sys.argv[2], sys.argv[3:]

SYSTEM = ("You are an expert Kaggle competitor. Write ONE complete, self-contained Python 3 script that trains a model on the "
          "competition data in the current working directory and writes predictions to ./submission.csv in exactly the format of "
          "the sample submission. Constraints: CPU only, no internet, must finish in under 15 minutes. Available libraries: numpy, "
          "pandas, scipy, scikit-learn, lightgbm, xgboost. Reply with a single ```python code block and nothing else.")


def colab(*args, timeout=None):
    return subprocess.run(["colab", *args], capture_output=True, text=True, timeout=timeout)


def ask(messages, tag):
    req = urllib.request.Request(SHIM, data=json.dumps({"messages": messages, "max_tokens": MAX_TOKENS, "tag": tag}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def context(pub):
    parts = [open(os.path.join(pub, "description.md")).read()[:7000], "\n## Files in working directory"]
    for root, dirs, files in os.walk(pub):
        rel = os.path.relpath(root, pub)
        if rel != "." and len(files) > 5:
            parts.append(f"{rel}/ ({len(files)} files, e.g. {sorted(files)[:3]})")
            dirs[:] = []
            continue
        for f in sorted(files):
            if f == "description.md":
                continue
            p = os.path.join(root, f)
            name = os.path.normpath(os.path.join(rel, f))
            parts.append(f"- {name} ({os.path.getsize(p)} bytes)")
            if f.endswith((".csv", ".json", ".txt", ".xyz")) and rel == ".":
                with open(p, errors="replace") as fh:
                    parts.append("```\n" + fh.read(1200) + "\n```")
    return "\n".join(parts)


def run_comp(comp):
    pub = os.path.join(DATA, comp, "prepared", "public")
    out = os.path.join(OUT, comp)
    os.makedirs(out, exist_ok=True)
    tgz = os.path.join(out, "_public.tar.gz")
    with tarfile.open(tgz, "w:gz") as t:
        t.add(pub, arcname=comp)
    remote = f"/content/{comp}"
    r = colab("upload", "-s", SESSION, tgz, f"/content/{comp}.tar.gz", timeout=600)
    os.remove(tgz)
    setup = os.path.join(out, "_setup.py")
    open(setup, "w").write(f"import tarfile; tarfile.open('/content/{comp}.tar.gz').extractall('/content'); print('extracted')\n")
    r2 = colab("exec", "-s", SESSION, "-f", setup, "--timeout", "300", timeout=400)
    log = {"comp": comp, "upload_rc": r.returncode, "setup_out": (r2.stdout + r2.stderr)[-500:], "attempts": []}
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": context(pub)}]
    for a in range(1, MAX_ATTEMPTS + 1):
        resp = ask(messages, f"{comp}#{a}")
        text = resp.get("text", "")
        m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
        code = m.group(1) if m else text
        sol = os.path.join(out, f"solution_attempt{a}.py")
        open(sol, "w").write(code)
        colab("upload", "-s", SESSION, sol, f"{remote}/solution.py", timeout=300)
        runner = os.path.join(out, "_runner.py")
        open(runner, "w").write(
            "import subprocess, os\n"
            f"os.chdir('{remote}')\n"
            "if os.path.exists('submission.csv'): os.remove('submission.csv')\n"
            "try:\n"
            f"    p = subprocess.run(['python', 'solution.py'], capture_output=True, text=True, timeout={RUN_TIMEOUT})\n"
            "    rc, so, se = p.returncode, p.stdout, p.stderr\n"
            "except subprocess.TimeoutExpired:\n"
            "    rc, so, se = -9, '', 'TIMEOUT'\n"
            "print('RC=', rc, 'HAS_SUB=', os.path.exists('submission.csv'))\n"
            "print('STDOUT_TAIL', so[-1500:])\n"
            "print('STDERR_TAIL', se[-3000:])\n")
        t0 = time.time()
        rr = colab("exec", "-s", SESSION, "-f", runner, "--timeout", str(RUN_TIMEOUT + 120), timeout=RUN_TIMEOUT + 300)
        exec_out = rr.stdout + rr.stderr
        ok = "HAS_SUB= True" in exec_out and "RC= 0" in exec_out
        att = {"attempt": a, "prompt_tokens": resp.get("prompt_tokens"), "sample_tokens": resp.get("sample_tokens"),
               "shim_error": resp.get("error"), "exec_seconds": round(time.time() - t0, 1), "ok": ok, "exec_tail": exec_out[-4000:]}
        log["attempts"].append(att)
        print(comp, a, ok, flush=True)
        if ok:
            d = colab("download", "-s", SESSION, f"{remote}/submission.csv", os.path.join(out, "submission.csv"), timeout=300)
            log["download_rc"] = d.returncode
            break
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "The script failed or produced no submission.csv. Output tail:\n" + exec_out[-3500:] +
                      "\nFix the problem and reply with the full corrected script in a single ```python block."}]
    json.dump(log, open(os.path.join(out, "agent_log.json"), "w"), indent=1)


if __name__ == "__main__":
    for c in COMPS:
        try:
            run_comp(c)
        except Exception as e:
            print(c, "ERROR", repr(e), flush=True)
            json.dump({"comp": c, "error": repr(e)}, open(os.path.join(OUT, c, "agent_error.json"), "w"))
