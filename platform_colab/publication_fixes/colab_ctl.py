"""Local helpers for trusted operator commands on a Colab session.

Structured launch/status/tail input is data, never shell or Python source.
The explicit sh operation intentionally executes a trusted operator shell command.
"""
import subprocess
import sys

COLAB = "/Users/arvind/.local/bin/colab"
MAX_CONCURRENCY = 16


def exec_py(session, code, timeout=600):
    if type(timeout) is not int or not 1 <= timeout <= 14400:
        raise ValueError("timeout must be an integer between 1 and 14400 seconds")
    r = subprocess.run([COLAB, "exec", "-s", session, "--timeout", str(timeout)], input=code,
                       capture_output=True, text=True, timeout=timeout + 120, check=True)
    return r.stdout + r.stderr


def sh(session, cmd, timeout=600):
    code = (f"import subprocess\nr=subprocess.run({cmd!r},shell=True,capture_output=True,"
            f"text=True,errors='ignore',timeout={timeout!r})\n"
            "print(r.stdout[-20000:]);print(r.stderr[-5000:])\n"
            "raise SystemExit(r.returncode)\n")
    return exec_py(session, code, timeout)


def launch_code(jobs, concurrency):
    n = int(concurrency)
    if not 1 <= n <= MAX_CONCURRENCY:
        raise ValueError("Concurrency must be between 1 and 16")
    return ("import subprocess,os\n"
            f"p=subprocess.Popen(['python','/content/pf/vm_queue.py',{jobs!r},{str(n)!r}],"
            f"stdout=open({jobs!r}+'.queue.log','w'),stderr=subprocess.STDOUT,start_new_session=True)\n"
            "print('pid',p.pid)\n")


def status_code(logdir):
    return ("from pathlib import Path\nimport subprocess\n"
            f"root=Path({logdir!r})\n"
            "print((root/'status.json').read_text()[-20000:])\n"
            "r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',"
            "'--format=csv,noheader'],capture_output=True,text=True,timeout=30)\n"
            "print(r.stdout[-20000:])\n"
            "for path in sorted(root.glob('*.log'))[:100]:\n"
            "    print('==',path.name)\n"
            "    with path.open('rb') as handle:\n"
            "        handle.seek(0,2)\n"
            "        handle.seek(max(0,handle.tell()-2000))\n"
            "        print(handle.read(2000).decode('utf-8','replace'))\n")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    op, session = argv[:2]
    if op == "push":
        result = subprocess.run([COLAB, "upload", "-s", session, argv[2], argv[3]],
                                capture_output=True, text=True, check=True, timeout=180)
        print(result.stdout)
    elif op == "launch":
        print(exec_py(session, launch_code(argv[2], argv[3])))
    elif op == "status":
        print(exec_py(session, status_code(argv[2]), 120))
    elif op == "tail":
        n = int(argv[3]) if len(argv) > 3 else 2000
        if not 1 <= n <= 20000:
            raise ValueError("tail size must be between 1 and 20000 bytes")
        code = (f"with open({argv[2]!r},'rb') as f:\n"
                f"    f.seek(0,2)\n    f.seek(max(0,f.tell()-{n}))\n"
                f"    print(f.read({n}).decode('utf-8','replace'))\n")
        print(exec_py(session, code, 120))
    elif op == "sh":
        print(sh(session, argv[2], int(argv[3]) if len(argv) > 3 else 600))
    else:
        raise ValueError("Unknown operation")


if __name__ == "__main__":
    main()
