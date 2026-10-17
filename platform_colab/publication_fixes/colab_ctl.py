"""Local helpers to drive long jobs on a Colab session via the colab CLI.

  python colab_ctl.py push SESSION LOCAL REMOTE          upload a file
  python colab_ctl.py launch SESSION REMOTE_JOBS N        start vm_queue.py detached
  python colab_ctl.py status SESSION REMOTE_LOGDIR        print status.json + gpu
  python colab_ctl.py sh SESSION 'shell cmd' [timeout]    run a shell command
"""
import subprocess, sys

COLAB = "/Users/arvind/.local/bin/colab"


def exec_py(session, code, timeout=600):
    r = subprocess.run([COLAB, "exec", "-s", session, "--timeout", str(timeout)], input=code,
                       capture_output=True, text=True, timeout=timeout + 120)
    return r.stdout + r.stderr


def sh(session, cmd, timeout=600):
    code = f"import subprocess\nr=subprocess.run({cmd!r},shell=True,capture_output=True,text=True,errors='ignore')\nprint(r.stdout[-20000:]);print(r.stderr[-5000:])\n"
    return exec_py(session, code, timeout)


if __name__ == "__main__":
    op, s = sys.argv[1], sys.argv[2]
    if op == "push":
        print(subprocess.run([COLAB, "upload", "-s", s, sys.argv[3], sys.argv[4]], capture_output=True, text=True))
    elif op == "launch":
        jobs, n = sys.argv[3], sys.argv[4]
        print(exec_py(s, f"import subprocess,os\np=subprocess.Popen(['python','/content/pf/vm_queue.py',{jobs!r},'{n}'],"
                         f"stdout=open({jobs!r}+'.queue.log','w'),stderr=subprocess.STDOUT,start_new_session=True)\nprint('pid',p.pid)\n"))
    elif op == "status":
        print(sh(s, f"cat {sys.argv[3]}/status.json; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; "
                    f"for f in {sys.argv[3]}/*.log; do echo == $f; tail -n 2 $f; done", 120))
    elif op == "tail":  # tail SESSION FILE [NBYTES]
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
        print(exec_py(s, f"import os\nb=open({sys.argv[3]!r},'rb').read()[-{n}:]\nprint(b.decode('utf-8','ignore'))\n", 120))
    elif op == "sh":
        print(sh(s, sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 600))
