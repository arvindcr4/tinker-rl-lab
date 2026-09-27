"""Probe: can a Modal CPU sandbox on the cached E1 native image (im-pAVtHHKH6EXHe7tlbHI7kb) run dockerd?"""
import sys
import time

import modal

app = modal.App.lookup("pavlov-e1-finish-native-0927", create_if_missing=True)
image = modal.Image.from_id("im-pAVtHHKH6EXHe7tlbHI7kb")
sb = modal.Sandbox.create(image=image, app=app, cpu=4, memory=16 * 1024, timeout=900,
                          experimental_options={"enable_docker": True})
print("sandbox", sb.object_id, flush=True)
try:
    def run(cmd, t=300):
        p = sb.exec("bash", "-lc", cmd, timeout=t)
        p.wait()
        print(f"$ {cmd}\nrc={p.returncode}\n{p.stdout.read()[-3000:]}\n{p.stderr.read()[-2000:]}", flush=True)
        return p.returncode
    run("ls /native | head; /native/.venv/bin/swebench --help | head -3; which dockerd docker")
    run("ls /usr/sbin/iptables*; update-alternatives --set iptables /usr/sbin/iptables-legacy; update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy; (dockerd >/tmp/dockerd.log 2>&1 &) ; for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 1; done; docker info | head -20; tail -5 /tmp/dockerd.log")
    run("docker pull hello-world && docker run --rm hello-world | head -3", 300)
finally:
    sb.terminate()
