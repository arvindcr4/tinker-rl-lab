#!/bin/bash
# E6: native WebArena harness (upstream web-arena-x/webarena @ dce04686) on python3.10 + playwright 1.32.1.
set -x
exec >> /root/setup_harness.log 2>&1
export DEBIAN_FRONTEND=noninteractive
while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 5; done
apt-get install -y python3.10-venv git
mkdir -p /opt/e6 && cd /opt/e6
[ -d webarena ] || git clone https://github.com/web-arena-x/webarena.git
cd webarena && git checkout dce04686a56253aefba7b18a4fa0937cf1dc987b && git rev-parse HEAD
python3.10 -m venv /opt/e6/venv
. /opt/e6/venv/bin/activate
pip install -U pip wheel "setuptools<70"
pip install -r requirements.txt
pip install -e .
pip install requests
playwright install-deps chromium
playwright install chromium
python -c "import nltk; nltk.download('punkt')"
sha256sum config_files/test.raw.json
echo SETUP_HARNESS_DONE
