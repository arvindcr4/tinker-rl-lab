#!/bin/bash
# E6: map + wikipedia data via webarena-verified CLI (original archive.org/kiwix sources are dark/404).
set -x
exec >> /root/setup_mapwiki.log 2>&1
while ! systemctl is-active docker; do sleep 10; done
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
mkdir -p /data/wa_dl
cd /data
uvx webarena-verified --help
uvx webarena-verified env setup init --site map --data-dir /data/wa_dl/map && echo MAP_INIT_DONE &
uvx webarena-verified env setup init --site wikipedia --data-dir /data/wa_dl/wiki && echo WIKI_INIT_DONE &
wait
echo SETUP_MAPWIKI_DONE
