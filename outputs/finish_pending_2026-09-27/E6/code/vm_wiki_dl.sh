#!/bin/bash
# E6: WebArena Wikipedia ZIM (kiwix 2022-05). Original CMU/archive webarena-env-wiki item is dark/403.
# Mirrors: HF dataset Atesting1/webarena-env + archive.org item wikipedia_en_all_maxi_2022-05 (md5 054c8bd86af847d2a8f9c832db26a038).
cd /data/wa_dl/wiki
aria2c -x 16 -s 32 --file-allocation=none --max-tries=0 --retry-wait=10 --summary-interval=60 \
  -o wikipedia_en_all_maxi_2022-05.zim \
  https://huggingface.co/datasets/Atesting1/webarena-env/resolve/main/wikipedia_en_all_maxi_2022-05.zim \
  https://archive.org/download/wikipedia_en_all_maxi_2022-05/wikipedia_en_all_maxi_2022-05.zim > /root/wiki_dl2.log 2>&1 \
  && md5sum wikipedia_en_all_maxi_2022-05.zim > /root/wiki_md5.txt && echo WIKI_DL_DONE >> /root/wiki_dl2.log
