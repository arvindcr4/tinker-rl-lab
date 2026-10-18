#!/bin/bash
# Remote driver (Colab VM). Merge chain (CPU, base env) runs in parallel with the vLLM gen chain (GPU, /content/vv).
cd /content
BASEPY=$(command -v python3)
export PATH=/content/vv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
OUT=/content/out
mkdir -p $OUT /content/merged
BASE_SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
CONDS_BASE=${CONDS_BASE:-wc_think_512,wc_nothink_4096,matched_think_512,matched_nothink_4096,wc_think_4096,matched_think_4096}
CONDS_Q1=${CONDS_Q1:-wc_think_512,wc_nothink_4096,wc_think_4096}
CONDS_Q2=${CONDS_Q2:-matched_think_512,matched_nothink_4096,matched_think_4096}

merge_chain() {
  for n in q1_scale8b tinker_s42 tinker_s123 tinker_s456; do
    ad=/content/ad/$n; [ $n = q1_scale8b ] && ad=/content/ad/q1_scale8b/final
    until [ -f $ad/adapter_model.safetensors ] && [ ! -e /content/ad/$n/part_aa ]; do sleep 20; done
    echo "[merge] $n start $(date -u +%T)"
    $BASEPY gsm8k_eval.py merge $n $ad > $OUT/merge_$n.log 2>&1 && touch /content/merged/$n/.done
    rc=$?; echo "[merge] $n end $(date -u +%T) rc=$rc"
  done
}
merge_chain >> $OUT/driver.log 2>&1 &

echo "[gen] base start $(date -u +%T)" >> $OUT/driver.log
/content/vv/bin/python gsm8k_eval.py gen $BASE_SNAP base $OUT $CONDS_BASE > $OUT/gen_base.log 2>&1
rc=$?; echo "[gen] base end $(date -u +%T) rc=$rc" >> $OUT/driver.log
for n in q1_scale8b tinker_s42 tinker_s123 tinker_s456; do
  c=$CONDS_Q2; [ $n = q1_scale8b ] && c=$CONDS_Q1
  until [ -f /content/merged/$n/.done ]; do sleep 20; done
  echo "[gen] $n start $(date -u +%T)" >> $OUT/driver.log
  /content/vv/bin/python gsm8k_eval.py gen /content/merged/$n $n $OUT $c > $OUT/gen_$n.log 2>&1
  rc=$?; echo "[gen] $n end $(date -u +%T) rc=$rc" >> $OUT/driver.log
  rm -rf /content/merged/$n/*.safetensors
done
echo ALLDONE >> $OUT/driver.log
