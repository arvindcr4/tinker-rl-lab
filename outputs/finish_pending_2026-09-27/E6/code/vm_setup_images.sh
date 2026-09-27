#!/bin/bash
# E6: install docker, pull official WebArena image tars from pre-staged GCS bucket, docker load.
set -x
exec >> /root/setup_images.log 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y docker.io python3-pip python3-venv git jq aria2 pigz
systemctl enable --now docker
mkdir -p /data/tars && cd /data/tars
gcloud storage cp 'gs://arvindcr-webarena-images-20260422/*.tar' /data/tars/ && echo GCS_COPY_DONE
gcloud storage hash /data/tars/*.tar > /data/tars/local_hashes.txt 2>&1
for t in shopping_admin_final_0719 shopping_final_0712 postmill-populated-exposed-withimg gitlab-populated-final-port8023; do
  docker load -i /data/tars/$t.tar && echo LOADED $t
done
docker images
echo SETUP_IMAGES_DONE
