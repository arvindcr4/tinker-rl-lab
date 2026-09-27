#!/bin/bash
# E6 (resume 2026-09-27 ~08:55Z): w2_shopping_admin ran serially at ~4.5 min/task (105 left, ~8 h).
# Shard the remaining w2 tasks over 3 workers, each with its OWN freshly-started shopping_admin
# container (7780 = original, 7781/7782 = fresh replicas of the same official image, same base-url setup
# as start_sites.sh). Replica workers run from a copy of the webarena repo whose config_files have
# localhost:7780 -> localhost:778x and SHOPPING_ADMIN overridden. Results go to w2_shopping_admin_s{0,1,2}.
set -x
H=localhost
for P in 7781 7782; do
  N=shopping_admin_$P
  sudo docker rm -f $N 2>/dev/null
  sudo docker run --name $N -p $P:80 -d shopping_admin_final_0719
done
sleep 90
for P in 7781 7782; do
  N=shopping_admin_$P
  sudo docker exec $N php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
  sudo docker exec $N php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
  sudo docker exec $N /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$H:$P"
  sudo docker exec $N mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value=\"http://$H:$P/\" WHERE path = \"web/secure/base_url\";"
  sudo docker exec $N /var/www/magento2/bin/magento cache:flush
  rm -rf /opt/e6/wa_$P; cp -r /opt/e6/webarena /opt/e6/wa_$P
  sed -i "s#$H:7780#$H:$P#g" /opt/e6/wa_$P/config_files/*.json
  echo "$P $(curl -s -o /dev/null -m 60 -w '%{http_code}' http://$H:$P/admin)"
done
# remaining ids (split order preserved), round-robin into 3 shards
python3 - <<'E'
import json
ids = open("/opt/e6/run/splits/w2_shopping_admin.txt").read().split()
done = {str(json.loads(l)["task_id"]) for l in open("/opt/e6/results/w2_shopping_admin/results.jsonl") if l.strip()}
rem = [i for i in ids if i not in done]
for s in range(3):
    open(f"/opt/e6/run/splits/w2_s{s}.txt", "w").write("\n".join(rem[s::3]) + "\n")
print("remaining", len(rem))
E
pkill -f "e6_driver.py --task_ids @/opt/e6/run/splits/w2_shopping_admin.txt"
sleep 3
for s in 0 1 2; do
  P=$((7780 + s)); D=/opt/e6/webarena; [ $s -gt 0 ] && D=/opt/e6/wa_$P
  setsid nohup bash -c "cd $D && . /opt/e6/venv/bin/activate && . /opt/e6/env && export SHOPPING_ADMIN=http://$H:$P/admin && python /opt/e6/run/e6_driver.py --task_ids @/opt/e6/run/splits/w2_s$s.txt --result_dir /opt/e6/results/w2_shopping_admin_s$s && echo SHARD_DONE > /opt/e6/results/w2_shopping_admin_s$s/DONE" > /opt/e6/results_w2_s$s.log 2>&1 < /dev/null &
done
sleep 5; pgrep -af e6_driver | cut -c1-160
echo W2_SHARD_LAUNCHED
