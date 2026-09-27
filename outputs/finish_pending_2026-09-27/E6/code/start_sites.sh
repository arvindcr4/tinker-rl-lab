#!/bin/bash
# E6: (re)create WebArena site containers from the official images = the official "environment reset"
# (environment_docker/README.md: stop+remove, docker run fresh, re-apply base-url/gitlab config).
# Usage: start_sites.sh [site ...]   (default: shopping shopping_admin forum gitlab wikipedia)
set -x
H=localhost
SITES=${@:-shopping shopping_admin forum gitlab wikipedia}
for s in $SITES; do docker rm -f $s 2>/dev/null; done
for s in $SITES; do case $s in
  shopping) docker run --name shopping -p 7770:80 -d shopping_final_0712 ;;
  shopping_admin) docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719 ;;
  forum) docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg ;;
  gitlab) docker run --name gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start ;;
  wikipedia) docker run -d --name=wikipedia --volume=/data/wa_dl/wiki/:/data -p 8888:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim ;;
esac; done
sleep 60
for s in $SITES; do case $s in
  shopping)
    docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$H:7770"
    docker exec shopping mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value=\"http://$H:7770/\" WHERE path = \"web/secure/base_url\";"
    docker exec shopping /var/www/magento2/bin/magento cache:flush ;;
  shopping_admin)
    docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
    docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
    docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$H:7780"
    docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value=\"http://$H:7780/\" WHERE path = \"web/secure/base_url\";"
    docker exec shopping_admin /var/www/magento2/bin/magento cache:flush ;;
  gitlab)
    sleep 240
    docker exec gitlab update-permissions
    docker exec gitlab sed -i "s|^external_url.*|external_url 'http://$H:8023'|" /etc/gitlab/gitlab.rb
    docker exec gitlab gitlab-ctl reconfigure ;;
esac; done
for p in 7770 7780 9999 8023 8888 3000 4399; do echo "$p $(curl -s -o /dev/null -m 30 -w '%{http_code}' http://$H:$p)"; done
echo START_SITES_DONE
