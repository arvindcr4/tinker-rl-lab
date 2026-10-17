#!/usr/bin/env bash
# Rewrite baked-in CMU reference URLs to point at local docker ports.
# WebArena images hard-code http://metis.lti.cs.cmu.edu:<port> in their
# database/config so Location: 302s from the reference deployment follow
# back to CMU and die when CMU is unreachable from GCP.
# Run AFTER `docker compose up -d` and a stabilization sleep.
set -euo pipefail
: "${WA_REQUIRED_SERVICES:?Provide the manifest-derived required services}"
for site in $WA_REQUIRED_SERVICES; do
  case "$site" in
    shopping) container=webarena-shopping ;;
    shopping_admin) container=webarena-shopping-admin ;;
    gitlab) container=webarena-gitlab ;;
    homepage) container=webarena-homepage ;;
    *) echo "Unsupported required service; refusing evaluation" >&2; exit 2 ;;
  esac
  sudo docker ps --format '{{.Names}}' | grep -Fxq "$container" || { echo "Required service container missing" >&2; exit 1; }
done

echo "==> Resetting WebArena base URLs to localhost..."

# --- Shopping (Magento OneStopShop, port 7770) ---
if [[ " $WA_REQUIRED_SERVICES " == *" shopping "* ]]; then
  echo "  shopping (7770)..."
  sudo docker exec webarena-shopping /var/www/magento2/bin/magento \
    setup:store-config:set --base-url="http://localhost:7770"
  sudo docker exec webarena-shopping /var/www/magento2/bin/magento \
    setup:store-config:set --base-url-secure="http://localhost:7770"
  sudo docker exec webarena-shopping mysql -u magentouser -pMyPassword magentodb \
    -e "UPDATE core_config_data SET value='http://localhost:7770/' WHERE path LIKE 'web/%/base_url';"
  sudo docker exec webarena-shopping /var/www/magento2/bin/magento cache:flush
fi

# --- Shopping Admin (Magento CMS Admin, port 7780) ---
if [[ " $WA_REQUIRED_SERVICES " == *" shopping_admin "* ]]; then
  echo "  shopping_admin (7780)..."
  sudo docker exec webarena-shopping-admin /var/www/magento2/bin/magento \
    setup:store-config:set --base-url="http://localhost:7780"
  sudo docker exec webarena-shopping-admin /var/www/magento2/bin/magento \
    setup:store-config:set --base-url-secure="http://localhost:7780"
  sudo docker exec webarena-shopping-admin mysql -u magentouser -pMyPassword magentodb \
    -e "UPDATE core_config_data SET value='http://localhost:7780/' WHERE path LIKE 'web/%/base_url';"
  sudo docker exec webarena-shopping-admin /var/www/magento2/bin/magento cache:flush
fi

# --- GitLab (port 8023) ---
if [[ " $WA_REQUIRED_SERVICES " == *" gitlab "* ]]; then
  echo "  gitlab (8023)..."
  sudo docker exec webarena-gitlab sed -i \
    "s|^external_url.*|external_url 'http://localhost:8023'|" /etc/gitlab/gitlab.rb
  sudo docker exec webarena-gitlab gitlab-ctl reconfigure >/dev/null 2>&1
fi

# Forum/maps/Wikipedia tasks are rejected before allocation by webarena_contract.py.
for site in $WA_REQUIRED_SERVICES; do
  case "$site" in
    shopping) url=http://localhost:7770 ;;
    shopping_admin) url=http://localhost:7780 ;;
    gitlab) url=http://localhost:8023 ;;
    homepage) url=http://localhost:4399 ;;
    *) echo "Unsupported required service; refusing evaluation" >&2; exit 2 ;;
  esac
  ready=false
  for attempt in $(seq 1 60); do
    if effective=$(curl -fsSL --max-time 10 -o /dev/null -w '%{url_effective}' "$url" 2>/dev/null); then
      if [[ "$effective" == "$url" || "$effective" == "$url/"* ]]; then ready=true; break; fi
    fi
    sleep 5
  done
  $ready || { echo "Required service is not ready or redirects outside its local origin" >&2; exit 1; }
done

echo "==> Required URL resets and readiness checks complete."
