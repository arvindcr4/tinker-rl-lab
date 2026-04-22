#!/usr/bin/env bash
# Rewrite baked-in CMU reference URLs to point at local docker ports.
# WebArena images hard-code http://metis.lti.cs.cmu.edu:<port> in their
# database/config so Location: 302s from the reference deployment follow
# back to CMU and die when CMU is unreachable from GCP.
# Run AFTER `docker compose up -d` and a stabilization sleep.
set -euo pipefail

echo "==> Resetting WebArena base URLs to localhost..."

# --- Shopping (Magento OneStopShop, port 7770) ---
if sudo docker ps --format '{{.Names}}' | grep -q '^webarena-shopping$'; then
  echo "  shopping (7770)..."
  sudo docker exec webarena-shopping /var/www/magento2/bin/magento \
    setup:store-config:set --base-url="http://localhost:7770" || true
  sudo docker exec webarena-shopping /var/www/magento2/bin/magento \
    setup:store-config:set --base-url-secure="http://localhost:7770" || true
  sudo docker exec webarena-shopping mysql -u magentouser -pMyPassword magentodb \
    -e "UPDATE core_config_data SET value='http://localhost:7770/' WHERE path LIKE 'web/%/base_url';" || true
  sudo docker exec webarena-shopping /var/www/magento2/bin/magento cache:flush || true
fi

# --- Shopping Admin (Magento CMS Admin, port 7780) ---
if sudo docker ps --format '{{.Names}}' | grep -q '^webarena-shopping-admin$'; then
  echo "  shopping_admin (7780)..."
  sudo docker exec webarena-shopping-admin /var/www/magento2/bin/magento \
    setup:store-config:set --base-url="http://localhost:7780" || true
  sudo docker exec webarena-shopping-admin /var/www/magento2/bin/magento \
    setup:store-config:set --base-url-secure="http://localhost:7780" || true
  sudo docker exec webarena-shopping-admin mysql -u magentouser -pMyPassword magentodb \
    -e "UPDATE core_config_data SET value='http://localhost:7780/' WHERE path LIKE 'web/%/base_url';" || true
  sudo docker exec webarena-shopping-admin /var/www/magento2/bin/magento cache:flush || true
fi

# --- GitLab (port 8023) ---
if sudo docker ps --format '{{.Names}}' | grep -q '^webarena-gitlab$'; then
  echo "  gitlab (8023)..."
  sudo docker exec webarena-gitlab sed -i \
    "s|^external_url.*|external_url 'http://localhost:8023'|" /etc/gitlab/gitlab.rb || true
  sudo docker exec webarena-gitlab gitlab-ctl reconfigure >/dev/null 2>&1 || true
fi

# --- Forum (postmill, port 9999) ---
# Postmill uses APP_URL env at startup; URLs in the populated DB may still
# point at metis. The DB is typically too large for a bulk rewrite in-place,
# so we set a per-request Host header in BrowserGym's tasks. If tasks still
# fail with cross-domain redirects, we would need to re-dump postmill with
# localhost URLs. For now, leave forum untouched and accept its failure
# rate in the final aggregation.
if sudo docker ps --format '{{.Names}}' | grep -q '^webarena-forum$'; then
  echo "  forum (9999) — skipping (DB-embedded URLs; see comment)"
fi

echo "==> URL reset complete."
