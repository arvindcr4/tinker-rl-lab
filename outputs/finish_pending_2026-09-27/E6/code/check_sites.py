"""E6: HTTP status of every WebArena site (localhost on the VM)."""
import urllib.request
U = {"shopping": "http://localhost:7770", "shopping_admin": "http://localhost:7780/admin",
     "reddit": "http://localhost:9999", "gitlab": "http://localhost:8023/explore",
     "wikipedia": "http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
     "map": "http://localhost:3000", "map_tile": "http://localhost:3000/tile/0/0/0.png",
     "homepage": "http://localhost:4399"}
for k, u in U.items():
    try:
        r = urllib.request.urlopen(u, timeout=30)
        print(k, r.status, len(r.read()))
    except Exception as e:
        print(k, "ERR", str(e)[:120])
