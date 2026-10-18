"""P9 Deep Review: print Crossref title/authors for DOIs."""
import json, sys, urllib.request
for doi in sys.argv[1:]:
    d = json.load(urllib.request.urlopen("https://api.crossref.org/works/" + doi, timeout=60))["message"]
    print(doi, "|", d.get("title"), "|", [(a.get("given"), a.get("family")) for a in d.get("author", [])], d.get("issued"))
