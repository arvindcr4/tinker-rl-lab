"""P9 Deep Review: print arXiv title/authors (and abstract with a 2nd arg) for comma-separated IDs."""
import sys, urllib.request, xml.etree.ElementTree as ET
ids = sys.argv[1]
url = "http://export.arxiv.org/api/query?max_results=40&id_list=" + ids
data = urllib.request.urlopen(url, timeout=60).read()
ns = {'a': 'http://www.w3.org/2005/Atom'}
r = ET.fromstring(data)
for e in r.findall('a:entry', ns):
    i = e.find('a:id', ns).text.split('/')[-1]
    t = ' '.join(e.find('a:title', ns).text.split())
    au = [a.find('a:name', ns).text for a in e.findall('a:author', ns)]
    print(i, '|', t, '|', ', '.join(au[:10])); print(' '.join(e.find('a:summary', ns).text.split())) if len(sys.argv)>2 else None
