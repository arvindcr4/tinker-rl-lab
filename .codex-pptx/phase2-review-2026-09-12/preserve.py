"""Transfer only Artifact-authored changed text into the original13-slide package."""
from pathlib import Path
import zipfile,xml.etree.ElementTree as E,json,hashlib,re
ROOT=Path('/Users/arvind/Developer/tinker-rl-lab');HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'outputs/public_portfolio_2026-09-05/presentation/E1_E14_Public_Research_Update_v15.pptx'
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
for prefix,uri in NS.items():E.register_namespace(prefix,uri)
def text(node):return '\n'.join(''.join(x.itertext()) for x in node.findall('.//a:p',NS))
def replace(parent,old,new):
 i=list(parent).index(old);parent.remove(old);parent.insert(i,new)
content=json.loads((HERE/'content.json').read_bytes())
changed=[];allowed_slides={1,3,4,5,9,10,12,13};allowed_notes={1,3,4,5,9,10,12,13}
with zipfile.ZipFile(SOURCE) as src,zipfile.ZipFile(HERE/'artifact-draft.pptx') as draft,zipfile.ZipFile(HERE/'candidate.pptx','w',compression=zipfile.ZIP_DEFLATED) as out:
 for info in src.infolist():
  data=src.read(info.filename);match=re.fullmatch(r'ppt/(slides/slide|notesSlides/notesSlide)(\d+)\.xml',info.filename)
  if match:
   number=int(match.group(2));notes='notes' in match.group(1);allowed=number in (allowed_notes if notes else allowed_slides)
   if not allowed:
    out.writestr(info,data);continue
   original=E.fromstring(data);edited=E.fromstring(draft.read(info.filename));count=0
   assert [x.get('id') for x in original.findall('.//p:cNvPr',NS)]==[x.get('id') for x in edited.findall('.//p:cNvPr',NS)],info.filename
   for selector in (['.//p:sp'] if notes else ['.//p:sp','.//a:tc']):
    left=original.findall(selector,NS);right=edited.findall(selector,NS);assert len(left)==len(right),(info.filename,selector)
    for a,b in zip(left,right):
     tag='p:txBody' if selector=='.//p:sp' else 'a:txBody';ta=a.find(tag,NS);tb=b.find(tag,NS);assert (ta is None)==(tb is None)
     if ta is not None and text(ta)!=text(tb):
      assert allowed,('Unintended text change',info.filename,text(ta),text(tb));replace(a,ta,tb);count+=1
   if count:
    # All relationship IDs remain the original package's IDs. New text must not
    # introduce an unresolved hyperlink, embedded object or image relationship.
    rel=info.filename.rsplit('/',1);rels=rel[0]+'/_rels/'+rel[1]+'.rels'
    ids={x.attrib['Id'] for x in E.fromstring(src.read(rels))} if rels in src.namelist() else set()
    for node in original.iter():
     for key,value in node.attrib.items():
      if key.startswith('{'+NS['r']+'}'):assert value in ids,(info.filename,key,value)
    data=E.tostring(original,encoding='utf-8',xml_declaration=True);changed.append({'part':info.filename,'text_bodies':count})
  out.writestr(info,data)
with zipfile.ZipFile(SOURCE) as src,zipfile.ZipFile(HERE/'candidate.pptx') as dst:
 charts=[n for n in src.namelist() if ('/charts/' in n or '/embeddings/' in n)]
 assert len(charts)==9 and all(src.read(n)==dst.read(n) for n in charts)
 assert src.read('ppt/slides/slide2.xml')==dst.read('ppt/slides/slide2.xml')
 assert src.read('ppt/notesSlides/notesSlide2.xml')==dst.read('ppt/notesSlides/notesSlide2.xml')
 for name in src.namelist():
  if name not in [x['part'] for x in changed]:assert src.read(name)==dst.read(name),name
(HERE/'preservation.json').write_text(json.dumps({'changed_text_parts':changed,'chart_and_workbook_parts_unchanged':charts,'all_other_parts_byte_identical':True,'context_slide_and_notes_byte_identical':True,'relationships_resolved_against_original_package':True},indent=2))
print('Preserved original structure and native chart/workbook parts')
