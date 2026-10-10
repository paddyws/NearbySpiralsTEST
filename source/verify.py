from pathlib import Path
from html.parser import HTMLParser
import json,numpy as np
from PIL import Image
S=Path(__file__).resolve().parent/'site';rows=json.loads((S/'analysis_summary.json').read_text());assert len(rows)==35
class P(HTMLParser):
 def __init__(self):super().__init__();self.urls=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ('href','src') and v and not v.startswith(('http','mailto:','#')):self.urls.append(v.split('?')[0])
missing=[];checked=0
for p in [S/'index.html']+[S/(r['stem']+'.html') for r in rows]:
 a=P();a.feed(p.read_text(encoding='utf-8'))
 for link in a.urls:
  checked+=1
  if not (p.parent/link).exists():missing.append((p.name,link))
for r in rows:
 d=S/'images'/r['stem'];assert (d/'original.jpg').is_file()
 if r['status']=='ok':
  for m in ['track_v2','gabor_curved','fourier','sparcfire']:
   assert m in r['methods']
   for name in ['arms.png','model.png','residual.png','overlay.jpg','face.jpg']:
    p=d/(m+'_'+name);assert p.exists(),p
    with Image.open(p) as im:im.verify()
  for name in ['before.png','bulge.png','disk.png','bar.png','smooth.png','gabor_response.png']:
   with Image.open(d/name) as im:im.verify()
assert not missing,missing[:10]
report=dict(galaxies=len(rows),success=sum(r['status']=='ok' for r in rows),missing_geometry=sum(r['status']=='no_autoprof_geometry' for r in rows),checked_local_links=checked,broken_links=missing,methods={m:sum(r.get('methods',{}).get(m,{}).get('segments',0)>0 for r in rows) for m in ['track_v2','gabor_curved','fourier','sparcfire']},quality='exploratory_not_physical_validation')
(S/'validation.json').write_text(json.dumps(report,indent=2));print('VERIFIED',json.dumps(report),flush=True)
