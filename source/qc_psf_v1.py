"""Conservative per-frame star/PSF diagnostics without Gaia. Does not change scientific outputs."""
from pathlib import Path
import numpy as np,json
from astropy.io import fits
from scipy.ndimage import gaussian_filter,maximum_filter,median_filter
from PIL import Image,ImageDraw
W=Path(__file__).resolve().parent;R=Path('E:/Astro/Nearby_IMBH/JUST_THE_35_GALAXIES');S=W/'qc_psf_v1';S.mkdir(exist_ok=True)
import pandas as pd
manifest=pd.read_csv('E:/Astro/Nearby_IMBH/input_manifest.csv').set_index('stem')
rows=[]
for stem in [s for s in manifest.index if (R/(s+'.fits')).exists() and np.isfinite(manifest.loc[s,'i']) and manifest.loc[s,'i']<=50]:
 with fits.open(R/(stem+'.fits')) as hdul:
  a=np.array(hdul[0].data,dtype=float);h=hdul[0].header
 px=float(h.get('PIXSCALE',np.nan));survey=str(h.get('SURVEY','unknown'))
 med=median_filter(a,size=31);res=a-med
 sigma=1.4826*np.median(abs(res-np.median(res)))+1e-9
 peaks=(res==maximum_filter(res,size=9))&(res>8*sigma)
 yy,xx=np.nonzero(peaks)
 sources=[]
 for y,x in zip(yy,xx):
  if x<12 or y<12 or x>=a.shape[1]-12 or y>=a.shape[0]-12:continue
  patch=np.maximum(res[y-7:y+8,x-7:x+8],0)
  total=patch.sum()
  if total<=0:continue
  Y,X=np.indices(patch.shape);cx=(patch*X).sum()/total;cy=(patch*Y).sum()/total
  variance=(patch*((X-cx)**2+(Y-cy)**2)).sum()/total
  fwhm=2.355*np.sqrt(max(variance/2,0))
  if not .9<fwhm<5:continue
  if np.max(a[y-2:y+3,x-2:x+3])>=np.percentile(a,99.99):continue
  if res[y,x]<15*sigma:continue
  neighborhood=res[y-10:y+11,x-10:x+11].copy();neighborhood[7:14,7:14]=0
  if neighborhood.max()>0.30*res[y,x]:continue
  sources.append((int(x),int(y),float(fwhm),float(res[y,x]/sigma)))
 fwhms=np.array([s[2] for s in sources]);fwhm=float(np.median(fwhms)) if len(fwhms)>=5 else None
 # Central galaxy is excluded from the automatic stellar mask; do not erase compact star-forming knots.
 cy,cx=np.array(a.shape)/2
 mask=np.zeros_like(a,bool)
 for x,y,width,snr in sources:
  if np.hypot(x-cx,y-cy)<75:continue
  rad=int(np.clip(2.5*width,3,15));Y,X=np.ogrid[-rad:rad+1,-rad:rad+1];disk=X*X+Y*Y<=rad*rad
  if x>=rad and y>=rad and x+rad<a.shape[1] and y+rad<a.shape[0]:mask[y-rad:y+rad+1,x-rad:x+rad+1]|=disk
 im=Image.fromarray(np.uint8(255*np.arcsinh(8*np.clip((a-np.percentile(a,2))/(max(np.percentile(a,99.6)-np.percentile(a,2),1e-8)),0,1))/np.arcsinh(8))).convert('RGB').resize((512,512))
 overlay=ImageDraw.Draw(im)
 for x,y,w,snr in sources:
  if np.hypot(x-cx,y-cy)<75:continue
  rad=int(np.clip(2.5*w,3,15));overlay.ellipse((x-rad,y-rad,x+rad,y+rad),outline='#ffbd5e',width=1)
 im.save(S/(stem+'_stars.jpg'),quality=90)
 record=dict(stem=stem,catalog_inclination=float(manifest.loc[stem,'i']),survey=survey,pixel_scale_arcsec=px,detected_psf_like_sources=len(sources),empirical_fwhm_px=fwhm,empirical_fwhm_arcsec=fwhm*px if fwhm is not None else None,mask_fraction=float(mask.mean()),status='candidate_only_not_validated_psf' if fwhm else 'insufficient_isolated_sources',note='Source masks are conservative; central 75-pixel radius excluded. No scientific image pixels modified.')
 rows.append(record);print('PSF_QC',stem,len(sources),fwhm,px,flush=True)
(S/'summary.json').write_text(json.dumps(rows,indent=2))
print('PSF_QC_DONE',len(rows))
