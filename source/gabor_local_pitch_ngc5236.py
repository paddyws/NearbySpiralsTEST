"""NGC5236 exploratory local orientation pitch: no arm-track requirement."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).parent))
from analyze import R,SRC,OUT,geo,face_image,components,gabor_curved,norm
stem='NGC5236';g=geo(stem);a=fits.getdata(SRC/(stem+'.fits')).astype(float)
maskfile=R/'STAR_MASKED_35/masks'/f'{stem}_gaia_mask.fits';mask=fits.getdata(maskfile)>0 if maskfile.exists() else np.zeros_like(a,bool)
clean=R/'STAR_MASKED_35/cleaned_fits'/f'{stem}.fits'
if clean.exists():
 b=fits.getdata(clean).astype(float);a=np.where(np.isfinite(b),b,a)
sky,bulge,disk,bar,meta=components(a,mask,g)
residual=a-sky-bulge-disk-bar
face=face_image(residual,g)
response,tangent=gabor_curved(face)
# Orientation is re-estimated from the enhanced response itself, not its input tensor.
for sigma in (1.5,2.5,4.0):
 f=gaussian_filter(response,sigma);gy,gx=np.gradient(f)
 Jxx=gaussian_filter(gx*gx,2*sigma);Jyy=gaussian_filter(gy*gy,2*sigma);Jxy=gaussian_filter(gx*gy,2*sigma)
 normal=.5*np.arctan2(2*Jxy,Jxx-Jyy);direction=normal+np.pi/2
 coherence=np.sqrt((Jxx-Jyy)**2+4*Jxy**2)/(Jxx+Jyy+1e-9)
 Y,X=np.indices(f.shape);dx=X-127.5;dy=Y-127.5;rr=np.hypot(dx,dy)
 radial=np.arctan2(dy,dx)
 # Pitch is angle between tangent and local azimuthal direction, not radial direction.
 azimuth=radial+np.pi/2
 pitch=np.degrees(np.arccos(np.clip(abs(np.cos(direction-azimuth)),0,1)))
 strength=np.sqrt(Jxx+Jyy)
 valid=(rr>18)&(rr<min(115,g['radius']*.55))&(coherence>.35)&(strength>np.percentile(strength[(rr>18)&(rr<110)],65))
 # Positive residual and strong Gabor evidence to limit bulge-driven false ridges.
 valid &= (face>np.percentile(face[(rr>18)&(rr<110)],50))
 n=int(valid.sum());median=float(np.median(pitch[valid])) if n else None
 out=Path(__file__).parent/'site/gabor_pitch_ngc5236';out.mkdir(parents=True,exist_ok=True)
 from matplotlib import pyplot as plt
 import matplotlib
 fig,ax=plt.subplots(figsize=(7,7));ax.imshow(norm(face),origin='upper');im=ax.imshow(np.ma.masked_where(~valid,pitch),cmap='turbo',vmin=0,vmax=75,alpha=.82,origin='upper');ax.set_title(f'NGC5236 local Gabor pitch, scale {sigma} px | {n} selected pixels');ax.axis('off');fig.colorbar(im,ax=ax,label='Local pitch (degrees)');fig.savefig(out/f'pitch_sigma_{sigma}.png',dpi=130,bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,7));ax.imshow(norm(face),origin='upper');ax.contour(valid,levels=[.5],colors='cyan',linewidths=.6);ax.set_title(f'NGC5236: selected coherent pixels | scale {sigma}');ax.axis('off');fig.savefig(out/f'evidence_sigma_{sigma}.png',dpi=130,bbox_inches='tight');plt.close(fig)
 if sigma==1.5:records=[]
 records.append(dict(sigma_px=sigma,selected_pixels=n,median_local_pitch_deg=median,pitch_p10_p90=np.percentile(pitch[valid],[10,90]).tolist() if n else None,notes='Local orientation only; no arm identity, not validated physical pitch'))
(out/'summary.json').write_text(json.dumps(records,indent=2))
print('GABOR_LOCAL_PITCH_NGC5236',json.dumps(records),flush=True)
