"""Ablation audit: is the bright central 'ring' in the data or produced by subtraction/filtering?"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze import R,SRC,geo,face_image,components,gabor_curved
stem='NGC5236';g=geo(stem)
a=fits.getdata(SRC/(stem+'.fits')).astype(float)
mask=np.zeros(a.shape,bool)
sky,bulge,disk,bar,meta=components(a,mask,g)
face=face_image(a-sky,g)
smooth=face_image(bulge+disk+bar,g)
residual=face-smooth
# Alternative to parametric subtraction: a local multiscale background from the original image.
highpass=face-gaussian_filter(face,12)
raw_response,_=gabor_curved(face)
res_response,_=gabor_curved(residual)
hp_response,_=gabor_curved(highpass)
Y,X=np.indices(face.shape);r=np.hypot(X-127.5,Y-127.5)
radial=[]
for rad in np.arange(3,85,2):
 m=(r>=rad-1)&(r<rad+1)
 radial.append(dict(radius_px=int(rad),raw_median=float(np.median(face[m])),smooth_median=float(np.median(smooth[m])),residual_median=float(np.median(residual[m])),highpass_median=float(np.median(highpass[m])),residual_p10=float(np.percentile(residual[m],10)),residual_p90=float(np.percentile(residual[m],90))))
D=Path(__file__).parent/'site/ngc5236_ring_audit';D.mkdir(exist_ok=True)
# Comparable scale across response panels, so a display normalization does not invent visual differences.
fig,axs=plt.subplots(2,3,figsize=(15,10),layout='constrained')
imgs=[face,residual,highpass,raw_response,res_response,hp_response]
titles=['Original face-on FITS (sky subtracted)','After provisional bulge/disk subtraction','Nonparametric high-pass (12 px)','Curved Gabor on original','Curved Gabor after bulge/disk subtraction','Curved Gabor after nonparametric high-pass']
for i,(ax,im,title) in enumerate(zip(axs.flat,imgs,titles)):
 if i==0:
  lo,hi=np.percentile(im,[2,99.7]);ax.imshow(np.arcsinh(np.clip((im-lo)/max(hi-lo,1e-6),0,1)*8),cmap='gray')
 elif i in (1,2):
  v=max(np.percentile(abs(residual),99),np.percentile(abs(highpass),99));ax.imshow(im,cmap='RdBu_r',vmin=-v,vmax=v)
 else:
  v=max(np.percentile(abs(x),99) for x in imgs[3:]);ax.imshow(im,cmap='gray',vmin=-v,vmax=v)
 ax.set_title(title,fontsize=11);ax.set_xlim(60,196);ax.set_ylim(196,60);ax.axis('off')
fig.savefig(D/'six_way_ablation.png',dpi=130);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5));rr=[x['radius_px'] for x in radial]
for name in ['residual_median','highpass_median']:
 ax.plot(rr,[x[name] for x in radial],label=name.replace('_',' '))
ax.axhline(0,color='black',linewidth=.6);ax.set_xlabel('Face-on radius (256 px frame)');ax.set_ylabel('Median intensity (original FITS units)');ax.set_title('Azimuthal-median radial profile: subtraction artifact test');ax.legend();fig.savefig(D/'radial_profiles.png',dpi=140,bbox_inches='tight');plt.close(fig)
records=dict(galaxy=stem,source_survey='DSS2',pixel_scale_arcsec=float(fits.getheader(SRC/(stem+'.fits')).get('PIXSCALE')),geometry=g,provisional_model=meta,radial_profile=radial,interpretation='This ablation compares input-dependent ring-like enhancement. It does not establish a physical stellar ring or calibrated PSF.',source_preserved=True)
(D/'audit.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('RING_AUDIT_DONE',D,flush=True)
