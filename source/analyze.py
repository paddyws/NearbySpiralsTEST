"""Exploratory 35-galaxy diagnostics. Never modifies source data or asserts GALFIT-quality fits."""
from pathlib import Path
import json,re,traceback
import numpy as np
from astropy.io import fits
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import gaussian_filter,map_coordinates,rotate,label,distance_transform_edt
from scipy.optimize import least_squares,nnls
from scipy.signal import fftconvolve
from skimage.morphology import skeletonize,remove_small_objects
from skimage.measure import find_contours
R=Path('E:/Astro/Nearby_IMBH');SRC=R/'JUST_THE_35_GALAXIES';V6=R/'BIRTHDAY_IMBH_20261009/BIRTHDAY_GITHUB_STYLE_V6/data';OUT=Path(__file__).parent/'site';(OUT/'images').mkdir(exist_ok=True)
COLORS=['#49e9ff','#ff98d7','#a4ff98','#ffd078','#b8a8ff','#7bdbff','#ffae8d']

def norm(a,lo=None,hi=None):
 a=np.nan_to_num(a,nan=0,posinf=0,neginf=0)
 if lo is None:lo=np.percentile(a,2)
 if hi is None:hi=np.percentile(a,99.6)
 z=np.clip((a-lo)/max(hi-lo,1e-10),0,1)
 return Image.fromarray(np.uint8(255*np.arcsinh(8*z)/np.arcsinh(8))).convert('RGB')
def signed(a,lim):
 z=np.clip(a/max(lim,1e-10),-1,1);z=np.sign(z)*np.arcsinh(4*abs(z))/np.arcsinh(4)
 rgb=np.zeros((*a.shape,3),np.uint8);rgb[:]=27
 rgb[...,0]=np.uint8(np.clip(27+215*np.maximum(z,0),0,255));rgb[...,2]=np.uint8(np.clip(27+215*np.maximum(-z,0),0,255))
 return Image.fromarray(rgb)
def save(im,p):im.resize((420,420),Image.Resampling.BILINEAR).save(p)
def svg_points(path):
 if not path:return np.empty((0,2))
 return np.array([[float(x),float(y)] for x,y in re.findall(r'[ML]\s*([\d.-]+),([\d.-]+)',path)],float)
def geo(stem):
 p=R/'autoprof_geometry'/f'{stem}.npz'
 if not p.exists():return None
 with np.load(p) as z:
  q=float(np.clip(1-np.median(z['fit_ellip'][-min(5,len(z['fit_ellip'])):]),.15,1));pa=float(np.median(z['fit_pa'][-min(5,len(z['fit_pa'])):]));cx=float(z['center_x']);cy=float(z['center_y']);rad=float(z['fit_R'][-1])
 return dict(q=q,pa=pa,cx=cx,cy=cy,radius=rad)
def face_coords(g,n=256):
 yy,xx=np.indices((n,n),float);dx=(xx+.5)*512/n-.5-g['cx'];dy=(yy+.5)*512/n-.5-g['cy'];c=np.cos(g['pa']);s=np.sin(g['pa']);u=c*dx+s*dy;v=(-s*dx+c*dy)/g['q'];return u,v

def components(image,mask,g):
 """Two-component radial approximation, not a GALFIT bulge/disk/bar decomposition."""
 yy,xx=np.indices(image.shape);dx=xx-g['cx'];dy=yy-g['cy'];c=np.cos(g['pa']);s=np.sin(g['pa']);u=c*dx+s*dy;v=(-s*dx+c*dy)/g['q'];r=np.hypot(u,v)
 sky=float(np.median(image[r>min(225,g['radius']*1.8)])) if np.sum(r>min(225,g['radius']*1.8))>1000 else float(np.median(image[:32,:]))
 signal=image-sky;valid=(~mask)&np.isfinite(signal)&(r<min(235,max(65,1.8*g['radius'])))
 rr=np.geomspace(2,min(220,max(70,1.65*g['radius'])),75);median=[];weights=[]
 for rad in rr:
  m=valid&(abs(r-rad)<max(1.5,rad*.05));vls=signal[m]
  median.append(float(np.median(vls)) if len(vls)>15 else np.nan);weights.append(len(vls))
 median=np.array(median);ok=np.isfinite(median)&(np.array(weights)>15)
 if ok.sum()<15:raise ValueError('Insufficient radial samples')
 # Exponential disk and fixed-index n=2 Sersic-like bulge. Robustly fitted to radial medians.
 def func(t,r):return np.exp(t[0])*np.exp(-r/np.exp(t[1]))+np.exp(t[2])*np.exp(-1.672*((r/np.exp(t[3]))**.5-1))
 scale=max(np.percentile(median[ok],90),1e-4)
 x0=np.log([max(scale*.55,1e-6),max(g['radius']*.55,8),max(scale*.08,1e-6),max(g['radius']*.12,2)])
 lower=np.log([1e-8,2,1e-9,1]);upper=np.log([max(scale*30,1e-3),500,max(scale*30,1e-3),200])
 x0=np.clip(x0,lower+.001,upper-.001)
 fit=least_squares(lambda t:(func(t,rr[ok])-median[ok])/max(scale*.1,1e-4),x0,bounds=(lower,upper),loss='soft_l1',max_nfev=90)
 a,h,bb,re=np.exp(fit.x)
 disk=a*np.exp(-r/h);bulge=bb*np.exp(-1.672*((r/re)**.5-1));bar=np.zeros_like(disk)
 return sky,bulge,disk,bar,dict(status='approximate_radial_profile_only',disk_scale_pix=float(h),bulge_scale_pix=float(re),fit_cost=float(fit.cost),bar_status='not_fit_no_reliable_bar_decomposition',PSF='not_convolved',photometry='mixed_uncalibrated_sources')

def face_image(img,g,n=256):
 y,x=np.indices((n,n),float);u=(x-(n-1)/2)*512/n;v=(y-(n-1)/2)*512/n;c=np.cos(g['pa']);s=np.sin(g['pa']);nx=g['cx']+c*u-s*g['q']*v;ny=g['cy']+s*u+c*g['q']*v
 return map_coordinates(img,[ny,nx],order=1,mode='constant',cval=float(np.median(img)))
def back_to_native(points,g):
 p=np.asarray(points,float);u=(p[:,0]-127.5)*2;v=(p[:,1]-127.5)*2;c=np.cos(g['pa']);s=np.sin(g['pa']);return np.c_[g['cx']+c*u-s*g['q']*v,g['cy']+s*u+c*g['q']*v]

def gabor_curved(img):
 """Local curved-Gabor response using structure-tensor orientation and traced tangent samples."""
 base=gaussian_filter(img,1.0);gx,gy=np.gradient(base,axis=1),np.gradient(base,axis=0)
 jxx=gaussian_filter(gx*gx,3);jyy=gaussian_filter(gy*gy,3);jxy=gaussian_filter(gx*gy,3)
 tangent=.5*np.arctan2(2*jxy,jxx-jyy)+np.pi/2
 # Trace tangent field for each pixel, sampling transverse oscillation at each step.
 Y,X=np.indices(img.shape,dtype=float);out=np.zeros_like(img);den=0.
 for longitudinal in range(-5,6):
  tx=X.copy();ty=Y.copy();sign=np.sign(longitudinal)
  for _ in range(abs(longitudinal)):
   a=map_coordinates(tangent,[ty,tx],order=1,mode='nearest');tx+=sign*np.cos(a);ty+=sign*np.sin(a)
  a=map_coordinates(tangent,[ty,tx],order=1,mode='nearest')
  for normal in range(-3,4):
   x=tx-normal*np.sin(a);y=ty+normal*np.cos(a)
   weight=np.exp(-.5*((longitudinal/3)**2+(normal/1.7)**2))*np.cos(2*np.pi*normal/4.2)
   out+=weight*map_coordinates(base,[y,x],order=1,mode='nearest');den+=abs(weight)
 out/=max(den,1e-10)
 return out,tangent

def gabor_tracks(response,g):
 smooth=gaussian_filter(response,1.0);lo=np.percentile(smooth,75);hi=np.percentile(smooth,97)
 positive=smooth>max(lo,hi*.23)
 # Centerline candidates are extracted from skeleton connected paths; no invented bridges.
 skel=skeletonize(remove_small_objects(positive,min_size=35))
 lab,n=label(skel,structure=np.ones((3,3)));paths=[]
 for k in range(1,n+1):
  ys,xs=np.nonzero(lab==k)
  if len(xs)<30:continue
  pts=np.c_[xs,ys];ctr=np.array([127.5,127.5]);rr=np.hypot(pts[:,0]-ctr[0],pts[:,1]-ctr[1]);valid=(rr>10)&(rr<min(122,g['radius']*.65))
  pts=pts[valid];rr=rr[valid]
  if len(pts)<25:continue
  # Sort radial points, reject disconnected or multi-valued angular tracks.
  ix=np.argsort(rr);pts=pts[ix];rr=rr[ix];ang=np.unwrap(np.arctan2(pts[:,1]-127.5,pts[:,0]-127.5));u=np.log(np.maximum(rr,1))
  if np.ptp(u)<.15 or np.ptp(ang)>5*np.pi:continue
  slopes=np.diff(ang)/np.maximum(np.diff(u),.005);slopes=slopes[np.isfinite(slopes)&(abs(slopes)<50)]
  if len(slopes)<8:continue
  pitch=float(np.degrees(np.arctan2(1,max(abs(np.median(slopes)),.01))))
  if not 4<=pitch<=75:continue
  native=back_to_native(pts,g)
  paths.append(dict(pitch_deg=pitch,xy=native.tolist(),face_xy=pts.tolist(),source='curved-Gabor response skeleton; exploratory'))
 paths.sort(key=lambda p:-len(p['xy']));return paths[:14]

def model_arms(residual,mask,curves):
 # One nonnegative ribbon per segment, fixed 2.5px width; same target for all methods.
 templates=[];Y,X=np.indices(residual.shape);region=(~mask)&np.isfinite(residual)
 for c in curves[:16]:
  pts=np.asarray(c['xy'],float)
  if len(pts)<8:continue
  img=np.ones(residual.shape,bool)
  for x,y in pts[::max(1,len(pts)//450)]:
   xi=int(round(x));yi=int(round(y))
   if 0<=xi<512 and 0<=yi<512:img[yi,xi]=False
  distance=distance_transform_edt(img);temp=np.exp(-.5*(distance/2.5)**2)
  if temp.max()>.01:templates.append(temp)
 if not templates:return np.zeros_like(residual),0
 # Fit on subsampled valid native pixels to avoid overfitting tiny bright knots.
 iy,ix=np.nonzero(region[::3,::3]);y=residual[::3,::3][iy,ix];A=np.stack([t[::3,::3][iy,ix] for t in templates],axis=1)
 coeff,_=nnls(A,y,maxiter=200)
 return sum(float(a)*t for a,t in zip(coeff,templates)),int(np.sum(coeff>max(coeff.max()*1e-6,1e-10)))

def old_curves(stem,method):
 p=V6/f'{stem}.json'
 if not p.exists():return []
 d=json.loads(p.read_text(encoding='utf-8'));items=d['tracks'] if method=='track_v2' else d['fourier'];out=[]
 for t in items:
  pts=svg_points(t.get('native',''))*.8
  if len(pts)<5:continue
  out.append(dict(pitch_deg=float(t['pitch']),xy=pts.tolist(),source='saved exploratory '+method+' overlay; not refitted'))
 return out

def overlay(im,curves,g,face=False):
 im=im.copy().convert('RGB');draw=ImageDraw.Draw(im)
 for i,c in enumerate(curves):
  pts=np.asarray(c['face_xy'] if face and 'face_xy' in c else c['xy'],float)
  if face and 'face_xy' not in c:
   dx=pts[:,0]-g['cx'];dy=pts[:,1]-g['cy'];co=np.cos(g['pa']);si=np.sin(g['pa']);u=co*dx+si*dy;v=(-si*dx+co*dy)/g['q'];pts=np.c_[127.5+u/2,127.5+v/2]
  if len(pts)<5:continue
  sx=420/(256 if face else 512);q=pts*sx
  # Do not bridge large gaps from unordered ridge skeleton points.
  col=COLORS[i%len(COLORS)]
  for a,b in zip(q[:-1],q[1:]):
   if np.linalg.norm(a-b)<30 and np.all(a>=0) and np.all(b>=0) and np.all(a<420) and np.all(b<420):draw.line([tuple(a),tuple(b)],fill=col,width=2)
  anchor=q[len(q)//2];x=float(np.clip(anchor[0],2,370));y=float(np.clip(anchor[1],2,397))
  draw.rectangle((x,y,x+49,y+17),fill='#121922',outline=col,width=1)
  draw.text((x+2,y+2),f'{c["pitch_deg"]:.1f}°',fill=col)
 return im

def run_one(stem):
 from time import time
 start=time();dest=OUT/'images'/stem;dest.mkdir(parents=True,exist_ok=True)
 img=fits.getdata(SRC/f'{stem}.fits').astype(float);assert img.shape==(512,512)
 rgb=SRC/f'{stem}.jpg';color=Image.open(rgb).convert('RGB').resize((420,420)) if rgb.exists() else norm(img).resize((420,420))
 color.save(dest/'original.jpg',quality=90)
 g=geo(stem);record=dict(stem=stem,geometry=g,has_color=rgb.exists(),methods={},status='ok')
 if g is None:
  record.update(status='no_autoprof_geometry',warning='No reliable face-on geometry. No arm pitches or component fits generated.')
  save(norm(img),dest/'native_g.png');(dest/'result.json').write_text(json.dumps(record));return record
 maskfile=R/'STAR_MASKED_35/masks'/f'{stem}_gaia_mask.fits'
 mask=fits.getdata(maskfile)>0 if maskfile.exists() else np.zeros(img.shape,bool)
 cleaned=R/'STAR_MASKED_35/cleaned_fits'/f'{stem}.fits'
 if cleaned.exists():
  cleaned_data=fits.getdata(cleaned).astype(float)
  if cleaned_data.shape==img.shape:img_fit=np.where(np.isfinite(cleaned_data),cleaned_data,img)
  else:img_fit=img
 else:img_fit=img
 record['star_mask_available']=maskfile.exists()
 try:sky,bulge,disk,bar,meta=components(img_fit,mask,g)
 except Exception as exc:
  record.update(status='smooth_fit_failed',warning=str(exc));(dest/'result.json').write_text(json.dumps(record));return record
 smooth=bulge+disk+bar;residual=img_fit-sky-smooth
 lim=max(np.percentile(abs(residual[~mask]),98.5),1e-8)
 lo=np.percentile((img_fit-sky)[~mask],2);hi=np.percentile((img_fit-sky)[~mask],99.5)
 native=norm(img_fit-sky,lo,hi);save(native,dest/'native_g.png')
 face=face_image(img_fit-sky,g);face_res=face_image(residual,g)
 save(norm(face,lo,hi),dest/'face_g.png');save(signed(residual,lim),dest/'before.png');save(signed(face_res,lim),dest/'face_residual.png')
 for name,arr in [('bulge',bulge),('disk',disk),('bar',bar),('smooth',smooth)]:save(norm(arr,lo,hi),dest/(name+'.png'))
 record['smooth_model']=meta;record['quality_flags']=[]
 if g['q']<.35:record['quality_flags'].append('HIGH_INCLINATION_UNRELIABLE_DEPROJECTION')
 if not maskfile.exists():record['quality_flags'].append('NO_GAIA_STAR_MASK')
 if meta['fit_cost']>300:record['quality_flags'].append('SMOOTH_PROFILE_POOR_FIT')
 # Existing tracks and Fourier are independent saved exploratory products, not new measurements.
 curves={'track_v2':old_curves(stem,'track_v2'),'fourier':old_curves(stem,'fourier')}
 response,tangent=gabor_curved(face_res)
 save(norm(response),dest/'gabor_response.png')
 curves['gabor_curved']=gabor_tracks(response,g)
 for method in ['track_v2','gabor_curved','fourier']:
  cs=curves[method]
  arm,terms=model_arms(residual,mask,cs)
  rec=smooth+arm;after=residual-arm
  save(norm(arm,0,max(hi-lo,1e-8)),dest/(method+'_arms.png'))
  save(norm(rec,lo,hi),dest/(method+'_model.png'))
  save(signed(after,lim),dest/(method+'_residual.png'))
  overlay(color,cs,g).save(dest/(method+'_overlay.jpg'),quality=92)
  overlay(norm(face,lo,hi).resize((420,420)),cs,g,face=True).save(dest/(method+'_face.jpg'),quality=92)
  region=(~mask)&(np.hypot(*np.indices(img.shape)-255.5)<200) if False else (~mask)
  base=float(np.sum(residual[region]**2));new=float(np.sum(after[region]**2))
  record['methods'][method]=dict(status='exploratory',segments=len(cs),active_brightness_terms=terms,
    residual_sse_reduction_fraction=(base-new)/max(base,1e-20),pitches_deg=[round(c['pitch_deg'],2) for c in cs],
    source=cs[0]['source'] if cs else 'no_retained_segments')
 record['elapsed_sec']=round(time()-start,2)
 (dest/'result.json').write_text(json.dumps(record,indent=2))
 return record

if __name__=='__main__':
 import concurrent.futures,sys
 stems=sorted(p.stem for p in SRC.glob('*.fits'))
 print('START',len(stems),flush=True)
 results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  fs={ex.submit(run_one,s):s for s in stems}
  for f in concurrent.futures.as_completed(fs):
   s=fs[f]
   try:row=f.result()
   except Exception as exc:row=dict(stem=s,status='error',error=str(exc),traceback=traceback.format_exc())
   results.append(row);print('GALAXY',len(results),'/',len(stems),s,row['status'],row.get('elapsed_sec'),flush=True)
 (OUT/'analysis_summary.json').write_text(json.dumps(sorted(results,key=lambda x:x['stem']),indent=2))
 print('COMPLETE',len(results),'errors',sum(r['status']=='error' for r in results),flush=True)
