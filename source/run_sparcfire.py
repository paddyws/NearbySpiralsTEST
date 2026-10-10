"""Run the genuine installed SpArcFiRe single-turn profile on 34 AutoProf-deprojected FITS frames."""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image
from astropy.io import fits
from scipy.io import loadmat
from analyze import R,SRC,OUT,geo,face_image,norm,model_arms,overlay,signed,save,components,old_curves
from scipy.ndimage import map_coordinates
W=Path(__file__).parent;IN=W/'sparcfire_input';IN.mkdir(exist_ok=True)

def prepare():
 for p in sorted(SRC.glob('*.fits')):
  g=geo(p.stem)
  if g is None:continue
  a=fits.getdata(p).astype(float);f=face_image(a,g)
  norm(f).convert('L').resize((224,224)).save(IN/f'{p.stem}.png')
 print('SPARCFIRE_INPUTS',len(list(IN.glob('*.png'))),flush=True)

def extract(stem,g,params):
 arr=np.asarray(params,float)
 if arr.size==0:return []
 arr=np.atleast_2d(arr);curves=[]
 for i,row in enumerate(arr):
  th0,slope_deg,r0,t0,t1=row[:5]
  if not np.isfinite(row[:5]).all() or r0<=0 or t1<=t0 or row[9]!=0 or row[10]<15:continue
  slope=np.deg2rad(slope_deg);pitch=np.degrees(np.arctan(abs(slope)))
  if not 3<=pitch<=75:continue
  theta=np.linspace(t0,t1,400);radius=r0*np.exp(-slope*theta);th=-(th0+theta)
  x=111.5+radius*np.cos(th);y=111.5+radius*np.sin(th)
  x=(x+.5)*256/224-.5;y=(y+.5)*256/224-.5
  valid=(x>=0)&(x<=255)&(y>=0)&(y<=255)&(radius>7)&(radius*512/224<=g['radius']*1.1)
  if np.sum(valid)<20:continue
  pts=np.c_[x[valid],y[valid]]
  from analyze import back_to_native
  native=back_to_native(pts,g)
  curves.append(dict(pitch_deg=float(pitch),xy=native.tolist(),face_xy=pts.tolist(),source='genuine SpArcFiRe r140 single-turn'))
 return curves

def finish():
 output=W/'sparcfire_output/matout';rows=json.loads((OUT/'analysis_summary.json').read_text())
 for rec in rows:
  s=rec['stem'];g=geo(s)
  if g is None or rec['status']!='ok':continue
  p=output/f'gxyParamsList-part-{s}.mat'
  if not p.exists():rec['methods']['sparcfire']=dict(status='missing_engine_output',segments=0,pitches_deg=[]);continue
  d=loadmat(p,simplify_cells=True).get('gxyParamsList',{});state=str(d.get('fit_state','missing_state'))
  cs=extract(s,g,d.get('arc_params',[])) if state=='OK' else []
  dest=OUT/'images'/s
  img=fits.getdata(SRC/f'{s}.fits').astype(float)
  clean=R/'STAR_MASKED_35/cleaned_fits'/f'{s}.fits'
  if clean.exists():
   a=fits.getdata(clean).astype(float);img=np.where(np.isfinite(a),a,img)
  maskfile=R/'STAR_MASKED_35/masks'/f'{s}_gaia_mask.fits';mask=fits.getdata(maskfile)>0 if maskfile.exists() else np.zeros(img.shape,bool)
  sky,bulge,disk,bar,meta=components(img,mask,g);smooth=bulge+disk+bar;residual=img-sky-smooth
  lo=np.percentile((img-sky)[~mask],2);hi=np.percentile((img-sky)[~mask],99.5);lim=max(np.percentile(abs(residual[~mask]),98.5),1e-8)
  arm,terms=model_arms(residual,mask,cs)
  save(norm(arm,0,max(hi-lo,1e-8)),dest/'sparcfire_arms.png')
  save(norm(smooth+arm,lo,hi),dest/'sparcfire_model.png')
  save(signed(residual-arm,lim),dest/'sparcfire_residual.png')
  from PIL import Image
  color=Image.open(dest/'original.jpg').convert('RGB')
  overlay(color,cs,g).save(dest/'sparcfire_overlay.jpg',quality=92)
  overlay(norm(face_image(img-sky,g),lo,hi).resize((420,420)),cs,g,face=True).save(dest/'sparcfire_face.jpg',quality=92)
  base=float(np.sum(residual[~mask]**2));after=float(np.sum((residual-arm)[~mask]**2))
  rec['methods']['sparcfire']=dict(status='engine_'+state,segments=len(cs),active_brightness_terms=terms,pitches_deg=[round(c['pitch_deg'],2) for c in cs],residual_sse_reduction_fraction=(base-after)/max(base,1e-20),source='genuine SpArcFiRe r140 single-turn; existing AutoProf geometry')
  (dest/'result.json').write_text(json.dumps(rec,indent=2))
  print('SPARCFIRE',s,state,len(cs),flush=True)
 (OUT/'analysis_summary.json').write_text(json.dumps(rows,indent=2))
 print('SPARCFIRE_FINISHED',len(rows),flush=True)

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='finish':finish()
 else:prepare()
