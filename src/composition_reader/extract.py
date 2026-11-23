from __future__ import annotations
import hashlib, io, warnings
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.ndimage import gaussian_filter
from skimage import measure, segmentation
from skimage.morphology import h_maxima
from parallax_notan.notan import notan_mass, value_map
from parallax_notan.colour import oklab_chroma
from .model import Scene

MAX_PIXELS=40_000_000
Image.MAX_IMAGE_PIXELS=MAX_PIXELS

def load_image(data: bytes, max_side=512):
    if len(data)>30_000_000: raise ValueError('Image exceeds 30 MB upload limit.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            im=Image.open(io.BytesIO(data))
            if im.width*im.height>MAX_PIXELS: raise ValueError('Image exceeds 40 megapixels.')
            im=ImageOps.exif_transpose(im)
            # Transparent pixels are composited against white; record the decision.
            transparency='A' in im.getbands() or 'transparency' in im.info
            if transparency:
                rgba=im.convert('RGBA');bg=Image.new('RGBA',rgba.size,'white');bg.alpha_composite(rgba);im=bg.convert('RGB')
            else: im=im.convert('RGB')
            size=list(im.size)
            if min(size)<32: raise ValueError('Image must be at least 32 pixels on each side.')
            im.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
            rgb=np.array(im,dtype=np.float64)/255
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning) as e:
        raise ValueError('Cannot read this image safely. Use PNG, JPEG, WebP or TIFF.') from e
    return rgb,{'sha256':hashlib.sha256(data).hexdigest(),'upright_size':size,
                'working_size':list(im.size),'exif_orientation_applied':True,
                'alpha_background':'white' if transparency else None}

def _regions(mass, lum):
    h,w=mass.shape
    # Separate polarities so adjacent light and dark bodies retain their identities.
    labels=np.zeros((h,w),np.int32);regions=[];next_id=1
    toned=gaussian_filter(lum,4);median=float(np.median(lum))
    total=float(mass.sum())
    # Prominence, rather than every spaced local peak, keeps a softly textured body
    # together. Each connected regional maximum supplies one watershed seed.
    peaks=measure.label(h_maxima(gaussian_filter(mass,3),.06))
    for polarity in [-1,1]:
        mask=(mass>=.06)&((toned<median) if polarity<0 else (toned>=median))
        markers=measure.label((peaks>0)&mask)
        if not markers.any():continue
        lab=segmentation.watershed(-mass,markers,mask=mask)
        for reg in measure.regionprops(lab,intensity_image=mass):
            sel=lab==reg.label;share=float(mass[sel].sum()/max(total,1e-12))
            if reg.area<h*w*.003 or share<.012:continue
            y,x=reg.centroid_weighted
            contours=measure.find_contours(np.pad(sel,1),.5)
            contour=max(contours,key=len)-1 if contours else np.empty((0,2))
            contour=measure.approximate_polygon(contour,2)
            contour=np.clip(contour[:,::-1]/[max(w-1,1),max(h-1,1)],0,1)
            labels[sel]=next_id
            y0,x0,y1,x1=reg.bbox
            regions.append({'id':f'r{next_id}','label':f'Mass {next_id}',
                'center':[x/(w-1),y/(h-1)],'mass_share':share,'area_share':float(reg.area/(h*w)),
                'polarity':'dark' if polarity<0 else 'light','bbox':[x0/w,y0/h,x1/w,y1/h],
                'contour':contour.tolist(),'source':'notan-watershed','included':True})
            next_id+=1
    regions.sort(key=lambda x:-x['mass_share'])
    return labels,regions[:20],max(0,len(regions)-20)

def _lines(rgb, chroma, min_length=.065, limit=64):
    h,w=rgb.shape[:2];diag=np.hypot(w,h);result=[]
    lum=rgb@np.array([.2126,.7152,.0722])
    detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    for name,field,sigma in [('tone',lum,1.8),('chroma',chroma,2.5)]:
        if np.ptp(field)<.005:continue
        f=gaussian_filter(field,sigma)
        # Chroma is displayed on its physical scale with a fixed multiplier, never per-image stretch.
        im=np.clip(f*(255 if name=='tone' else 1020),0,255).astype('uint8')
        found=detector.detect(im)[0]
        if found is None:continue
        for v in found[:,0]:
            a,b=v[:2],v[2:];length=float(np.linalg.norm(b-a)/diag)
            if length<min_length:continue
            angle=float(np.arctan2(b[1]-a[1],b[0]-a[0])%np.pi)
            c=(a+b)/2/np.array([w-1,h-1])
            if any(np.linalg.norm(c-np.mean(o['points'],axis=0))<.025 and abs((angle-o['angle']+np.pi/2)%np.pi-np.pi/2)<.06 for o in result):continue
            result.append({'id':f'l{len(result)+1}','points':(np.array([a,b])/[w-1,h-1]).tolist(),
                           'length':length,'angle':angle,'channel':name})
    return sorted(result,key=lambda r:-r['length'])[:limit]

def extract(rgb, source=None, border='raw', excluded=None, anchors=None):
    h,w=rgb.shape[:2];lum=rgb@np.array([.2126,.7152,.0722]);chroma=oklab_chroma(rgb)
    absolute=notan_mass(rgb,absolute=True);mass=notan_mass(rgb)
    energy=float(np.percentile(absolute,99));blank=energy<.004
    if blank:mass=np.zeros_like(mass)
    if border=='protected':
        yy,xx=np.mgrid[:h,:w];d=np.minimum.reduce([xx/(w-1),(w-1-xx)/(w-1),yy/(h-1),(h-1-yy)/(h-1)])
        weight=.5+.5*np.clip((d-.02)/.05,0,1);mass=mass*weight;absolute=absolute*weight
    labels,regions,dropped=_regions(mass,lum) if not blank else (np.zeros((h,w),np.int32),[],0)
    detected=regions
    border_regions=[]
    if border=='protected' and regions:
        # A mat or frame survives the halved weighting as a ring-shaped mass whose
        # centroid falls in the empty middle of the frame. Drop tonal masses that lie
        # mostly inside the protected band (outer 7%); interior bodies that merely touch it stay.
        band=d<.07
        for r in regions:
            sel=labels==int(r['id'][1:])
            r['border_band_fraction']=float(band[sel].mean()) if sel.any() else 0.
        border_regions=[r['id'] for r in regions if r['border_band_fraction']>.6]
        regions=[r for r in regions if r['id'] not in border_regions]
    excluded=set(excluded or [])
    regions=[r for r in regions if r['id'] not in excluded]
    subject_map=np.zeros((h,w),dtype=float);subjects=[];subject_metadata={'method':'insufficient-contrast'}
    if not blank and anchors is None:
        from .subjects import select_subjects
        subject_map,subjects,subject_metadata=select_subjects(rgb,excluded)
    if anchors is not None:
        # Explicit correction replaces automatic anchor positions; map measurements remain unchanged.
        regions=[]
        for i,p in enumerate(anchors):
            x,y=map(float,p)
            if not (0<=x<=1 and 0<=y<=1):raise ValueError('Anchor coordinates must be in the image.')
            ix,iy=round(x*(w-1)),round(y*(h-1));rad=max(3,round(min(h,w)*.035))
            m=float(mass[max(0,iy-rad):iy+rad+1,max(0,ix-rad):ix+rad+1].sum())
            regions.append({'id':f'u{i+1}','label':f'Your anchor {i+1}','center':[x,y],
                            'mass_share':max(m,1e-8),'area_share':0,'polarity':'user',
                            'bbox':[x,y,x,y],'contour':[],'source':'user','included':True})
        total=sum(r['mass_share'] for r in regions)
        for r in regions:r['mass_share']/=total or 1
        subjects=[{**r,'placement_weight':r['mass_share'],'included':r['id'] not in excluded} for r in regions]
        subject_metadata={'method':'user-anchors','guide_independent':True,'category_probability':False}
    lines=_lines(rgb,chroma) if not blank else []
    edges=cv2.Canny(np.uint8(np.clip(lum*255,0,255)),50,130)>0
    total=mass.sum();yy,xx=np.mgrid[:h,:w]
    center=[float((xx*mass).sum()/total/(w-1)),float((yy*mass).sum()/total/(h-1))] if total>1e-12 else None
    coverage=sum(r['mass_share'] for r in detected)
    warning=[]
    if blank:warning.append('Insufficient absolute contrast for a structural reading; normalized noise is not treated as composition.')
    if coverage<.5 and not blank:warning.append('Detected regions account for less than half of the mass field. Placement hypotheses may miss diffuse structure.')
    if dropped:warning.append(f'{dropped} smaller regions exceed the 20-region reporting limit.')
    if subject_metadata.get('sky_like_set_aside'):warning.append(f'Foreground component(s) {", ".join(subject_metadata["sky_like_set_aside"])} look like sky (hanging from the top edge, bright, upper part of the frame) and are not used as subjects.')
    if border_regions:warning.append(f'Border protection set aside {len(border_regions)} tonal mass(es) lying mostly in the outer 7% band ({", ".join(border_regions)}); they read as frame or mat, not composition.')
    if anchors is not None:warning.append('Placement hypotheses use your corrected anchors. Field measurements still describe the complete image.')
    elif not blank and not any(r['included'] for r in subjects):warning.append('No distinct included foreground subject supports placement. Inspect the subject map or mark anchors; tonal masses are not silently substituted.')
    summary={'mass_center':center,'flat_fraction':float((mass<.06).mean()),'absolute_signal':energy,
             'region_coverage':min(1,coverage),'region_count':len(regions),'line_count':len(lines),
             'blank':blank,'border':border,'border_regions_set_aside':border_regions,'anchor_mode':'user' if anchors is not None else 'automatic',
             'detected_regions':detected,'subject_count':sum(r['included'] for r in subjects),
             'placement_mode':'user' if anchors is not None else 'foreground'}
    return Scene(rgb,mass,absolute,lum,chroma,edges,labels,regions,lines,summary,warning,source or {},subjects,subject_map,subject_metadata)
