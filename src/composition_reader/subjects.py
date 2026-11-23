"""Composition-blind foreground selection, separate from tonal mass measurements."""
import cv2
import numpy as np
from skimage import measure
from .subject_model import foreground,SPEC

def regions_from_mask(field,excluded=()):
    h,w=field.shape;excluded=set(excluded)
    if float(field.max())<.55 or float(np.ptp(field))<.25:return []
    mask=(field>=.5).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,connectivity=8)
    candidates=[]
    for id in range(1,count):
        x,y,rw,rh,area=stats[id];share=area/(h*w)
        if share<.001 or share>.85:continue
        sel=labels==id;strength=float(field[sel].mean())
        contours=measure.find_contours(np.pad(sel,1),.5)
        contour=measure.approximate_polygon(max(contours,key=len)-1,1.5) if contours else np.empty((0,2))
        contour=np.clip(contour[:,::-1]/[w-1,h-1],0,1)
        # Bounding-box center is an object's location, rather than its brightest
        # subregion. Importance depends on foreground area/strength, never a guide.
        weight=float(np.sqrt(share)*strength**2)
        candidates.append({'center':[(x+(rw-1)/2)/(w-1),(y+(rh-1)/2)/(h-1)],
            'bbox':[x/w,y/h,(x+rw)/w,(y+rh)/h],'area_share':float(share),
            'model_strength':strength,'selection_weight':weight,'contour':contour.tolist(),
            'source':'u2netp-foreground','polarity':'foreground'})
    candidates.sort(key=lambda r:(-r['selection_weight'],r['center']))
    candidates=candidates[:8]
    for i,r in enumerate(candidates,1):
        r.update(id=f's{i}',label=f'Subject {i}',included=f's{i}' not in excluded)
    total=sum(r['selection_weight'] for r in candidates if r['included'])
    for r in candidates:r['placement_weight']=r['selection_weight']/total if r['included'] and total else 0
    return candidates

def _mask(contour,shape):
    from skimage.draw import polygon as fill_polygon
    if not contour or len(contour)<3:return None
    h,w=shape;c=np.asarray(contour,float);rr,cc=fill_polygon(c[:,1]*(h-1),c[:,0]*(w-1),shape)
    m=np.zeros(shape,bool);m[rr,cc]=True
    return m if m.any() else None


def sky_like(region,lum):
    """A foreground component that is really sky (or a bright ceiling): it hangs from the top
    edge and a side (or spans half the width), does not reach the bottom, sits in the upper 40% and is brighter than the
    picture's median. Head-cropped portraits reach the bottom edge and are not affected."""
    x0,y0,x1,y1=region['bbox'];t=.02
    if not (y0<=t and (x0<=t or x1>=1-t or x1-x0>=.5) and y1<1-t and region['center'][1]<.4):return False
    m=_mask(region.get('contour'),lum.shape)
    return m is not None and float(lum[m].mean())>float(np.median(lum))


def select_subjects(rgb,excluded=()):
    field=foreground(rgb);regions=regions_from_mask(field,excluded)
    # Set aside sky-like components before anything uses them; recorded, never silent.
    lum=rgb@np.array([.2126,.7152,.0722]);set_aside=[]
    for r in regions:
        if r['included'] and sky_like(r,lum):
            r['included']=False;r['excluded_reason']='sky-like: hangs from the top edge across a side or half the width, upper 40%, brighter than the median';set_aside.append(r['id'])
    if set_aside:
        total=sum(r['selection_weight'] for r in regions if r['included'])
        for r in regions:r['placement_weight']=r['selection_weight']/total if r['included'] and total else 0
    return field,regions,{'method':'foreground-components/1','model':SPEC['name'],'model_sha256':SPEC['sha256'],'sky_like_set_aside':set_aside,
        'provider':'CPUExecutionProvider','network_size':[320,320],'threshold':.5,
        'importance':'sqrt(foreground area) × mean raw foreground strength squared, normalized across included components',
        'guide_independent':True,'category_probability':False,'foreground_fraction':float((field>=.5).mean()),
        'training_overlap_note':SPEC['training_note']}
