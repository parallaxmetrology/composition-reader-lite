"""Placement geometry and extent-conditioned controls, shared by app and audits."""
from __future__ import annotations
import copy
import numpy as np
from scipy.stats import rankdata
from .geometry import PHI,templates,point

POSITIONS={'rule-of-thirds':[1/3,2/3],'golden-grid':[1-1/PHI,1/PHI],'rule-of-fifths':[.2,.4,.6,.8],'centre-cross':[.5]}

def subject_spans(s):
    spans=[]
    for r in s.placement_regions:
        b=r.get('bbox',[0,0,0,0]);spans.append(np.clip(np.array(b[2:])-b[:2],0,1))
    return np.array(spans,dtype=float).reshape(-1,2)

def frame_contacts(region,shape):
    """Conservative frame contact within one working pixel, before any guide."""
    p=np.array(region['center'],float);b=np.array(region.get('bbox',np.r_[p,p]),float)
    if not np.any(b[2:]>b[:2]):return np.zeros(4,dtype=bool)
    h,w=shape[:2];tol=np.array([1/w,1/h])
    return np.r_[b[:2]<=tol,b[2:]>=1-tol]

def randomize_subjects(s,rng):
    """Preserve footprints, anchor offsets, and observed frame contacts."""
    regions=s.subjects if s.subjects is not None else s.regions
    out=[]
    for r in regions:
        original=np.array(r['center'],float);b=np.array(r.get('bbox',np.r_[original,original]),float)
        left=original-b[:2];right=b[2:]-original
        low=np.maximum(.03,left);high=np.minimum(.97,1-right)
        p=rng.uniform(low,np.maximum(low,high))
        contacts=frame_contacts(r,s.rgb.shape);pinned=contacts[:2]|contacts[2:]
        p[pinned]=original[pinned]
        shift=p-original;box=b+np.r_[shift,shift]
        out.append({**r,'center':p.tolist(),'bbox':box.tolist()})
    return out

def features(s,id,positions,measure_only=False):
    pts=s.placement_points;weights=s.placement_weights;scale=s.metric_scale
    targets=np.array([(x,y) for x in positions for y in positions]);spans=subject_spans(s)
    parameters={'tolerance_diagonal':.045,'fixed_frame':True,'selection':s.subject_metadata.get('method','explicit-geometry'),
                'control_scope':'subject footprints and anchor offsets preserved; observed frame contacts remain fixed'}
    ev=[];point_fit=0.;axis_fit=0.;fit=0.
    if len(pts):
        d=np.linalg.norm((pts[:,None]-targets)*scale,axis=2);nearest=d.argmin(1);dist=d.min(1)
        affin=np.exp(-.5*(dist/.045)**2);point_fit=float(weights@affin)
        supports=np.c_[affin,np.zeros((len(pts),2))];distances=np.c_[dist,np.zeros((len(pts),2))]
        matches=np.repeat(targets[nearest,None],3,axis=1)
        if id=='rule-of-thirds':
            # Orientation/extent gates are chosen before comparing any guide.
            physical=spans*scale;vertical=np.clip((physical[:,1]/np.maximum(physical[:,0],1e-9)-1.2)/.8,0,1)*np.minimum(1,spans[:,1]/.35)
            horizontal=np.clip((physical[:,0]/np.maximum(physical[:,1],1e-9)-1.2)/.8,0,1)*np.minimum(1,spans[:,0]/.35)
            for axis,gate in [(0,vertical),(1,horizontal)]:
                dd=np.abs(pts[:,axis,None]-np.array(positions));which=dd.argmin(1);dd=dd.min(1)
                supports[:,axis+1]=(.85+.15*gate)*np.exp(-.5*(dd/.035)**2);distances[:,axis+1]=dd*scale[axis]
                matches[:,axis+1]=pts;matches[:,axis+1,axis]=np.array(positions)[which]
            axis_fit=float(weights@supports[:,1:].max(1))
            parameters.update(axis_tolerance_frame=.035,axis_gate='line alignment base 0.85; elongated-subject boost up to 1.0, chosen independently of guide position',point_intersection_fit=point_fit,line_alignment_fit=axis_fit)
        modes=supports.argmax(1);best=supports.max(1);fit=float(weights@best)
        axis_fit_values=[float(weights@np.exp(-.5*(((pts[:,a]-.5)*scale[a])/.045)**2)) for a in [0,1]] if id=='centre-cross' else None
        if measure_only:return {'fit':fit,'axis_fit':axis_fit_values}
        for j,(r,w) in enumerate(zip(s.placement_regions,weights)):
            mode=int(modes[j]);ev.append({'element_id':r['id'],'measure':['distance to nearest intersection','subject center to vertical thirds line','subject center to horizontal thirds line'][mode],
                'value':float(distances[j,mode]),'unit':'image diagonal','support':float(best[j]),'point_support':float(affin[j]),
                'alignment_mode':['intersection','vertical-grid-line','horizontal-grid-line'][mode],
                'target':matches[j,mode].tolist(),'placement_weight':float(w),'subject_extent':spans[j].tolist()})
        parameters['weighted_subject_extent']=(weights@spans).tolist()
        contacts=np.array([frame_contacts(r,s.rgb.shape) for r in s.placement_regions])
        parameters['frame_contacts']={r['id']:c.tolist() for r,c in zip(s.placement_regions,contacts)}
        parameters['frame_contact_order']=['left','top','right','bottom']
        parameters['frame_contact_tolerance_pixels']=1
        parameters['constrained_axis_weight']=(weights@(contacts[:,:2]|contacts[:,2:])).tolist()
        if id=='centre-cross':parameters['axis_fit']=axis_fit_values
    if measure_only:return {'fit':0.,'axis_fit':[0.,0.] if id=='centre-cross' else None}
    limits=['Foreground selection and weights are chosen before any guide. Segmentation can miss or merge subjects.',
            'A high geometric fit can be unsurprising for a large or cropped subject whose position is constrained by the frame.',
            'Frame contact is treated conservatively as a positional constraint; deliberate framing and truncation are not distinguished.']
    if id=='rule-of-thirds':
        limits.append('Separates point intersections from subject alignment with a grid line. Does not infer horizons, eye position, artistic intent or semantic composition classes.')
        explanation=f'{sum(e["support"]>=.6 for e in ev)} of {len(ev)} subjects support thirds placement: intersection fit {point_fit:.2f}, grid-line fit {axis_fit:.2f}.'
    else:explanation=f'{sum(e["support"]>=.6 for e in ev)} of {len(ev)} selected subject anchors lie close to this placement system.'
    contacted=sum(any(v) for v in parameters.get('frame_contacts',{}).values())
    if contacted:explanation+=f' {contacted} of {len(ev)} subject boxes touch the frame; controls hold those contacts fixed.'
    return dict(fit=fit,geometry=templates(s.aspect)[id]+[point(p) for p in targets],evidence=ev,explanation=explanation,
                limits=limits,eligible=len(pts)>0,parameters=parameters)

def calibrate(readings,nulls,center_axes,seed=173):
    """Single-step minP over exchangeable observed+null draws; ties conservative.

Raw family maxima mix non-comparable guide densities. First rank each guide's
own full-search statistic, then compare the family's minimum marginal tail.
This is geometric-null calibration, not semantic category confidence.
"""
    rows=[h for h in readings if h['family']=='placement'];n=len(next(iter(nulls.values())))
    if not n:return
    augmented=np.array([[h['fit'],*nulls[h['id']]] for h in rows]);tails=rankdata(-augmented,method='max',axis=1)/(n+1)
    family_min=tails.min(axis=0);raw_max=augmented.max(axis=0)
    for j,h in enumerate(rows):
        arr=augmented[j,1:];p=float(tails[j,0]);adjusted=float(np.mean(family_min<=p));raw=float(np.mean(raw_max>=h['fit']))
        h['control']={'method':'footprint and frame-contact conditioned translations; anchor offsets preserved; full mode search; rank-calibrated minP family comparison',
            'samples':n,'seed':seed,'rng_seed':seed+100003,'mean_fit':float(arr.mean()),'excess_fit':float(h['fit']-arr.mean()),'minimum_excess_fit':.1,'tail_fraction':p,
            'family_max_tail_fraction':raw,'adjusted_tail_fraction':adjusted,'decision_tail_key':'adjusted_tail_fraction',
            'comparison_label':'Rank-calibrated family comparison','interpretation':'Exploratory geometric null, not category probability. Raw family maximum is retained only as a diagnostic.'}
        informative=True
        if h['id']=='centre-cross' and h['eligible']:
            expected=np.mean(center_axes,axis=0);excess=np.array(h['parameters']['axis_fit'])-expected
            informative=bool(np.all(excess>=.05))
            h['control'].update(axis_mean_fit=expected.tolist(),axis_excess_fit=excess.tolist(),minimum_axis_excess=.05,two_axis_informative=informative)
            if not informative:h['explanation']+=' Centering is constrained by the subject footprint or frame contact in at least one axis; it is not a supported two-dimensional placement claim.'
        if h['eligible']:
            h['status']='supported' if h['fit']>=.5 and adjusted<=.1 and h['control']['excess_fit']>=.1 and informative else ('ambiguous' if h['fit']>=.45 else 'weak')
            if h['id']=='rule-of-thirds' and h['status']!='supported' and h['fit']>=.65 and h['control']['excess_fit']>=.2:
                h['status']='descriptive'
                h['explanation']+=' Descriptive thirds alignment only. Its fit exceeds the translation-control mean, but the family comparison has not ruled out accidental alignment.'
            h['control']['decision_rule']='Supported: fit >=0.5, family tail <=0.1 and excess over the conditional control mean >=0.10, with informative centering in both axes. Thirds may be descriptive at fit >=0.65 and excess over the control mean >=0.20; this is not a significance claim.'

def evaluate_placements(s,null_samples=64,seed=173):
    """Production placement-only path for sensitivity checks and feature-cache audits."""
    from .hypotheses import grid_read
    rows=[grid_read(s,id,pos) for id,pos in POSITIONS.items()]
    if null_samples and not s.summary['blank']:
        requested=null_samples;null_samples=max(256,null_samples)
        rng=np.random.default_rng(seed+100003);nulls={id:[] for id in POSITIONS};axes=[]
        for _ in range(null_samples):
            ns=copy.copy(s);ns.subjects=randomize_subjects(s,rng)
            for id,pos in POSITIONS.items():
                h=features(ns,id,pos,measure_only=True);nulls[id].append(h['fit'])
                if id=='centre-cross':axes.append(h['axis_fit'])
        calibrate(rows,nulls,axes,seed)
        for h in rows:h['control'].update(requested_samples=requested,minimum_samples=256,resolution_note='Placement uses at least 256 samples to resolve low tails in the four-guide comparison.')
    return rows
