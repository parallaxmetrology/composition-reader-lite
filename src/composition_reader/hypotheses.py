"""Named hypotheses measured against spatial evidence.

Fit values are within-method support indices, NOT probabilities or aesthetic grades.
"""
from __future__ import annotations
from itertools import combinations
import copy, math
import cv2
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, distance_transform_edt
from scipy.spatial import cKDTree
from .geometry import PHI, templates, segment_dist, angle_distance, line, point, poly
from .model import Scene

from .registry import NAMES, DETECTORS


def result(id,family,fit,geometry=None,evidence=None,explanation='',limits=None,eligible=True,parameters=None):
    spec=DETECTORS[id]
    if family!=spec.family:raise ValueError(f'Mismatched family for {id}')
    fit=float(np.clip(fit,0,1))
    return {'id':id,'name':NAMES.get(id,id),'family':family,'fit':fit,
            'status':('descriptive' if fit>=(parameters or {}).get('decision_threshold',.55) else 'weak') if eligible else 'insufficient',
            'geometry':geometry or [],'evidence':evidence or [],'explanation':explanation,
            'limitations':limits or [],'parameters':parameters or {},'control':None,'stability':None,
            'score_kind':'geometric support; not probability or quality','eligible':bool(eligible),'method_version':spec.version}


def grid_read(s,id,positions):
    from .placement import features
    return result(id,'placement',**features(s,id,positions))


def template_read(s,id,geometry):
    scale=s.metric_scale
    if not s.lines:
        return result(id,'armature',0,geometry,explanation='No sufficiently long line segments support this construction.',eligible=False)
    segments=np.array([a['points'] for a in s.lines]);mids=segments.mean(1)
    d=segment_dist(mids,geometry,scale)
    tg=np.array([g['points'] for g in geometry])*scale
    ta=np.arctan2(tg[:,1,1]-tg[:,0,1],tg[:,1,0]-tg[:,0,0])
    la=np.array([a['angle'] for a in s.lines]);length=np.array([a['length'] for a in s.lines])
    angle=angle_distance(la[:,None],ta[None])
    # Direction and spatial support both matter; a nearby perpendicular edge does not support a guide.
    affin=np.exp(-.5*(d/.025)**2-.5*(angle/.14)**2)
    best=affin.max(1);which=affin.argmax(1)
    fit=float(np.average(best,weights=length))
    ev=[{'element_id':l['id'],'measure':'line alignment','value':float(v),'unit':'support',
         'guide':int(k),'distance_diagonal':float(di),'angle_error_degrees':float(np.degrees(an))}
        for l,v,k,di,an in zip(s.lines,best,which,d[np.arange(len(d)),which],angle[np.arange(len(d)),which]) if v>.05]
    ev.sort(key=lambda e:-e['value'])
    return result(id,'armature',fit,geometry,ev,
        f'{sum(best>.6)} of {len(best)} long segments align in position and direction; {fit:.0%} of length-weighted support.',
        ['Measured on detected tone and chroma segments. Dense armatures have more opportunities to fit; inspect the randomized comparison.',
         'Alignment does not establish the artist’s construction method.'],parameters={'distance_tolerance':.025,'angle_tolerance_deg':8.02})


def direction_read(s,id,angle):
    if not s.lines:return result(id,'direction',0,eligible=False,explanation='No long line segments were detected.')
    a=np.array([x['angle'] for x in s.lines]);w=np.array([x['length'] for x in s.lines])
    v=np.exp(-.5*(angle_distance(a,angle)/.15)**2);fit=np.average(v,weights=w)
    selected=[(l,float(q)) for l,q in zip(s.lines,v) if q>.3]
    return result(id,'direction',fit,[line(*l['points']) for l,v in selected],
                  [{'element_id':l['id'],'measure':'orientation alignment','value':v,'unit':'support'} for l,v in selected],
                  f'{fit:.0%} length-weighted support for {"horizontal" if angle==0 else "vertical"} organization.',
                  ['Long-edge orientation is not the same as an implied route between figures.'])


def triangle_read(s):
    pts=s.points;w=s.weights;scale=s.metric_scale
    if len(pts)<3:return result('triangle-masses','arrangement',0,eligible=False,explanation='A triangular arrangement needs at least three measured anchors.')
    best=None
    for ids in combinations(range(min(len(pts),12)),3):
        p=pts[list(ids)]*scale
        u,v=p[1]-p[0],p[2]-p[0]
        twice=abs(u[0]*v[1]-u[1]*v[0])
        sides=np.linalg.norm(p-np.roll(p,1,axis=0),axis=1)
        shape=2*np.sqrt(3)*twice/max(np.sum(sides*sides),1e-12)
        area_gate=min(1,twice/.07)
        coverage=float(w[list(ids)].sum())
        balance=float(w[list(ids)].min()/max(w[list(ids)].max(),1e-9))**.25
        fit=coverage*shape*area_gate*balance
        if best is None or fit>best[0]:best=(fit,ids,shape,coverage,twice/2)
    fit,ids,shape,cov,area=best
    return result('triangle-masses','arrangement',fit,[poly(pts[list(ids)],True)],
                  [{'element_id':s.regions[i]['id'],'measure':'anchor mass share','value':float(w[i]),'unit':'fraction'} for i in ids],
                  f'The selected three anchors account for {cov:.0%} of represented mass; triangular spread {shape:.2f}.',
                  ['Any three non-collinear points form a triangle. This reports their importance and spread, not proof of deliberate triangular composition.',
                   'Search examines at most the 12 strongest anchors.'],parameters={'area_diagonal_squared':area,'coverage':cov,'shape':shape})


def _reflect_score(mass,center,angle):
    h,w=mass.shape;yy,xx=np.mgrid[:h,:w]
    cx,cy=np.array(center)*[w-1,h-1]
    dx,dy=xx-cx,yy-cy;ca,sa=np.cos(angle),np.sin(angle)
    proj=dx*ca+dy*sa
    rx=cx+2*proj*ca-dx;ry=cy+2*proj*sa-dy
    ref=map_coordinates(mass,[ry,rx],order=1,mode='constant',cval=0)
    # Penalize mass whose reflection leaves the image. Full original mass stays in denominator.
    return float(2*np.minimum(mass,ref).sum()/max(mass.sum()+ref.sum()+np.maximum(mass-ref,0)[(rx<0)|(rx>w-1)|(ry<0)|(ry>h-1)].sum(),1e-9))


def symmetry_read(s):
    if s.summary['blank']:return result('bilateral-symmetry','symmetry',0,eligible=False,explanation='An empty field does not supply evidence of compositional symmetry.')
    h,w=s.mass.shape;small=cv2.resize(s.mass,(max(32,round(w*128/max(h,w))),max(32,round(h*128/max(h,w)))))
    shape=(max(32,round(w*192/max(h,w))),max(32,round(h*192/max(h,w))))
    tone=cv2.resize(s.tone,shape,interpolation=cv2.INTER_AREA)
    gradients=[]
    for sigma in [1.2,3.0]:
        gy,gx=np.gradient(gaussian_filter(tone,sigma))
        gradients.append((gx,gy))
    sh,sw=tone.shape;yy,xx=np.mgrid[:sh,:sw]
    centers=[[.5,.5],s.summary['mass_center']]
    def axis_score(c,a):
        cx,cy=np.array(c)*[sw-1,sh-1];dx,dy=xx-cx,yy-cy
        ca,sa=np.cos(a),np.sin(a);projection=dx*ca+dy*sa
        rx=cx+2*projection*ca-dx;ry=cy+2*projection*sa-dy
        outside=(rx<0)|(rx>sw-1)|(ry<0)|(ry>sh-1)
        scores=[];fractions=[]
        for gx,gy in gradients:
            normal=-sa*gx+ca*gy
            reflected=map_coordinates(normal,[ry,rx],order=1,mode='constant',cval=0)
            energy=float(np.sum(normal**2));total=float(np.sum(gx**2+gy**2))
            fraction=energy/max(total,1e-12)
            # A reflected normal derivative changes sign. Penalize unmatched
            # structure outside the frame and axes with no transverse evidence.
            denominator=energy+float(np.sum(reflected**2))+float(np.sum(normal[outside]**2))
            correlation=float(-2*np.sum(normal*reflected)/max(denominator,1e-12))
            scores.append(max(0,correlation)*min(1,fraction/.15) if energy>1e-5 else 0)
            fractions.append(fraction)
        return (float(np.mean(scores)),list(c),a,scores,fractions)
    candidates=[axis_score(c,a) for c in centers for a in np.linspace(0,np.pi,18,endpoint=False)]
    # Refine a bounded neighborhood of three coarse axes. Exact frame/mass
    # centers need not coincide with a real photographic symmetry axis.
    for _,c,a,_,_ in sorted(candidates,key=lambda x:-x[0])[:3]:
        for offset in np.linspace(-.06,.06,9):
            shifted=np.array(c)+offset*np.array([-np.sin(a),np.cos(a)])/s.metric_scale
            for angle_delta in [-np.pi/72,0,np.pi/72]:
                candidates.append(axis_score(shifted,a+angle_delta))
    fit,c,a,scores,fractions=max(candidates,key=lambda x:x[0]);v=np.array([np.cos(a),np.sin(a)])/s.metric_scale*.7
    geom=[line(np.array(c)-v,np.array(c)+v)]
    tonal_overlap=_reflect_score(small,c,a)
    extra={'tonal_mass_overlap':tonal_overlap,'scales_pixels':[1.2,3.0],
           'structural_agreement_by_scale':scores,'normal_gradient_energy_fraction':fractions,
           'axis_candidates':len(candidates),'decision_threshold':.45,'decision_basis':'structural agreement; not tonal mass overlap',
           'threshold_selection':'Previous CADB development cohort only; coarsely swept thresholds, chosen for under 5% development false-positive rate.'}
    try:
        import imgsym
        bgr=np.uint8(np.clip(s.rgb[...,::-1]*255,0,255))
        ends=[tuple(np.asarray(p)*[w-1,h-1]) for p in geom[0]['points']]
        extra['imgsym_hog']=float(imgsym.score_along_axis(bgr,ends,method='hog'))
        extra['imgsym_role']='texture symmetry along selected structural axis; supplementary, not fused'
    except (ValueError,RuntimeError,ImportError,cv2.error) as e:
        extra['imgsym_unavailable']=str(e)[:160]
    return result('bilateral-symmetry','symmetry',fit,geom,
                  [{'measure':'paired structural gradient agreement','value':fit,'unit':'support'},
                   {'measure':'tonal mass overlap (diagnostic only)','value':tonal_overlap,'unit':'fraction'}],
                  f'{fit:.0%} matching structural evidence across this axis; tonal overlap alone is not counted as symmetry.',
                  ['Requires opposite gradients normal to a reflected axis at two scales; similar broad light/dark bands are insufficient.',
                   'A symmetric subject can dominate. This does not establish semantic symmetry or artistic intent.',
                   'Bounded axis search: 36 coarse axes plus 81 local refinements; fit is not a category probability.'],
                  parameters={'center':c,'angle_radians':a,**extra})


def circle_read(s):
    if s.summary['blank']:return result('circular-organization','curve',0,eligible=False)
    contours=measure_contours(s.mass)
    best=None
    for p in contours:
        q=p*s.metric_scale
        A=np.c_[2*q,np.ones(len(q))];b=(q*q).sum(1)
        sol=np.linalg.lstsq(A,b,rcond=None)[0];center=sol[:2]
        rad2=sol[2]+np.dot(center,center)
        if rad2<=0:continue
        r=np.sqrt(rad2)
        if not .06<r<.65:continue
        dist=np.linalg.norm(q-center,axis=1)
        residual=float(np.sqrt(np.mean((dist-r)**2))/r)
        angles=np.arctan2(q[:,1]-center[1],q[:,0]-center[0]);bins=np.unique(np.floor((angles+np.pi)/(2*np.pi)*24).astype(int))
        coverage=min(1,len(bins)/24);scale_gate=min(1,r/.16)
        fit=float(np.exp(-residual*12)*coverage*scale_gate)
        if best is None or fit>best[0]:best=(fit,center,r,residual,coverage,p)
    if best is None:return result('circular-organization','curve',0,eligible=False,explanation='No sufficiently extended contour supports a circle fit.')
    fit,c,r,res,cov,p=best;t=np.linspace(0,2*np.pi,100)
    fitted=(c+r*np.c_[np.cos(t),np.sin(t)])/s.metric_scale
    return result('circular-organization','curve',fit,[poly(fitted,True),{'kind':'evidence-path','points':p.tolist()}],
                  [{'measure':'circle relative residual','value':res,'unit':'radius'},{'measure':'angular coverage','value':cov,'unit':'fraction'}],
                  f'A contour covers {cov:.0%} of angular sectors with {res:.1%} radius-relative fitting error.',
                  ['Detects a circular mass boundary. A circular object alone does not establish whole-image circular composition.'],
                  parameters={'center':(c/s.metric_scale).tolist(),'radius_diagonal':r})


def measure_contours(mass):
    from skimage.measure import find_contours
    h,w=mass.shape;out=[]
    for threshold in [.10,.20,.35]:
        for c in find_contours(mass,threshold):
            if len(c)<50:continue
            p=c[::max(1,len(c)//250),::-1]/[w-1,h-1]
            if np.linalg.norm(np.ptp(p,axis=0))<.2:continue
            out.append(p)
    return out[:60]


def spiral_read(s, null_samples=0, seed=173):
    # Fixed golden-growth logarithmic spirals; eight frame-anchored variants.
    # No free curve fitting and no quarter-circle approximation.
    if s.summary['blank'] or not s.edge.any():return result('golden-spiral','curve',0,eligible=False)
    h,w=s.edge.shape;yy,xx=np.nonzero(s.edge);edgepts=np.c_[xx/(w-1),yy/(h-1)]*s.metric_scale
    tree=cKDTree(edgepts);theta=np.linspace(-3*np.pi,0,200);r=.53*np.exp((2*np.log(PHI)/np.pi)*theta)
    best=None;variants=[]
    for cx in [1-1/PHI,1/PHI]:
        for cy in [1-1/PHI,1/PHI]:
            for sign in [-1,1]:
                phase=np.arctan2((.5-cy)*s.metric_scale[1],(.5-cx)*s.metric_scale[0])
                q=np.c_[cx*s.metric_scale[0]+r*np.cos(sign*theta+phase),cy*s.metric_scale[1]+r*np.sin(sign*theta+phase)]
                p=q/s.metric_scale;inside=((p>=0)&(p<=1)).all(1);q=q[inside];p=p[inside]
                if len(p)<80:continue
                d=tree.query(q)[0]
                # Weight outer, longer arc sections, preventing a tiny central blob from explaining the spiral.
                weights=np.r_[0,np.linalg.norm(np.diff(q,axis=0),axis=1)]
                variants.append((p,weights))
                support=float(np.average(np.exp(-.5*(d/.012)**2),weights=weights))
                edge_fraction=float(s.edge.mean())
                fit=support
                if best is None or fit>best[0]:best=(fit,p,cx,cy,sign,edge_fraction)
    if best is None:return result('golden-spiral','curve',0,eligible=False)
    fit,p,cx,cy,sign,frac=best
    out=result('golden-spiral','curve',fit,[poly(p)],
                  [{'measure':'length-weighted edge proximity','value':fit,'unit':'support'},
                   {'measure':'edge density','value':frac,'unit':'fraction'}],
                  f'{fit:.0%} edge-proximity support along the strongest of eight fixed golden spirals.',
                  ['Exploratory template alignment, not a detected spiral narrative or proof of golden-ratio construction.',
                   'Dense texture can support incidental curves. Shift controls test frame alignment, not artistic intent.'],
                  parameters={'center':[cx,cy],'handedness':sign,'growth':float(2*np.log(PHI)/np.pi),'variants':8})
    if null_samples:
        # Preserve the complete edge texture and repeat the eight-template selection at
        # randomized toroidal offsets. Periodic padding handles seams in the distance field.
        distance=distance_transform_edt(~np.tile(s.edge,(3,3)),sampling=s.metric_scale[::-1]/[h-1,w-1])[h:2*h,w:2*w]
        rng=np.random.default_rng(seed);scores=[]
        for shift in rng.uniform(0,1,(null_samples,2)):
            fits=[]
            for points,weights in variants:
                q=(points+shift)%1
                d=map_coordinates(distance,[q[:,1]*(h-1),q[:,0]*(w-1)],order=1,mode='wrap')
                fits.append(float(np.average(np.exp(-.5*(d/.012)**2),weights=weights)))
            scores.append(max(fits))
        tail=float((1+np.sum(np.array(scores)>=fit))/(null_samples+1))
        out['control']={'method':'toroidal edge-field shifts; repeat all eight golden-spiral variants',
            'samples':null_samples,'seed':seed,'mean_fit':float(np.mean(scores)),
            'excess_fit':float(fit-np.mean(scores)),'tail_fraction':tail,
            'family_max_tail_fraction':tail,
            'interpretation':'Frame-alignment control preserving texture; not a category probability.'}
        out['status']='supported' if fit>=.5 and tail<=.1 else ('ambiguous' if fit>=.45 else 'weak')
    return out


def convergence_read(s):
    lines=s.lines[:32];scale=s.metric_scale
    if len(lines)<3:return result('convergence','space',0,eligible=False,explanation='At least three long segments are required to assess convergence.')
    seg=np.array([a['points'] for a in lines])*scale;v=seg[:,1]-seg[:,0];length=np.linalg.norm(v,axis=1)
    unit=v/length[:,None];normal=np.c_[-unit[:,1],unit[:,0]];rhs=(normal*seg[:,0]).sum(1);mid=seg.mean(1)
    best=None
    for i,j in combinations(range(len(lines)),2):
        A=normal[[i,j]]
        if abs(np.linalg.det(A))<.2:continue
        p=np.linalg.solve(A,rhs[[i,j]])
        if ((p/scale < -2)|(p/scale>3)).any():continue
        distance=np.abs(normal@p-rhs)
        error=distance/np.maximum(np.linalg.norm(mid-p,axis=1),.05)
        support=np.exp(-.5*(error/.055)**2);good=support>.6
        if good.sum()<3:continue
        angle=np.array([l['angle'] for l in lines])[good]
        spread=1-abs(np.mean(np.exp(2j*angle)))
        fit=float(np.average(support,weights=length)*min(1,spread/.18))
        if best is None or fit>best[0]:best=(fit,p,good,support,spread)
    if best is None:return result('convergence','space',0,eligible=False,explanation='Segments do not establish a sufficiently diverse convergence family.')
    fit,p,good,support,spread=best;pn=p/scale
    geom=[point(pn)]+[line(np.mean(l['points'],axis=0),pn) for l,g in zip(lines,good) if g]
    return result('convergence','space',fit,geom,
                  [{'element_id':l['id'],'measure':'convergence support','value':float(v),'unit':'support'} for l,v,g in zip(lines,support,good) if g],
                  f'{sum(good)} segments support a common convergence location ({pn[0]:.2f}, {pn[1]:.2f}).',
                  ['A convergence point is not sufficient to establish one-point perspective. Painted rays and accidental junctions can also converge.',
                   'Search covers the 32 longest segments and points within two frame widths beyond the image.'],
                  parameters={'vanishing_point':pn.tolist(),'direction_diversity':spread})


def relational_reads(s):
    pts=s.points;weights=s.weights;scale=s.metric_scale;out=[]
    if len(pts)>1:
        center=np.average(pts,axis=0,weights=weights);base=float(np.linalg.norm((center-.5)*scale));ev=[]
        for i,(r,p,w) in enumerate(zip(s.regions,pts,weights)):
            if w>=.999:continue
            after=(center-w*p)/(1-w);delta=float(np.linalg.norm((after-.5)*scale)-base)
            ev.append({'element_id':r['id'],'measure':'imbalance increase if anchor omitted','value':delta,'unit':'image diagonal'})
        best=max(ev,key=lambda x:x['value']) if ev else None
        fit=min(1,max(0,best['value']/.10)) if best else 0
        out.append(result('counterbalanced','relations',fit,[point(center),line((.5,.5),center)],ev,
            f'{best["element_id"]} changes anchor-system imbalance by {best["value"]:+.3f} image diagonals when omitted.' if best else '',
            ['This is omission from the measured anchor system, not an inpainted removal from the image.'],parameters={'anchor_imbalance':base}))
    else:out.append(result('counterbalanced','relations',0,eligible=False,explanation='Counterweight analysis needs at least two anchors.'))
    flat=s.summary['flat_fraction']
    out.append(result('negative-space','relations',flat,explanation=f'{flat:.0%} of the Notan field lies below its flat-field threshold.',
                      evidence=[{'measure':'flat field fraction','value':flat,'unit':'fraction'}],
                      limits=['Flat field is not necessarily untouched background or intentional negative space.'],eligible=not s.summary['blank']))
    # Radial field organization: agreement of local gradient normals with a ray from mass center.
    if not s.summary['blank']:
        gy,gx=np.gradient(gaussian_filter(s.mass,3));mag=np.hypot(gx,gy)
        h,w=s.mass.shape;yy,xx=np.mgrid[:h,:w];c=s.summary['mass_center'];dx=xx-c[0]*(w-1);dy=yy-c[1]*(h-1)
        radial=(gx*dx+gy*dy)**2/(np.maximum(mag**2*(dx**2+dy**2),1e-12))
        alignment=float(np.sum(radial*mag)/max(mag.sum(),1e-12))
        fit=max(0,2*alignment-1)
        out.append(result('radial-organization','symmetry',fit,[point(c)],
            [{'measure':'radial gradient agreement','value':alignment,'unit':'mean squared cosine'}],
            f'Gradient alignment around the mass center is {alignment:.2f} (isotropic direction reference: 0.50).',
            ['A round isolated body can produce radial agreement; this does not establish a radial composition of multiple objects.']))
    else:out.append(result('radial-organization','symmetry',0,eligible=False,explanation='An empty field provides no radial evidence.'))
    return out


def quick_reads(s,include_placement=True):
    output=[]
    for id,pos in [('rule-of-thirds',[1/3,2/3]),('golden-grid',[1-1/PHI,1/PHI]),('rule-of-fifths',[.2,.4,.6,.8]),('centre-cross',[.5])]:
        if include_placement:output.append(grid_read(s,id,pos))
    for id,geom in templates(s.aspect).items():
        if id not in ['rule-of-thirds','golden-grid','rule-of-fifths','centre-cross','quadrant']:output.append(template_read(s,id,geom))
    output.extend([direction_read(s,'horizontal-organization',0),direction_read(s,'vertical-organization',np.pi/2),triangle_read(s)])
    return output


def evaluate(s, null_samples=64, seed=173):
    output=quick_reads(s)
    if null_samples and not s.summary['blank']:
        from .placement import evaluate_placements
        rng=np.random.default_rng(seed);subject_rng=np.random.default_rng(seed+100003);nulls={r['id']:[] for r in output}
        # Geometric null: preserve anchor weights/count and segment lengths, randomize placement/orientation.
        for _ in range(null_samples):
            n=copy.copy(s);n.regions=[{**r,'center':rng.uniform(.03,.97,2).tolist()} for r in s.regions];n.lines=[]
            for l in s.lines:
                # Preserve segment length and require the entire segment inside the frame.
                for attempt in range(1000):
                    ang=rng.uniform(0,np.pi);v=np.array([np.cos(ang),np.sin(ang)])*l['length']/s.metric_scale/2
                    if (np.abs(v)<=.5).all():break
                else:
                    ang=l['angle'];v=(np.array(l['points'][1])-l['points'][0])/2
                extent=np.minimum(np.abs(v),.5);center=rng.uniform(extent,1-extent)
                n.lines.append({**l,'points':[(center-v).tolist(),(center+v).tolist()],'angle':ang})
            for r in quick_reads(n,include_placement=False):nulls[r['id']].append(r['fit'])
        # The full search is repeated for each null. Family max percentiles expose template-selection optimism.
        for r in output:
            if r['family']=='placement':continue
            arr=np.array(nulls[r['id']]);same=[x['id'] for x in output if x['family']==r['family']]
            maximum=np.max([nulls[x] for x in same],axis=0)
            r['control']={'method':'randomized anchor placement and line pose; geometry only',
                'samples':null_samples,'seed':seed,'mean_fit':float(arr.mean()),
                'excess_fit':float(r['fit']-arr.mean()),
                'tail_fraction':float((1+np.sum(arr>=r['fit']))/(null_samples+1)),
                'family_max_tail_fraction':float((1+np.sum(maximum>=r['fit']))/(null_samples+1)),
                'interpretation':'Exploratory comparison under this geometric null, not a probability the label is true.'}
            if r['eligible']:
                r['status']='supported' if r['fit']>=.5 and r['control']['family_max_tail_fraction']<=.1 else ('ambiguous' if r['fit']>=.45 else 'weak')
        output=evaluate_placements(s,null_samples,seed)+[h for h in output if h['family']!='placement']
    from .rings import ring_read
    output.extend([symmetry_read(s),circle_read(s),ring_read(s),spiral_read(s,null_samples,seed),convergence_read(s),*relational_reads(s)])
    # Expose quadrant mass explicitly as a diagnostic, never a composition-class probability.
    h,w=s.mass.shape;tot=max(s.mass.sum(),1e-9)
    quadrants=[float(a.sum()/tot) for a in [s.mass[:h//2,:w//2],s.mass[:h//2,w//2:],s.mass[h//2:,:w//2],s.mass[h//2:,w//2:]]]
    output.append(result('quadrant','diagnostic',max(quadrants),templates(s.aspect)['quadrant'],
                         [{'measure':n,'value':v,'unit':'mass fraction'} for n,v in zip(['upper left','upper right','lower left','lower right'],quadrants)],
                         'Quadrants report how the field is distributed; they are not a style classification.',eligible=not s.summary['blank']))
    if len(output)!=len(DETECTORS) or {r['id'] for r in output}!=set(DETECTORS):
        raise RuntimeError('Detector output does not satisfy the registry contract.')
    return output
