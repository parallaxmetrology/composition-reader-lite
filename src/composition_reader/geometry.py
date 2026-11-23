"""Frame geometry. Stored coordinates are x-right/y-down fractions; distances use aspect."""
import numpy as np

PHI = (1 + 5**.5) / 2

def line(a,b): return {'kind':'line','points':[list(a),list(b)]}
def point(p): return {'kind':'point','point':list(p)}
def poly(points,closed=False): return {'kind':'polyline','points':np.asarray(points).tolist(),'closed':closed}

def grid(positions):
    return [line((x,0),(x,1)) for x in positions]+[line((0,y),(1,y)) for y in positions]

def foot(p,a,b,scale):
    p,a,b=np.asarray(p)*scale,np.asarray(a)*scale,np.asarray(b)*scale
    v=b-a
    return (a + v*np.dot(p-a,v)/max(np.dot(v,v),1e-12))/scale

def templates(aspect):
    scale=np.array([aspect,1.0])/np.hypot(aspect,1)
    diag=[line((0,0),(1,1)),line((0,1),(1,0))]
    recip=[]
    for a,b,cs in [((0,0),(1,1),[(0,1),(1,0)]),((0,1),(1,0),[(0,0),(1,1)])]:
        recip.extend(line(c,foot(c,a,b,scale)) for c in cs)
    # 2 diagonals + 8 corner/opposite-midpoint lines + 4 midpoint diamond edges.
    # No median axes or reciprocal perpendiculars belong to this construction.
    # https://gridmakerpro.com/grids/advanced-composition/armature-14-line/
    mids=[]
    for a in [(0,0),(1,0),(1,1),(0,1)]:
        mids += [line(a,(1-a[0],.5)),line(a,(.5,1-a[1]))]
    diamond_points=[(.5,0),(0,.5),(.5,1),(1,.5)]
    diamond=[line(p,diamond_points[(i+1)%4]) for i,p in enumerate(diamond_points)]
    r=1/aspect if aspect>=1 else aspect
    rabatt=([line((r,0),(r,1)),line((1-r,0),(1-r,1))] if aspect>=1 else
            [line((0,r),(1,r)),line((0,1-r),(1,1-r))])
    # Westhoff diagonals are 45 degrees in physical pixel space, not normalized space.
    d=[]
    for x,y in [(0,0),(1,0),(0,1),(1,1)]:
        dx=min(1,1/aspect);dy=min(1,aspect)
        d.append(line((x,y),(x+(1 if x==0 else -1)*dx,y+(1 if y==0 else -1)*dy)))
    return {
        'rule-of-thirds':grid([1/3,2/3]),'golden-grid':grid([1-1/PHI,1/PHI]),
        'rule-of-fifths':grid([.2,.4,.6,.8]),'centre-cross':grid([.5]),
        'quadrant':grid([.5]),'baroque-diagonal':[diag[0]],'sinister-diagonal':[diag[1]],
        'diagonal-method':d,'golden-triangle':[diag[0],*recip[:2]],
        # Mirror form on the other diagonal; the detector reports the better of the two.
        'golden-triangle-ascending':[diag[1],*recip[2:]],
        'rabatment':rabatt,'reciprocal-lines':[*diag,*recip],
        'armature-14':[*diag,*mids,*diamond],
    }

def segment_dist(points, segments, scale):
    p=np.asarray(points).reshape(-1,2)*scale
    if not len(segments): return np.full((len(p),0),np.inf)
    seg=np.array([s['points'] for s in segments])*scale
    a=seg[:,0];v=seg[:,1]-a
    t=np.clip(np.einsum('nkd,kd->nk',p[:,None]-a,v)/(np.sum(v*v,axis=1)+1e-12),0,1)
    return np.linalg.norm(p[:,None]-(a[None]+t[:,:,None]*v[None]),axis=2)

def angle_distance(a,b):
    return np.abs((np.asarray(a)-np.asarray(b)+np.pi/2)%np.pi-np.pi/2)
