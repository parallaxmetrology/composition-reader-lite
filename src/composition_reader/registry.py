"""Stable detector vocabulary and evidence contract shared by engine and clients.

A new named interpretation declares its evidence requirements and control scope here;
the scene representation and report envelope do not change when the vocabulary grows.
"""
from dataclasses import dataclass, asdict

NAMES={
 'rule-of-thirds':'Rule of thirds','golden-grid':'Golden grid','rule-of-fifths':'Rule of fifths',
 'centre-cross':'Centered placement','quadrant':'Quadrant distribution',
 'baroque-diagonal':'Descending diagonal','sinister-diagonal':'Ascending diagonal',
 'diagonal-method':'Corner diagonal method','golden-triangle':'Diagonal and perpendiculars',
 'rabatment':'Rabatment','reciprocal-lines':'Reciprocal lines','armature-14':'14-line armature',
 'triangle-masses':'Triangular mass arrangement','bilateral-symmetry':'Bilateral structural symmetry',
 'horizontal-organization':'Horizontal organization','vertical-organization':'Vertical organization',
 'radial-organization':'Radial organization','circular-organization':'Circular contour organization','concentric-rings':'Concentric rings',
 'golden-spiral':'Golden spiral alignment','convergence':'Line convergence',
 'counterbalanced':'Counterweighted arrangement','negative-space':'Open field / negative space',
}

@dataclass(frozen=True)
class DetectorSpec:
    id: str
    name: str
    family: str
    requires: tuple[str,...]
    control: str
    version: str = "1"

def _spec(id):
    if id in {"rule-of-thirds","golden-grid","rule-of-fifths","centre-cross"}:
        family,requires,control="placement",("subject-anchors",),"frame-contact-conditioned-rank-family"
    elif id in {"baroque-diagonal","sinister-diagonal","diagonal-method","golden-triangle","rabatment","reciprocal-lines","armature-14"}:
        family,requires,control="armature",("segments",),"geometric-family-max"
    elif id in {"horizontal-organization","vertical-organization"}:
        family,requires,control="direction",("segments",),"geometric-family-max"
    elif id=="triangle-masses":
        family,requires,control="arrangement",("anchors",),"geometric-family-max"
    elif id=="bilateral-symmetry":
        family,requires,control="symmetry",("paired-structural-gradients",),"descriptive-with-perturbation-check"
    elif id=="radial-organization":
        family,requires,control="symmetry",("mass-field",),"descriptive"
    elif id=="golden-spiral":
        family,requires,control="curve",("edge-field",),"toroidal-shift-best-of-eight"
    elif id=="concentric-rings":
        family,requires,control="curve",("edge-field",),"descriptive-with-perturbation-check"
    elif id=="circular-organization":
        family,requires,control="curve",("mass-contours",),"descriptive"
    elif id=="convergence":
        family,requires,control="space",("segments",),"descriptive"
    elif id=="counterbalanced":
        family,requires,control="relations",("anchors",),"analytic-omission"
    elif id=="negative-space":
        family,requires,control="relations",("mass-field",),"descriptive"
    elif id=="quadrant":
        family,requires,control="diagnostic",("mass-field",),"descriptive"
    else:raise ValueError(f"Missing detector contract: {id}")
    if id=='rule-of-thirds':
        requires=("face-landmarks-or-visible-horizon-or-subject-anchors",);control="persistent-object-ownership-with-rigid-group-rank-family"
    return DetectorSpec(id,NAMES[id],family,requires,control,"9" if id=="rule-of-thirds" else "4" if family=="placement" else "2" if id=="bilateral-symmetry" else "1")

DETECTORS={id:_spec(id) for id in NAMES}

def contracts():
    return [asdict(spec) for spec in DETECTORS.values()]
