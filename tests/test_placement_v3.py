import copy
import numpy as np
import pytest
from test_measurements import geometry_scene,picture
from composition_reader.extract import extract,load_image
from composition_reader.placement import randomize_subjects,evaluate_placements,features,POSITIONS,calibrate
from composition_reader.hypotheses import evaluate

def subject_scene(center=(.5,.5),span=(.2,.2)):
    s=geometry_scene([center]);p=np.array(center);half=np.array(span)/2
    s.subjects=[{'id':'s1','center':list(center),'bbox':np.r_[p-half,p+half].tolist(),'placement_weight':1.,'included':True}]
    return s

def test_footprint_is_preserved_and_cannot_escape_frame():
    s=subject_scene(span=(.8,1.));rng=np.random.default_rng(4)
    for _ in range(200):
        r=randomize_subjects(s,rng)[0];box=np.array(r['bbox'])
        assert np.all(box>=0) and np.all(box<=1)
        assert box[2:]-box[:2]==pytest.approx([.8,1.])
        assert r['center'][1]==.5
    assert s.subjects[0]['center']==[.5,.5]

def test_filling_the_frame_does_not_establish_informative_centering():
    small=next(h for h in evaluate_placements(subject_scene(),64) if h['id']=='centre-cross')
    full=next(h for h in evaluate_placements(subject_scene(span=(1.,1.)),64) if h['id']=='centre-cross')
    assert small['fit']==full['fit']==1
    assert small['status']=='supported'
    assert full['status']=='ambiguous' and not full['control']['two_axis_informative']
    assert full['control']['axis_excess_fit']==pytest.approx([0,0])
    assert small['control']['samples']>=256

def test_thirds_separates_intersections_from_line_alignment():
    tall=features(subject_scene((1/3,.5),(.12,.7)),'rule-of-thirds',[1/3,2/3])
    compact=features(subject_scene((1/3,.5),(.12,.12)),'rule-of-thirds',[1/3,2/3])
    assert tall['fit']>.99 and tall['parameters']['point_intersection_fit']<.05
    assert tall['evidence'][0]['alignment_mode']=='vertical-grid-line'
    assert compact['fit']==pytest.approx(.85)
    assert compact['parameters']['point_intersection_fit']<.05
    assert compact['evidence'][0]['alignment_mode']=='vertical-grid-line'
    wide=features(subject_scene((.5,2/3),(.7,.12)),'rule-of-thirds',[1/3,2/3])
    assert wide['fit']>.99 and wide['evidence'][0]['alignment_mode']=='horizontal-grid-line'

def test_numeric_null_path_matches_exposed_geometry():
    for center,span in [((.5,.5),(.8,.9)),((1/3,.5),(.1,.8)),((.2,.4),(.2,.3))]:
        s=subject_scene(center,span)
        for id,pos in POSITIONS.items():
            a=features(s,id,pos);b=features(s,id,pos,measure_only=True)
            assert a['fit']==pytest.approx(b['fit'],abs=1e-12)
            if id=='centre-cross':assert a['parameters']['axis_fit']==pytest.approx(b['axis_fit'])

def test_rank_family_calibration_has_conservative_ties_and_search_correction():
    readings=[{'id':id,'family':'placement','fit':v,'eligible':True,'parameters':{},'explanation':''} for id,v in zip(['a','b','c'],[.8,.7,.6])]
    nulls={'a':[.8,.1,.2,.3],'b':[.1,.7,.1,.9],'c':[.6,.1,.5,.2]}
    # Independent pairwise enumeration of augmented marginal upper-tail ranks.
    values=np.array([[h['fit'],*nulls[h['id']]] for h in readings]);tails=np.array([[sum(row>=x)/len(row) for x in row] for row in values])
    expected=[sum(tails.min(axis=0)<=p)/5 for p in tails[:,0]]
    calibrate(readings,nulls,[])
    for h,p in zip(readings,expected):
        assert h['control']['adjusted_tail_fraction']==pytest.approx(p)
        assert h['control']['adjusted_tail_fraction']>=h['control']['tail_fraction']

def test_real_extraction_uses_identical_controls_in_full_engine_and_placement_path():
    scene=extract(load_image(picture(((1/3,1/3),)))[0]);a={h['id']:h for h in evaluate_placements(scene,64)}
    b={h['id']:h for h in evaluate(scene,64)}
    for id,h in a.items():
        assert h['fit']==b[id]['fit']
        assert h['control']==b[id]['control'] and h['status']==b[id]['status']

def test_a_line_match_can_be_described_without_claiming_it_survived_controls():
    h=next(h for h in evaluate_placements(subject_scene((.35,.5),(.12,.12)),64) if h['id']=='rule-of-thirds')
    assert h['status']=='descriptive'
    assert h['control']['adjusted_tail_fraction']>.1
    assert 'not ruled out accidental alignment' in h['explanation']
    assert h['parameters']['point_intersection_fit']<.05
