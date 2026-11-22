import numpy as np
import pytest
from test_placement_v3 import subject_scene
from composition_reader.placement import randomize_subjects,evaluate_placements,calibrate

def test_cropped_group_cannot_gain_evidence_from_its_box_midpoint():
    s=subject_scene((.4481409,.6612903),(.8964844,.6783626))
    s.subjects[0]['bbox']=[0.,.3216374,.8964844,1.]
    h=next(h for h in evaluate_placements(s,64) if h['id']=='rule-of-thirds')
    assert h['fit']>.8  # Strong alignment even in this square geometric fixture.
    assert h['status']=='ambiguous'
    assert h['control']['tail_fraction']==1
    assert h['control']['excess_fit']==pytest.approx(0,abs=1e-12)
    assert h['parameters']['constrained_axis_weight']==[1,1]

def test_a_bottom_contact_pins_only_that_axis():
    s=subject_scene((.3,.7),(.2,.6));s.subjects[0]['bbox']=[.2,.4,.4,1.]
    rng=np.random.default_rng(83);rows=[randomize_subjects(s,rng)[0] for _ in range(150)]
    assert np.ptp([r['center'][0] for r in rows])>.6
    assert all(r['center'][1]==.7 and r['bbox'][3]==1 for r in rows)
    assert all(0<=r['bbox'][0]<r['bbox'][2]<=1 for r in rows)

def test_translations_preserve_anchor_offset_instead_of_recentering_the_box():
    s=subject_scene((.45,.47));s.subjects[0]['bbox']=[.2,.3,.6,.8];rng=np.random.default_rng(91)
    for _ in range(80):
        r=randomize_subjects(s,rng)[0];p=np.array(r['center']);b=np.array(r['bbox'])
        assert p-b[:2]==pytest.approx([.25,.17])
        assert b[2:]-p==pytest.approx([.15,.33])
        assert np.all(b>=0) and np.all(b<=1)

def test_zero_area_manual_anchor_does_not_imply_a_cropped_object():
    s=subject_scene((0,.5),(0,0));rng=np.random.default_rng(14)
    assert np.ptp([randomize_subjects(s,rng)[0]['center'][0] for _ in range(80)])>.7

def test_low_tail_without_material_excess_is_not_supported():
    h={'id':'a','family':'placement','fit':1.,'eligible':True,'parameters':{},'explanation':''}
    calibrate([h],{'a':[.99]*256},[])
    assert h['control']['adjusted_tail_fraction']<.01
    assert h['status']=='ambiguous'

def test_changed_frame_constraints_are_sensitive_even_when_fit_is_unchanged(monkeypatch):
    from composition_reader import engine
    original=subject_scene();changed=subject_scene(span=(1.,.2))
    h=next(h for h in evaluate_placements(original,64) if h['id']=='centre-cross')
    assert h['status']=='supported'
    monkeypatch.setattr(engine,'extract',lambda *args,**kwargs:changed)
    engine.apply_stability(original.rgb,{},[h])
    assert h['stability']['max_fit_change']==0
    assert h['stability']['max_contact_weight_change']==1
    assert h['status']=='sensitive'
