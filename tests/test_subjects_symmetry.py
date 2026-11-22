import io,json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image,ImageDraw
from composition_reader.extract import load_image,extract
from composition_reader.hypotheses import grid_read,symmetry_read,evaluate
from composition_reader.subjects import regions_from_mask
from composition_reader.subject_model import checked_model

def test_competing_tonal_masses_do_not_dilute_a_selected_subject():
    # A small real foreground shape against a broad background; then deliberately
    # corrupt tonal-anchor weights to demonstrate placement's evidence separation.
    im=Image.new('RGB',(192,192),'white');ImageDraw.Draw(im).ellipse((80,80,112,112),fill='black')
    b=io.BytesIO();im.save(b,format='PNG');s=extract(load_image(b.getvalue())[0])
    assert s.subjects and len(s.placement_regions)==1
    original=grid_read(s,'centre-cross',[.5])['fit'];assert original>.95
    s.regions=[{'id':'background','center':[.05,.05],'mass_share':1000}]
    assert grid_read(s,'centre-cross',[.5])['fit']==original
    assert len(s.placement_points)==1

def test_selection_weights_exclusions_and_empty_evidence():
    field=np.zeros((160,160),np.float32);field[20:40,20:40]=.9;field[80:120,80:120]=.9
    selected=regions_from_mask(field)
    assert len(selected)==2 and sum(r['placement_weight'] for r in selected)==pytest.approx(1)
    assert selected[0]['placement_weight']/selected[1]['placement_weight']==pytest.approx(2)
    omitted=regions_from_mask(field,['s1'])
    assert not omitted[0]['included'] and omitted[1]['placement_weight']==1
    assert regions_from_mask(np.ones((100,100)))==[]
    assert regions_from_mask(field*.1)==[]

def test_reflection_requires_structure_not_broad_tonal_bands():
    a=np.ones((160,240,3))*.8;a[75:]=.12
    horizon=symmetry_read(extract(a))
    assert horizon['fit']<.15 and horizon['status']=='weak'
    rng=np.random.default_rng(91)
    left=rng.uniform(.15,.85,(160,120,3))
    # Large mirrored blocks plus detail, with independently randomized counterpart.
    mirrored=np.concatenate([left,left[:,::-1]],axis=1)
    unrelated=np.concatenate([left,rng.uniform(.15,.85,left.shape)],axis=1)
    same=symmetry_read(extract(mirrored));different=symmetry_read(extract(unrelated))
    assert same['fit']>.85
    assert different['fit']<.4
    assert same['parameters']['tonal_mass_overlap']>.8

def test_subject_null_randomizes_subjects_and_does_not_reuse_observed_positions():
    im=Image.new('RGB',(192,192),'white');ImageDraw.Draw(im).ellipse((48,48,80,80),fill='black')
    b=io.BytesIO();im.save(b,format='PNG');s=extract(load_image(b.getvalue())[0])
    before=json.dumps(s.subjects)
    r=next(h for h in evaluate(s,32) if h['id']=='rule-of-thirds')
    assert r['fit']>.9 and r['control']['mean_fit']<.3
    assert json.dumps(s.subjects)==before
    assert r['control']['family_max_tail_fraction']>=r['control']['tail_fraction']

def test_manual_anchors_take_priority_and_model_is_pinned(monkeypatch,tmp_path):
    a=np.ones((160,160,3));a[30:80,30:80]=0
    s=extract(a,anchors=[[.5,.5]])
    assert s.subject_metadata['method']=='user-anchors'
    assert grid_read(s,'centre-cross',[.5])['fit']==1
    monkeypatch.setenv('COMPOSITION_MODELS',str(tmp_path))
    with pytest.raises(RuntimeError,match='not installed'):checked_model()
    (tmp_path/'u2netp.onnx').write_bytes(b'not the verified model')
    with pytest.raises(RuntimeError,match='checksum'):checked_model()
