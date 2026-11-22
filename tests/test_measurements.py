"""Known-answer checks of actual pixels and measured geometry, not classifier accuracy."""
import copy
import io
import json
import hashlib
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw
from composition_reader.engine import analyze, compare_reports, frame_context
from composition_reader.extract import load_image, extract
from composition_reader.hypotheses import evaluate, grid_read, triangle_read, symmetry_read, convergence_read, spiral_read
from composition_reader.model import Scene


def picture(points=((.5,.5),), size=(256,256), radius=20):
    im=Image.new('RGB',size,'white');d=ImageDraw.Draw(im)
    for x,y in points:
        x*=size[0]-1;y*=size[1]-1
        d.ellipse((x-radius,y-radius,x+radius,y+radius),fill='black')
    out=io.BytesIO();im.save(out,format='PNG');return out.getvalue()


def geometry_scene(points, lines=()):
    """Independent exact geometry isolates scorer behavior from extraction errors."""
    m=np.zeros((200,200));m[70:130,70:130]=1
    return Scene(np.repeat(m[...,None],3,axis=2),m,m,m,m,m>0,np.zeros_like(m),
        [{'id':str(i),'center':list(p),'mass_share':1/len(points)} for i,p in enumerate(points)],
        list(lines),{'blank':False,'mass_center':[.5,.5],'flat_fraction':.91})


@pytest.mark.parametrize('shade',[0,128,255])
def test_blank_pixels_never_claim_composition(shade):
    out=io.BytesIO();Image.new('RGB',(128,128),(shade,)*3).save(out,format='PNG')
    r,fields=analyze(out.getvalue(),stability=True,null_samples=8)
    assert r['summary']['blank'] and not r['regions'] and not r['lines']
    assert all(h['status'] in ['insufficient','weak'] for h in r['hypotheses'])
    assert set(fields)=={'image','mass','notan','chroma','edges','subjects'}
    json.dumps(r,allow_nan=False)


def test_subthreshold_noise_is_not_contrast_stretched_into_shapes():
    rng=np.random.default_rng(75);rgb=np.full((192,192,3),.5)+rng.normal(0,.0002,(192,192,3))
    s=extract(rgb)
    assert s.summary['blank'] and s.mass.max()==0 and not s.regions


def test_actual_pixels_center_vs_thirds():
    centered=extract(load_image(picture())[0])
    thirds=extract(load_image(picture(((1/3,1/3),)))[0])
    assert len(centered.regions)==len(thirds.regions)==1
    assert grid_read(centered,'centre-cross',[.5])['fit']>.98
    assert grid_read(centered,'rule-of-thirds',[1/3,2/3])['fit']<.05
    assert grid_read(thirds,'rule-of-thirds',[1/3,2/3])['fit']>.98
    assert grid_read(thirds,'centre-cross',[.5])['fit']<.05


def test_triangle_requires_spread_and_important_anchors():
    triangular=geometry_scene([[.2,.75],[.8,.75],[.5,.23]])
    collinear=geometry_scene([[.2,.5],[.5,.5],[.8,.5]])
    assert triangle_read(triangular)['fit']>.95
    assert triangle_read(collinear)['fit']<1e-8
    assert not triangle_read(geometry_scene([[.2,.5],[.8,.5]]))['eligible']


def test_randomization_is_reproducible_and_accounts_for_template_search():
    s=geometry_scene([[1/3,1/3],[2/3,2/3]])
    a=evaluate(s,32);b=evaluate(s,32)
    for x,y in zip(a,b):
        assert x['fit']==y['fit'] and x['control']==y['control']
        if x['control']:
            assert x['control']['family_max_tail_fraction']>=x['control']['tail_fraction']
    assert next(x for x in a if x['id']=='rule-of-thirds')['status']=='supported'


def test_symmetry_uses_real_imgsym_and_detects_bilateral_mass():
    s=extract(load_image(picture(((.25,.5),(.75,.5))))[0])
    result=symmetry_read(s)
    assert result['fit']>.93
    assert np.isfinite(result['parameters']['imgsym_hog'])
    assert 'imgsym_unavailable' not in result['parameters']


def test_convergence_distinguishes_parallel_lines():
    def line(a,b,i):
        v=np.array(b)-a
        return {'id':str(i),'points':[a,b],'angle':float(np.arctan2(v[1],v[0])%np.pi),'length':float(np.linalg.norm(v)/2**.5)}
    converging=[line([.5,.25],p,i) for i,p in enumerate([[.1,.9],[.5,.9],[.9,.9]])]
    parallel=[line([x,.2],[x,.9],i) for i,x in enumerate([.2,.5,.8])]
    assert convergence_read(geometry_scene([],converging))['fit']>.99
    assert not convergence_read(geometry_scene([],parallel))['eligible']


def test_dense_edges_do_not_pass_spiral_shift_control():
    s=geometry_scene([[.5,.5]])
    s.edge=np.ones_like(s.mass,dtype=bool)
    r=spiral_read(s,32)
    assert r['fit']>.95
    assert r['status']!='supported' and r['control']['tail_fraction']>.9


def test_corrections_change_anchors_not_underlying_field():
    rgb,_=load_image(picture(((.3,.3),(.7,.7))))
    automatic=extract(rgb);manual=extract(rgb,anchors=[[.5,.5]])
    assert manual.regions[0]['source']=='user'
    assert np.array_equal(automatic.mass,manual.mass)
    omitted=extract(rgb,excluded=[automatic.regions[0]['id']])
    assert len(omitted.regions)==len(automatic.regions)-1
    with pytest.raises(ValueError):extract(rgb,anchors=[[1.2,.5]])


def test_crop_metadata_comparison_and_reproducible_json():
    data=picture(((.3,.3),(.7,.7)))
    a,_=analyze(data,null_samples=0,stability=False)
    b,fields=analyze(data,crop=[.1,.1,.9,.8],null_samples=0,stability=False)
    assert b['source']['sha256']==a['source']['sha256']
    assert b['source']['analysis_size']==[204,179]
    assert Image.open(io.BytesIO(fields['image'])).size==(204,179)
    assert len(compare_reports(a,b)['warnings'])>=2
    json.dumps(b,allow_nan=False)
    with pytest.raises(ValueError):analyze(data,crop=[.1,.1,.1,.9])


def test_alpha_and_exif_are_applied_before_geometry():
    out=io.BytesIO();Image.new('RGBA',(100,60),(10,20,30,0)).save(out,format='PNG')
    rgb,meta=load_image(out.getvalue());assert np.all(rgb==1) and meta['alpha_background']=='white'
    im=Image.new('RGB',(100,60),'red');exif=Image.Exif();exif[274]=6
    out=io.BytesIO();im.save(out,format='JPEG',exif=exif)
    rgb,meta=load_image(out.getvalue());assert meta['upright_size']==[60,100] and rgb.shape[:2]==(100,60)


def test_bad_images_and_invalid_parameters_are_rejected():
    for data in [b'not an image',picture(size=(10,10))]:
        with pytest.raises(ValueError):load_image(data)
    for kw in [{'border':'unknown'},{'null_samples':-1},{'null_samples':257}]:
        with pytest.raises(ValueError):analyze(picture(),**kw)




def test_copied_sources_remain_byte_identical():
    root=Path(__file__).resolve().parents[1]
    for source in json.loads((root/'vendor/SOURCES.json').read_text()):
        assert hashlib.sha256((root/source['path']).read_bytes()).hexdigest()==source['sha256']


def test_armature_matches_reference_intersections_and_aspect_invariance():
    from composition_reader.geometry import templates
    from itertools import combinations
    lines=templates(1.5)['armature-14']
    assert len(lines)==14 and lines==templates(.6)['armature-14']
    intersections=set()
    for a,b in combinations(lines,2):
        p,q=np.array(a['points']),np.array(b['points'])
        u,v=p[1]-p[0],q[1]-q[0];matrix=np.column_stack([u,-v])
        if abs(np.linalg.det(matrix))<1e-10:continue
        t,s=np.linalg.solve(matrix,q[0]-p[0])
        if -1e-10<=t<=1+1e-10 and -1e-10<=s<=1+1e-10:
            intersections.add(tuple(np.round(p[0]+t*u,8)))
    assert len(intersections)==37
    assert (round(1/3,8),round(1/3,8)) in intersections


def test_perpendiculars_and_corner_diagonals_preserve_pixel_angles():
    from composition_reader.geometry import templates
    for aspect in [.5,1,2]:
        t=templates(aspect);scale=np.array([aspect,1])
        a=np.array(t['golden-triangle'][0]['points'])*scale;v=a[1]-a[0]
        for line in t['golden-triangle'][1:]:
            p=np.array(line['points'])*scale
            assert abs(np.dot(v,p[1]-p[0]))<1e-12
        for line in t['diagonal-method']:
            p=np.array(line['points'])*scale
            assert abs(abs(p[1,0]-p[0,0])-abs(p[1,1]-p[0,1]))<1e-12


