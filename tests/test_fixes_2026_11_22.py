"""Regression checks for the 22 Nov 2026 reading fixes (known-answer pictures)."""
import io, math
import numpy as np
from PIL import Image, ImageDraw
from composition_reader.extract import load_image, extract
from composition_reader.hypotheses import symmetry_read, relational_reads, quick_reads
from composition_reader.engine import held_back_reason, near_misses
from composition_reader.subjects import sky_like
from composition_reader.model import Scene

W, H = 360, 240


def scene(draw, border='raw'):
    im = Image.new('RGB', (W, H), (235,)*3); draw(ImageDraw.Draw(im))
    out = io.BytesIO(); im.save(out, format='PNG')
    return extract(load_image(out.getvalue())[0], border=border)


def by_id(rows, id): return next(r for r in rows if r['id'] == id)


def geometry(points, weights):
    m = np.zeros((200, 300)); m[80:120, 130:170] = 1
    return Scene(np.repeat(m[..., None], 3, axis=2), m, m, m, m, m > 0, np.zeros_like(m),
                 [{'id': f'r{i}', 'center': list(p), 'mass_share': w} for i, (p, w) in enumerate(zip(points, weights))],
                 [], {'blank': False, 'mass_center': [.5, .5], 'flat_fraction': .9})


def test_radial_counts_radiating_spokes():
    def spokes(d):
        for k in range(16):
            a = k*math.pi/8; d.line([180+16*math.cos(a), 120+16*math.sin(a), 180+300*math.cos(a), 120+300*math.sin(a)], fill=(20,)*3, width=4)
    r = by_id(relational_reads(scene(spokes)), 'radial-organization')
    assert r['parameters']['kind'] == 'radiating' and r['fit'] > .6


def test_radial_needs_reach_and_surround():
    blob = by_id(relational_reads(scene(lambda d: d.ellipse([162, 102, 198, 138], fill=(30,)*3))), 'radial-organization')
    line = by_id(relational_reads(scene(lambda d: d.line([0, 0, W, H], fill=(20,)*3, width=6))), 'radial-organization')
    assert blob['fit'] < .45 and line['fit'] < .45


def test_symmetry_separates_off_centre_subject_from_frame_symmetry():
    def off(d):
        d.polygon([(90, 50), (45, 110), (135, 110)], fill=(40,)*3); d.rectangle([55, 110, 125, 200], fill=(90,)*3)
    r = symmetry_read(scene(off))
    assert r['fit'] < .45
    assert r['parameters']['subject_symmetry']['structural_agreement'] > .9


def test_symmetry_centred_pair_keeps_full_credit():
    r = symmetry_read(scene(lambda d: [d.ellipse([60, 90, 120, 150], fill=(0,)*3), d.ellipse([240, 90, 300, 150], fill=(0,)*3)]))
    assert r['fit'] > .9 and r['parameters']['axis_frame_coverage'] > .9


def test_counterweight_is_asymmetric_balance():
    lopsided = by_id(relational_reads(geometry([[.15, .3], [.25, .7]], [.7, .3])), 'counterbalanced')
    steelyard = by_id(relational_reads(geometry([[.3, .5], [.95, .5]], [.75, .25])), 'counterbalanced')
    twins = by_id(relational_reads(geometry([[.2, .5], [.8, .5]], [.5, .5])), 'counterbalanced')
    centred = by_id(relational_reads(geometry([[.5, .52], [.5, .4]], [.9, .1])), 'counterbalanced')
    assert lopsided['fit'] < .2 and twins['fit'] < .05 and centred['fit'] < .3
    assert steelyard['fit'] > .5 and steelyard['parameters']['counter_mass'] == 'r1'


def test_protected_border_sets_aside_mat():
    def matted(d):
        d.rectangle([0, 0, W-1, H-1], outline=(10,)*3, width=10); d.ellipse([220, 60, 260, 100], fill=(30,)*3)
    raw, prot = scene(matted), scene(matted, 'protected')
    assert prot.summary['border_regions_set_aside'] and not raw.summary['border_regions_set_aside']


def test_golden_triangle_tests_both_mirror_forms():
    from composition_reader.geometry import templates
    t = templates(1.5)
    for key, variant in [('golden-triangle', 'descending diagonal'), ('golden-triangle-ascending', 'ascending diagonal')]:
        lines = []
        for i, g in enumerate(t[key]):
            a, b = np.array(g['points'][0]), np.array(g['points'][1]); a, b = a+(b-a)*.15, a+(b-a)*.85
            v = (b-a)*[1.5, 1]
            lines.append({'id': f'l{i}', 'points': [a.tolist(), b.tolist()], 'angle': float(np.arctan2(v[1], v[0]) % np.pi), 'length': float(np.linalg.norm(v)/np.hypot(1.5, 1))})
        s = geometry([[.08, .5], [.92, .5]], [.5, .5]); s.lines = lines
        s.rgb = np.zeros((200, 300, 3)); s.mass = np.zeros((200, 300))
        r = by_id(quick_reads(s, include_placement=False), 'golden-triangle')
        assert r['fit'] > .9 and r['parameters']['variant'] == variant


def test_sky_like_component_is_set_aside_but_bottom_reaching_portrait_is_not():
    lum = np.full((100, 150), .3); lum[:30, 20:130] = .9
    sky = {'bbox': [.13, 0, .87, .3], 'center': [.5, .15], 'contour': [[.13, 0], [.87, 0], [.87, .3], [.13, .3]]}
    portrait = {'bbox': [.2, 0, .8, 1], 'center': [.5, .5], 'contour': [[.2, 0], [.8, 0], [.8, 1], [.2, 1]]}
    assert sky_like(sky, lum) and not sky_like(portrait, lum)


def test_near_miss_reasons_name_the_rule():
    tail = {'id': 'x', 'name': 'X', 'family': 'placement', 'eligible': True, 'status': 'ambiguous', 'fit': .95, 'parameters': {}, 'stability': None,
            'control': {'decision_tail_key': 'adjusted_tail_fraction', 'adjusted_tail_fraction': .11, 'minimum_excess_fit': .1, 'excess_fit': .3}}
    low = {**tail, 'id': 'y', 'fit': .4996, 'control': {'family_max_tail_fraction': .04}}
    contact = {**tail, 'id': 'z', 'status': 'sensitive', 'stability': {'max_fit_change': .05, 'max_contact_weight_change': .3}}
    assert '11%' in held_back_reason(tail) and '0.499' in held_back_reason(low) and 'pinned by the frame' in held_back_reason(contact)
    assert [r['hypothesis_id'] for r in near_misses([tail, low, {**tail, 'id': 'w', 'status': 'supported'}], set())] == ['x', 'y']
