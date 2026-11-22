"""Known-answer tests for the concentric ring read."""
import io
import numpy as np
import cv2
from PIL import Image
from composition_reader.extract import load_image, extract
from composition_reader.rings import ring_read

DIAG = np.hypot(512, 512)


def scene(circles, size=512, noise=0.0, seed=0):
    im = np.full((size, size, 3), 235, np.uint8)
    for (cx, cy), r in circles:
        cv2.circle(im, (cx, cy), r, (30, 30, 30), 4)
    if noise:
        im = np.clip(im + np.random.default_rng(seed).normal(0, noise, im.shape), 0, 255).astype(np.uint8)
    b = io.BytesIO(); Image.fromarray(im).save(b, 'PNG')
    rgb, src = load_image(b.getvalue()); src['name'] = 't'
    return extract(rgb, src, 'raw', None, None)


def test_four_concentric_rings_are_counted_at_their_radii():
    radii = [50, 95, 140, 190]
    r = ring_read(scene([((256, 256), x) for x in radii]))
    assert r['parameters']['ring_count'] == 4 and r['status'] == 'descriptive'
    found = r['parameters']['ring_radii_diagonal']
    for want, got in zip(radii, found):
        assert abs(want / DIAG - got) < .02


def test_one_ring_is_not_a_ring_system():
    r = ring_read(scene([((256, 256), 90)]))
    assert r['parameters']['ring_count'] == 1 and r['status'] == 'weak' and r['fit'] < .5


def test_rings_around_an_off_frame_centre_are_found_and_the_centre_reported():
    r = ring_read(scene([((170, 300), x) for x in (40, 80, 120, 160)]))
    assert r['parameters']['ring_count'] >= 3 and r['status'] == 'descriptive'
    cx, cy = r['parameters']['center']
    assert abs(cx - 170 / 511) < .06 and abs(cy - 300 / 511) < .06


def test_noise_and_random_structure_give_no_rings():
    rng = np.random.default_rng(1)
    im = np.uint8(np.clip(rng.normal(128, 60, (512, 512, 3)), 0, 255))
    b = io.BytesIO(); Image.fromarray(im).save(b, 'PNG')
    rgb, src = load_image(b.getvalue()); src['name'] = 'n'
    r = ring_read(extract(rgb, src, 'raw', None, None))
    assert r['parameters']['ring_count'] == 0 and r['fit'] == 0


def test_blank_field_is_ineligible():
    b = io.BytesIO(); Image.fromarray(np.full((128, 128, 3), 255, np.uint8)).save(b, 'PNG')
    rgb, src = load_image(b.getvalue()); src['name'] = 'b'
    r = ring_read(extract(rgb, src, 'raw', None, None))
    assert not r['eligible']


def test_reference_rings_cover_both_centres_and_are_labelled_as_drawn():
    from composition_reader.rings import reference_rings
    ref = reference_rings(scene([((256, 256), 90)]), n=10)
    assert ref['rings'] == 10 and 'not detected' in ref['note']
    for key in ('frame', 'mass'):
        assert len(ref[key]['geometry']) >= 10 and len(ref[key]['center']) == 2
