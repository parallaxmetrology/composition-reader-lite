"""Concentric ring organization: how many rings does the image hold around one centre?

The older circle read keeps only the single best circular contour. This read
asks a different question, taken from the radial-profile idea of RCA-2: around
a centre, at which radii does edge structure gather into a ring that wraps the
centre (rather than one lobe or a smear of texture)?

Candidate centres are the frame centre, the Notan mass centre and the two strongest edge-normal vote peaks, so a picture
organised around the middle of the frame is not missed when its mass sits
elsewhere. The better centre is reported with both fits recorded.

A ring is a radius where (1) edge density per unit circumference stands out
from its radial neighbours, and (2) the edge pixels occupy most of the angular
sectors that lie inside the frame at that radius. Fit is mean ring strength
times a saturating count term, so one ring cannot score as a ring system.
"""
from __future__ import annotations
import numpy as np
from .geometry import poly, point

NSECTORS = 24
BIN = .008            # radial bin width, image diagonals
RMIN, RMAX = .05, .58
MIN_SEPARATION = .03
STRENGTH_MIN = .35
SATURATION = 3.0      # rings needed for the count term to reach 1
MAX_RINGS = 8


def _edge_points(s):
    """Edge pixels as (x, y) fractions, with the unit gradient direction at each."""
    from scipy.ndimage import gaussian_filter
    ys, xs = np.nonzero(s.edge)
    h, w = s.edge.shape
    if len(xs) > 80000:
        keep = np.random.default_rng(0).choice(len(xs), 80000, replace=False)
        ys, xs = ys[keep], xs[keep]
    gy, gx = np.gradient(gaussian_filter(np.asarray(s.tone, float), 1.5))
    g = np.c_[gx[ys, xs], gy[ys, xs]]
    mag = np.hypot(g[:, 0], g[:, 1])
    g = g / np.maximum(mag, 1e-9)[:, None]
    usable = mag > 1e-6
    pts = np.c_[xs / max(w - 1, 1), ys / max(h - 1, 1)][usable]
    return pts, g[usable]


def _vote_centres(pts, normals, scale, extra=2, grid=48):
    """Edge-normal voting (circle Hough): rings' centres lie along their edges' normals."""
    acc = np.zeros((grid, grid))
    step = np.linspace(RMIN, RMAX, 40)
    sub = np.random.default_rng(1).choice(len(pts), min(len(pts), 15000), replace=False)
    p, n = pts[sub], normals[sub]
    for sign in (1, -1):
        for t in step:
            q = p + sign * (t * n) / scale
            ok = _inside(q)
            ix = np.clip((q[ok, 0] * grid).astype(int), 0, grid - 1)
            iy = np.clip((q[ok, 1] * grid).astype(int), 0, grid - 1)
            np.add.at(acc, (iy, ix), 1)
    from scipy.ndimage import gaussian_filter
    acc = gaussian_filter(acc, 1.0)
    out = []
    for _ in range(extra):
        iy, ix = np.unravel_index(np.argmax(acc), acc.shape)
        out.append([(ix + .5) / grid, (iy + .5) / grid])
        y0, y1, x0, x1 = max(0, iy - 5), iy + 6, max(0, ix - 5), ix + 6
        acc[y0:y1, x0:x1] = 0
    return out


def _inside(p):
    return (p[..., 0] >= 0) & (p[..., 0] <= 1) & (p[..., 1] >= 0) & (p[..., 1] <= 1)


def _rings_around(pts, normals, scale, center):
    """Return list of ring dicts around one centre, with the radial profile."""
    c = np.asarray(center, float)
    d = (pts - c) * scale
    r = np.hypot(d[:, 0], d[:, 1])
    th = np.arctan2(d[:, 1], d[:, 0])
    nb = int(np.ceil((RMAX - RMIN) / BIN))
    rb = np.floor((r - RMIN) / BIN).astype(int)
    ok = (rb >= 0) & (rb < nb)
    # Only edges that face the centre count toward a ring: a ring's edge normal
    # is radial, while clutter lines cross radii at all angles.
    radial = np.abs(np.einsum('ij,ij->i', normals, d / np.maximum(r, 1e-9)[:, None])) ** 3
    sec = np.floor((th[ok] + np.pi) / (2 * np.pi) * NSECTORS).astype(int) % NSECTORS
    counts = np.zeros((nb, NSECTORS))
    np.add.at(counts, (rb[ok], sec), radial[ok])
    radii = RMIN + (np.arange(nb) + .5) * BIN
    # Sectors whose arc centre lies inside the frame at this radius.
    ang = (np.arange(NSECTORS) + .5) / NSECTORS * 2 * np.pi - np.pi
    unit = np.c_[np.cos(ang), np.sin(ang)]
    valid = np.zeros((nb, NSECTORS), bool)
    for k, rad in enumerate(radii):
        q = c + unit * rad / scale
        valid[k] = _inside(q)
    total = counts.sum(1)
    density = total / radii
    base = density.mean() + 1e-9
    strength = np.zeros(nb);coverage = np.zeros(nb);prominence = np.zeros(nb)
    for k in range(nb):
        v = valid[k]
        if v.sum() < NSECTORS // 2 or total[k] < 20:
            continue
        lo, hi = max(0, k - 7), min(nb, k + 8)
        neighbours = np.r_[density[lo:k - 1], density[k + 2:hi]]
        ref = (np.median(neighbours) if len(neighbours) else base) + .25 * base
        prom = density[k] / ref
        # Adjacent bins belong to the same ring: take the best of k-1..k+1 sectors.
        sector_counts = counts[max(0, k - 1):k + 2].sum(0)
        need = max(1, .25 * sector_counts[v].mean())
        cov = float(np.mean(sector_counts[v] >= need))
        prominence[k] = prom;coverage[k] = cov
        strength[k] = float(np.clip((prom - 1.4) / 2.2, 0, 1) * cov ** 2)
    # Local maxima above threshold, thinned to a minimum separation.
    order = [k for k in np.argsort(-strength) if strength[k] >= STRENGTH_MIN]
    rings = []
    for k in order:
        if all(abs(radii[k] - x['radius']) >= MIN_SEPARATION for x in rings):
            rings.append({'radius': float(radii[k]), 'strength': float(strength[k]),
                          'coverage': float(coverage[k]), 'prominence': float(prominence[k]),
                          'frame_sectors': float(valid[k].mean())})
        if len(rings) >= MAX_RINGS:
            break
    rings.sort(key=lambda x: x['radius'])
    return rings


def _fit(rings):
    if not rings:
        return 0.
    return float(np.mean([x['strength'] for x in rings]) * min(1., len(rings) / SATURATION))


def _arcs(center, radius, scale):
    t = np.linspace(0, 2 * np.pi, 181)
    q = np.asarray(center) + radius * np.c_[np.cos(t), np.sin(t)] / scale
    keep = _inside(q)
    runs, cur = [], []
    for p, k in zip(q, keep):
        if k:
            cur.append(p)
        elif cur:
            runs.append(cur);cur = []
    if cur:
        runs.append(cur)
    if len(runs) > 1 and keep[0] and keep[-1]:   # join the seam
        runs[0] = runs.pop() + runs[0]
    return [poly(np.array(r), False) for r in runs if len(r) > 2]


def ring_read(s):
    from .hypotheses import result
    if s.summary['blank']:
        return result('concentric-rings', 'curve', 0, eligible=False,
                      explanation='An empty field supplies no ring evidence.')
    pts, normals = _edge_points(s)
    if len(pts) < 200:
        return result('concentric-rings', 'curve', 0, eligible=False,
                      explanation='Too little edge structure to look for rings.')
    scale = s.metric_scale
    centres = {'frame centre': [.5, .5], 'mass centre': list(map(float, s.summary['mass_center']))}
    for i, c in enumerate(_vote_centres(pts, normals, scale), 1):
        if all(np.linalg.norm((np.array(c) - np.array(o)) * scale) > .05 for o in centres.values()):
            centres[f'edge-vote centre {i}'] = c
    found = {name: _rings_around(pts, normals, scale, c) for name, c in centres.items()}
    fits = {name: _fit(r) for name, r in found.items()}
    name = max(fits, key=fits.get)
    rings, fit, c = found[name], fits[name], centres[name]
    geometry = [point(c)]
    for x in rings:
        geometry += _arcs(c, x['radius'], scale)
    n = len(rings)
    explanation = (f'{n} ring{"s" if n != 1 else ""} around the {name}, at radii '
                   + ', '.join(f'{x["radius"]:.2f}' for x in rings) + ' of the image diagonal.') if n else \
        'No radius gathers edge structure into a ring around either centre.'
    return result('concentric-rings', 'curve', fit, geometry,
                  [{'element_id': f'ring{i + 1}', 'measure': 'ring strength', 'value': x['strength'], 'unit': 'support',
                    'radius_diagonal': x['radius'], 'angular_coverage': x['coverage'],
                    'density_prominence': x['prominence']} for i, x in enumerate(rings)],
                  explanation,
                  ['A ring is a radius where edge density stands out and wraps the centre; it is not proof the picture was composed as concentric.',
                   'Up to four centres are tried (frame, mass, two edge-vote peaks) and the best is reported; every fit is recorded, so the choice is visible.',
                   'Fit rises with the number of rings; one ring scores low by design (see circular contour organization).',
                   'Rings that leave the frame are judged only on the part inside it.'],
                  parameters={'center': c, 'center_name': name, 'ring_count': n,
                              'ring_radii_diagonal': [x['radius'] for x in rings],
                              'fit_by_center': fits,
                              'rings_by_center': {k: [x['radius'] for x in v] for k, v in found.items()},
                              'decision_threshold': .5})


def reference_rings(s, n=14):
    """Evenly spaced rings around the frame centre and the mass centre (the RCA-2 view).

    This is a drawn reference, not a detection: it lets the eye judge how an
    image's structure pulls toward or away from each centre.
    """
    scale = s.metric_scale
    centres = {'frame': [.5, .5], 'mass': list(map(float, s.summary['mass_center']))}
    out = {'rings': n, 'note': 'Evenly spaced reference rings; drawn, not detected.'}
    for name, c in centres.items():
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
        rmax = float(np.max(np.linalg.norm((corners - np.array(c)) * scale, axis=1)))
        polys = []
        for i in range(1, n + 1):
            polys += _arcs(c, i / (n + 1) * rmax, scale)
        out[name] = {'center': c, 'geometry': polys}
    return out
