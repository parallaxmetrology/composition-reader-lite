from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

SCHEMA = "composition-reader/1"
ENGINE = "0.8.0"

@dataclass
class Scene:
    rgb: np.ndarray
    mass: np.ndarray
    absolute_mass: np.ndarray
    tone: np.ndarray
    chroma: np.ndarray
    edge: np.ndarray
    labels: np.ndarray
    regions: list[dict]
    lines: list[dict]
    summary: dict
    warnings: list[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)
    subjects: list[dict] | None = None
    subject_map: np.ndarray | None = None
    subject_metadata: dict = field(default_factory=dict)

    @property
    def placement_regions(self):
        return [r for r in (self.subjects if self.subjects is not None else self.regions) if r.get('included',True)]

    @property
    def placement_points(self):
        return np.array([r['center'] for r in self.placement_regions],dtype=float).reshape(-1,2)

    @property
    def placement_weights(self):
        weights=np.array([r.get('placement_weight',r.get('mass_share',0)) for r in self.placement_regions],dtype=float)
        return weights/weights.sum() if weights.sum() else weights

    @property
    def points(self):
        return np.array([r['center'] for r in self.regions], dtype=float).reshape(-1, 2)

    @property
    def weights(self):
        w = np.array([r['mass_share'] for r in self.regions], dtype=float)
        return w / w.sum() if w.sum() else w

    @property
    def aspect(self):
        return self.rgb.shape[1] / self.rgb.shape[0]

    @property
    def metric_scale(self):
        h, w = self.mass.shape
        return np.array([w, h]) / np.hypot(w, h)


def native(value):
    if isinstance(value, dict): return {str(k):native(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [native(v) for v in value]
    if isinstance(value, np.ndarray): return native(value.tolist())
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 7) if np.isfinite(value) else None
    if isinstance(value, np.bool_): return bool(value)
    return value
