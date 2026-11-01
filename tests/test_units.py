"""
Component behaviour tests.

Run: python tests/test_units.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from lstpipe import Config
from lstpipe import covariates as cov
from lstpipe import model as mdl
from lstpipe import preprocess as pre

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- config
cfg = Config.manila(mode="Day", base_dir="/tmp/units")
assert cfg.windows == ((6, 18),) and cfg.product_hours == (9, 16)
assert Config.manila(mode="Night", base_dir="/tmp/units").windows == ((18, 23), (0, 5))
assert cfg.hot_threshold_c == 35.0
assert Config.manila(mode="Night", base_dir="/tmp/units").hot_threshold_c == 26.0
assert all(d.is_dir() for d in cfg.dirs.values())
try:
    Config.manila(mode="Evening", base_dir="/tmp/units"); raise AssertionError("should reject")
except ValueError:
    pass
print("config: windows, thresholds, directories, validation: PASS")

# Valenzuela must not overlap the Manila training area
m, v = Config.manila(base_dir="/tmp/units").bbox, Config.valenzuela(base_dir="/tmp/units").bbox
ox = max(0, min(m[2], v[2]) - max(m[0], v[0]))
oy = max(0, min(m[3], v[3]) - max(m[1], v[1]))
assert ox * oy == 0, "the near-transfer control must not overlap the training area"
dist = np.hypot((np.mean(v[::2]) - np.mean(m[::2])) * 111 * np.cos(np.radians(14.7)),
                (np.mean(v[1::2]) - np.mean(m[1::2])) * 111)
assert 5 < dist < 25
print(f"valenzuela: zero overlap with training, {dist:.0f} km away: PASS")

# ---------------------------------------------------------------- covariates
t, s = cov.reference_grid((120.90, 14.55, 121.04, 14.70), "EPSG:32651", 70)
assert t.a == 70 and t.e == -70 and s[0] > 200 and s[1] > 200
from rasterio.warp import transform_bounds
minx, miny, maxx, maxy = transform_bounds("EPSG:4326", "EPSG:32651", 120.90, 14.55, 121.04, 14.70)
assert t.c <= minx and t.f >= maxy                       # grid must cover the whole area
assert t.c + s[1] * 70 >= maxx and t.f - s[0] * 70 <= miny
print(f"reference grid {s} covers the study area: PASS")

nir, red = np.array([0.4, 0.0, 0.5]), np.array([0.1, 0.0, 0.5])
r = cov.safe_ratio(nir, red)
assert abs(r[0] - 0.6) < 1e-12 and np.isnan(r[1]) and r[2] == 0.0
print("safe_ratio: correct value, NaN on a zero denominator: PASS")

feats = {k: rng.random((20, 18)) for k in cfg.features}
feats["ndvi"][5, 5] = np.nan
m_ = cov.valid_mask(feats, cfg.features)
assert not m_[5, 5] and m_.sum() == 20 * 18 - 1
X = cov.stack(feats, cfg.features, m_)
assert X.shape == (m_.sum(), 10) and np.isfinite(X).all()
print("valid_mask and stack: non-finite excluded, order preserved: PASS")

# ---------------------------------------------------------------- model
mae, rmse, r2 = mdl.metrics(np.array([1., 2, 3]), np.array([1., 2, 3]))
assert mae == 0 and rmse == 0 and r2 == 1
print("metrics: exact on a perfect fit: PASS")

# ---------------------------------------------------------------- harvest cache
import tempfile, time
from lstpipe.harvest import Harvest, SceneStub

class _S:
    def __init__(self, hour): self.local_hour = hour
class _SD:
    def __init__(self, hour): self.scene = _S(hour); self.s2_info = {"gap_days": 2.0, "n_items": 3}; self.lst = object()

n = 5000
h = Harvest(X=rng.random((n, 10)), y=30 + rng.random(n) * 8,
            scene_key=np.array([f"2025-0{i%5+1}-15_0830" for i in range(n)], dtype=object),
            lulc=rng.choice([10, 50, 80], n).astype(float),
            scenes={f"2025-0{i+1}-15_0830": _SD(8 + i) for i in range(5)},
            gaps=pd.DataFrame(), rc=rng.integers(0, 200, (n, 2)))
assert h.has_rasters

with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "h.npz"
    h.save(p)
    back = Harvest.load(p)
    assert np.allclose(back.X, h.X) and np.allclose(back.y, h.y)
    assert np.array_equal(back.scene_key.astype(str), h.scene_key.astype(str))
    assert np.allclose(back.lulc, h.lulc) and np.array_equal(back.rc, h.rc)
    assert back.n_scenes == h.n_scenes
    assert [back.scenes[k].scene.local_hour for k in sorted(back.scenes)] == \
           [h.scenes[k].scene.local_hour for k in sorted(h.scenes)]
    assert not back.has_rasters, "a cached harvest must report that it has no rasters"
    size_mb = p.stat().st_size / 1e6
print(f"harvest cache: arrays, keys, land cover and local hours restored exactly "
      f"({size_mb:.1f} MB for {n:,} samples): PASS")

# a cached harvest must be refused by map generation, with a clear message
from lstpipe import products as prd
try:
    prd.per_scene_maps(cfg, None, back, None, verbose=False)
    raise AssertionError("should have refused a cached harvest")
except RuntimeError as e:
    assert "cached harvest" in str(e)
print("map generation refuses a cached harvest with a clear message: PASS")

# lulc=None survives the round trip
h2 = Harvest(X=rng.random((100, 10)), y=rng.random(100),
             scene_key=np.array(["2025-01-15_0830"] * 100, dtype=object), lulc=None,
             scenes={"2025-01-15_0830": _SD(9)}, gaps=pd.DataFrame(),
             rc=rng.integers(0, 50, (100, 2)))
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "h2.npz"; h2.save(p)
    assert Harvest.load(p).lulc is None
print("harvest cache: absent land cover round-trips as None: PASS")

print("\nALL UNIT TESTS PASS")
