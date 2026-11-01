"""Restoring a run from disk must reproduce the original per-scene metrics exactly."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lstpipe import Config, session
from lstpipe import model as mdl, harvest as hv
from lstpipe.harvest import Harvest

rng = np.random.default_rng(0)
cfg = Config.manila(mode="Day", base_dir=tempfile.mkdtemp())

class _S:
    def __init__(s, h): s.local_hour = h
class _SD:
    def __init__(s, h): s.scene = _S(h); s.s2_info = {"gap_days": 2.0, "n_items": 3}; s.lst = object()

n_sc, n_px = 6, 3000
X, y, keys, rc = [], [], [], []
for i in range(n_sc):
    xi = rng.random((n_px, 10))
    X.append(xi); y.append(rng.normal(35, 6) + 5 * xi[:, 3] - 2 * xi[:, 7] + rng.normal(0, .3, n_px))
    keys.append(np.full(n_px, f"2025-0{i+1}-15_0830", dtype=object))
    rc.append(rng.integers(0, 220, (n_px, 2)))
h = Harvest(X=np.vstack(X), y=np.concatenate(y), scene_key=np.concatenate(keys),
            lulc=rng.choice([10, 50, 80], n_sc * n_px).astype(float),
            scenes={f"2025-0{i+1}-15_0830": _SD(9 + i) for i in range(n_sc)},
            gaps=pd.DataFrame(), rc=np.vstack(rc))

fit0 = mdl.train(cfg, h, verbose=False)
mdl.save(cfg, fit0)
h.save(hv.cache_path(cfg), key=cfg.key)
h1, fit1 = session.load(cfg, verbose=False)

assert np.allclose(h1.X, h.X) and np.allclose(h1.y, h.y) and np.array_equal(h1.rc, h.rc)
assert np.array_equal(fit1.train_idx, fit0.train_idx) and np.array_equal(fit1.test_idx, fit0.test_idx)
for a, b in zip(fit0.overall, fit1.overall):
    assert abs(a - b) < 1e-12
k = h.scene_key[:500]
assert np.allclose(fit0.predict(h.X[:500], k), fit1.predict(h.X[:500], k))
print(f"session restore: spatial blocks, per-scene models and metrics identical "
      f"(MAE {fit1.overall[0]:.4f} °C): PASS")
print("\nSESSION RESTORE: PASS")
