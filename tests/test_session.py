"""Restoring a run from disk must reproduce the original metrics exactly."""
import sys; sys.path.insert(0, "/home/claude/v2")
import numpy as np, pandas as pd
from lstpipe import Config, session
from lstpipe import model as mdl
from lstpipe.harvest import Harvest

rng = np.random.default_rng(0)
cfg = Config.manila(mode="Day", base_dir="/tmp/sess2")

class _S:
    def __init__(s, h): s.local_hour = h
class _SD:
    def __init__(s, h): s.scene=_S(h); s.s2_info={"gap_days":2.0,"n_items":3}; s.lst=object()

n_sc, n_px = 8, 4000
X, y, keys = [], [], []
base = rng.random((n_px, 10))
for i in range(n_sc):
    lvl = rng.normal(305, 6)
    X.append(base + rng.normal(0, 0.02, base.shape))
    y.append(lvl + 5*base[:,3] - 2*base[:,7] + rng.normal(0, .3, n_px))
    keys.append(np.full(n_px, f"2025-0{i+1}-15_0830", dtype=object))
h = Harvest(X=np.vstack(X), y=np.concatenate(y), scene_key=np.concatenate(keys),
            lulc=rng.choice([10,50,80], n_sc*n_px).astype(float),
            scenes={f"2025-0{i+1}-15_0830": _SD(8+i) for i in range(n_sc)},
            gaps=pd.DataFrame())

# --- original training run, then persist ---
fit0 = mdl.train(cfg, h, cross_validate=False, verbose=False)
mdl.save(cfg, fit0)
from lstpipe import harvest as hv
h.save(hv.cache_path(cfg))

# --- restore from disk alone ---
h1, fit1 = session.load(cfg, verbose=False)

assert np.allclose(h1.X, h.X) and np.allclose(h1.y, h.y)
assert np.array_equal(h1.scene_key.astype(str), h.scene_key.astype(str))
assert np.allclose(h1.lulc, h.lulc)
assert h1.n_scenes == h.n_scenes
print(f"arrays restored: {h1.X.shape[0]:,} samples, {h1.n_scenes} scenes")

# the split must be regenerated identically, so metrics match to machine precision
assert np.array_equal(fit1.train_idx, fit0.train_idx), "train split differs"
assert np.array_equal(fit1.test_idx, fit0.test_idx), "test split differs"
for a, b, lbl in [(fit0.pattern, fit1.pattern, "pattern"),
                  (fit0.absolute, fit1.absolute, "absolute")]:
    for x, z in zip(a, b):
        assert abs(x - z) < 1e-12, f"{lbl}: {x} vs {z}"
print(f"pattern  MAE {fit1.pattern[0]:.6f}  R2 {fit1.pattern[2]:.6f}  (identical)")
print(f"absolute MAE {fit1.absolute[0]:.6f}  R2 {fit1.absolute[2]:.6f}  (identical)")

assert np.allclose(fit0.rf.predict(h.X[:500]), fit1.rf.predict(h.X[:500]))
assert list(fit0.importance.feature) == list(fit1.importance.feature)
print("model predictions and feature importances identical")

# local hours survive, so time-of-day diagnostics work from a restored session
assert [h1.scenes[k].scene.local_hour for k in sorted(h1.scenes)] == \
       [h.scenes[k].scene.local_hour for k in sorted(h.scenes)]
print("scene local hours preserved - time-of-day analysis works from cache")

st = session.available(cfg)
assert st["harvest"] and st["model"]
print(f"session.available -> {st}")
print("\nSESSION RESTORE: PASS - no network, metrics identical to 1e-12")
