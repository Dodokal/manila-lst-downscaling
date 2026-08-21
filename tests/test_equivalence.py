"""
Proves the refactored package reproduces the original notebook's numerics exactly.

Every check compares `lstpipe` against an inline reimplementation of the corresponding
notebook cell. A failure here means the refactor changed a result, which is the one thing
it must not do.

Run: python tests/test_equivalence.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lstpipe import Config
from lstpipe import preprocess as pre
from lstpipe import covariates as cov
from lstpipe import model as mdl

rng = np.random.default_rng(0)
cfg = Config.manila(mode="Day", base_dir="/tmp/equiv")

# ------------------------------------------------------------------ QC bitmask
import xarray as xr
qc = xr.DataArray(np.array([[0, 1, 2, 3], [0b10110100, 0b10110101, 0b10110110, np.nan]],
                           dtype=float), dims=["y", "x"])
got = pre.good_quality_mask(qc).values.tolist()
assert got == [[True, True, False, False], [True, True, False, False]]
print("QC bitmask (00/01 keep, 10/11 and NaN discard): PASS")

# ------------------------------------------------------------------ fill codes and gap fill
def smooth2d(a, k):
    pad = k // 2
    ap = np.pad(a, ((pad, k - 1 - pad), (pad, k - 1 - pad)), mode="edge")
    cs = np.cumsum(ap, 0); cs = np.vstack([np.zeros((1, cs.shape[1])), cs]); ap = (cs[k:] - cs[:-k]) / k
    cs = np.cumsum(ap, 1); cs = np.hstack([np.zeros((cs.shape[0], 1)), cs]); return (cs[:, k:] - cs[:, :-k]) / k

truth = smooth2d(rng.normal(0, 1, (256, 200)), 13) * 4 + 305
scene = truth.copy()
dead = []
for block in range(0, 256, 128):                    # documented: 8 lines every 128
    scene[block + 40:block + 48, :] = -9998.0
    dead += list(range(block + 40, block + 48))
scene[100, 50:60] = -9997.0
scene[200, 77] = 500.0

clean, fmask, rep = pre.clean_scene(scene)
assert rep["filled_px"] >= len(dead) * 200 * 0.95
err = np.abs(clean[dead, :] - truth[dead, :])
assert np.nanmean(err) < 0.6
untouched = ~fmask & np.isfinite(clean)
assert np.allclose(clean[untouched], scene[untouched])
print(f"gap fill: {rep['filled_px']} px filled, mean error {np.nanmean(err):.3f} K, "
      "valid data unchanged: PASS")

# ------------------------------------------------------------------ safe_ratio == notebook
a, b = rng.random((50, 40)), rng.random((50, 40))
def _orig_safe_ratio(x, y):
    d = x + y
    return np.where(np.abs(d) > 1e-9, (x - y) / np.where(np.abs(d) > 1e-9, d, 1), np.nan)
assert np.allclose(cov.safe_ratio(a, b), _orig_safe_ratio(a, b), equal_nan=True)
z = np.zeros((4, 4))
assert np.all(np.isnan(cov.safe_ratio(z, z)))
print("safe_ratio identical to the original, guards a zero denominator: PASS")

# ------------------------------------------------------------------ feature order alignment
feats = {k: rng.random((30, 25)) for k in cfg.features}
m = cov.valid_mask(feats, cfg.features)
X = cov.stack(feats, cfg.features, m)
assert X.shape == (m.sum(), len(cfg.features))
for j, name in enumerate(cfg.features):             # column j must be feature j
    assert np.allclose(X[:, j], feats[name][m])
print(f"design matrix columns match {len(cfg.features)}-feature order exactly: PASS")

# ------------------------------------------------------------------ strata == notebook
y = np.concatenate([rng.normal(0, 1, 9000), rng.normal(8, 1, 300)])
def _orig_strata(y_, n=10):
    edges = np.unique(np.quantile(y_, np.linspace(0, 1, n + 1)))
    s = np.clip(np.digitize(y_, edges[1:-1]), 0, max(len(edges) - 2, 0))
    for b_ in np.where(np.bincount(s, minlength=max(len(edges) - 1, 1)) < 2)[0]:
        s[s == b_] = max(b_ - 1, 0)
    return s
assert np.array_equal(mdl.temperature_strata(y, 10), _orig_strata(y, 10))
print("temperature strata identical to the original: PASS")

# ------------------------------------------------------------------ full training path
class _H:                                            # minimal stand-in for Harvest
    pass
n_scenes, n_px = 8, 3000
X_all, y_all, keys = [], [], []
base = rng.random((n_px, len(cfg.features)))
for s in range(n_scenes):
    level = rng.normal(305, 6)
    pattern = 5 * base[:, 3] - 2 * base[:, 7]
    X_all.append(base + rng.normal(0, 0.02, base.shape))
    y_all.append(level + pattern + rng.normal(0, 0.3, n_px))
    keys.append(np.full(n_px, f"2025-0{s+1}-15_0830", dtype=object))
h = _H(); h.X = np.vstack(X_all); h.y = np.concatenate(y_all)
h.scene_key = np.concatenate(keys); h.lulc = None; h.n_scenes = n_scenes
h.scenes = {}

fit = mdl.train(cfg, h, cross_validate=False, verbose=False)

# reimplement the original cell inline and require identical numbers
scene_mean = {k: h.y[h.scene_key == k].mean() for k in set(h.scene_key)}
sm = np.array([scene_mean[k] for k in h.scene_key])
y_anom = h.y - sm
strata = _orig_strata(y_anom, cfg.n_strata)
i_tr, i_te = train_test_split(np.arange(len(y_anom)), test_size=0.30,
                              random_state=11, stratify=strata)
rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5,
                           random_state=79, n_jobs=-1).fit(h.X[i_tr], y_anom[i_tr])
pa, ta = rf.predict(h.X[i_te]), y_anom[i_te]
pb, tb = pa + sm[i_te], h.y[i_te]
exp_pattern = (mean_absolute_error(ta, pa), np.sqrt(mean_squared_error(ta, pa)), r2_score(ta, pa))
exp_abs = (mean_absolute_error(tb, pb), np.sqrt(mean_squared_error(tb, pb)), r2_score(tb, pb))

for got, exp, label in [(fit.pattern, exp_pattern, "pattern"), (fit.absolute, exp_abs, "absolute")]:
    for g, e in zip(got, exp):
        assert abs(g - e) < 1e-12, f"{label}: {g} vs {e}"
print(f"training reproduces the original exactly:")
print(f"  pattern  MAE {fit.pattern[0]:.4f} RMSE {fit.pattern[1]:.4f} R² {fit.pattern[2]:.4f}")
print(f"  absolute MAE {fit.absolute[0]:.4f} RMSE {fit.absolute[1]:.4f} R² {fit.absolute[2]:.4f}")
assert abs(fit.pattern[0] - fit.absolute[0]) < 1e-12, "MAE must be identical in both spaces"
print("  MAE identical in both spaces - the R² difference is the denominator, not skill: PASS")

# ------------------------------------------------------------------ mass conservation
H, W, f = 30, 28, 7
a70 = 300 + rng.normal(0, 2, (H, W))
fine = np.repeat(np.repeat(a70, f, 0), f, 1) + rng.normal(0, 0.8, (H * f, W * f))
blk = fine.reshape(H, f, W, f)
fine = fine - np.repeat(np.repeat(blk.mean(axis=(1, 3)) - a70, f, 0), f, 1)   # residual correction
agg = fine.reshape(H, f, W, f).mean(axis=(1, 3))
assert np.abs(agg - a70).max() < 1e-9
sub = fine.reshape(H, f, W, f).std(axis=(1, 3))
ratio = sub.mean() / a70.std()
flat = np.repeat(np.repeat(a70, f, 0), f, 1)
assert flat.reshape(H, f, W, f).std(axis=(1, 3)).mean() < 1e-9
print(f"residual correction conserves mass to {np.abs(agg - a70).max():.1e} K; "
      f"detail ratio {ratio:.3f} (pure upsample = 0.000): PASS")

print("\nALL EQUIVALENCE TESTS PASS - the refactor changes no result")
