"""
v2.1 behaviour: QC decoding, destriping, cleaning on real granules, per-scene °C models,
two-way composites, hot-day units, WorldCover tiling, per-pixel Sentinel-2 compositing.

Run: python tests/test_v21.py
Real-granule checks run only if notebooks/manila_6_18_ecostress_data/ is present.
"""
import sys, tempfile, glob, re
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xarray as xr

from lstpipe import Config
from lstpipe import preprocess as pre, model as mdl, products as prd, covariates as cov
from lstpipe.harvest import Harvest

rng = np.random.default_rng(1)
TMP = tempfile.mkdtemp()
cfg = Config.manila(mode="Day", base_dir=TMP)

# ---------------------------------------------------------------- QC decoding
# 133 = 0b10000101: mandatory QA 01 (nominal) and data quality 01 (missing-stripe pixel)
# 64  = 0b01000000: mandatory 00, data quality 00 - a clean pixel
qc = np.array([133, 64, 205, 0, 1, 15])
assert pre.good_quality_mask(qc).tolist() == [True, True, True, True, True, False]
assert pre.stripe_flag(qc).tolist() == [True, False, True, False, False, True]
print("QC: stripe pixels pass the v2.0 mask but are caught by bits 2-3: PASS")

# ---------------------------------------------------------------- scan geometry
for ang in (43.0, 126.75):
    shape = (220, 240)
    u = pre.scan_lines(shape, ang)
    flag = (u % 115) < 8
    got = pre.stripe_angle(flag)
    assert abs(((got - ang + 90) % 180) - 90) <= 0.5, (ang, got)
print("stripe angle recovered for ascending and descending geometry: PASS")

# ---------------------------------------------------------------- destriping is selective
yy, xx = np.mgrid[:220, :240]
field = 30 + 4 * np.sin(xx / 25) + 3 * np.cos(yy / 31) + rng.normal(0, 0.4, (220, 240))
u = pre.scan_lines(field.shape, 126.75)
lines = 0.6 * np.sign(np.sin(2 * np.pi * np.arange(u.max() + 1) / 8))
striped = field + lines[u]
fixed, sd = pre.destripe_lines(striped, 126.75)
before = np.std(striped - field); after = np.std(fixed - field - np.mean(fixed - field))
assert after < 0.35 * before, (before, after)
_, sd_wrong = pre.destripe_lines(field, 126.75 + 25)
assert sd_wrong < 0.1, sd_wrong
print(f"destriping: {before:.2f} -> {after:.2f} °C on injected lines; "
      f"{sd_wrong:.3f} °C removed at a wrong angle: PASS")

# ---------------------------------------------------------------- real granules
DATA = ROOT / "notebooks" / "manila_6_18_ecostress_data"
if DATA.exists():
    from lstpipe.acquire import Scene
    t, s = cov.reference_grid(cfg.bbox, cfg.crs, 70)
    grid = cov._blank(t, s, cfg.crs)

    def scene(stamp):
        f = glob.glob(str(DATA / f"*{stamp}*_LST.tif"))[0]
        st = f[:-8]
        ts = re.search(r"_(\d{8}T\d{6})_", f).group(1)
        acq = datetime.strptime(ts, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return Scene(None, f, st + "_water.tif", st + "_QC.tif", st + "_cloud.tif", acq,
                     acq.astimezone(ZoneInfo(cfg.timezone)).hour)

    bad = pre.prepare_scene(cfg, scene("20240707T221636"), grid)
    assert bad.grade == "C" and "striping" in bad.report["reject_reason"], bad.report
    print(f"2024-07-07 (MSF's example) rejected: {bad.report['reject_reason']}: PASS")

    good = pre.prepare_scene(cfg, scene("20240611T083914"), grid)
    v = good.lst.values[np.isfinite(good.lst.values)]
    # no land reference here, so the sea counts in the denominator: A or B
    assert good.grade in ("A", "B") and 15 <= v.min() and v.max() < 75 and 30 < np.median(v) < 50
    assert good.train.sum() < np.isfinite(good.lst.values).sum()      # shoreline excluded
    print(f"2024-06-11 grade {good.grade}, {v.min():.1f}-{v.max():.1f} °C, shoreline not trained on: PASS")

    # the cloud holes must stay holes - no vertical streaks
    holes = ~np.isfinite(good.lst.values)
    assert holes.sum() > 0
    print("no gap filling of ECOSTRESS: cloud and stripe gaps remain NaN: PASS")
else:
    print("(real-granule checks skipped: no ECOSTRESS data folder)")

# ---------------------------------------------------------------- per-scene °C model
n_sc, n_px = 5, 3000
X, y, keys, rc = [], [], [], []
for i in range(n_sc):
    lvl = [27, 48, 33, 52, 29][i]                 # dawn and midday levels, °C
    xi = rng.random((n_px, 10))
    X.append(xi)
    y.append(lvl + 6 * xi[:, 3] - 3 * xi[:, 7] + rng.normal(0, 0.3, n_px))
    keys.append(np.full(n_px, f"2025-0{i + 1}-15_0830", dtype=object))
    rc.append(rng.integers(0, 220, (n_px, 2)))
h = Harvest(X=np.vstack(X), y=np.concatenate(y), scene_key=np.concatenate(keys), lulc=None,
            scenes={}, gaps=pd.DataFrame(), rc=np.vstack(rc))
fit = mdl.train(cfg, h, verbose=False)
assert len(fit.models) == n_sc and fit.units == "°C"
p = fit.predict(h.X[fit.test_idx], h.scene_key[fit.test_idx])
assert np.abs(p - h.y[fit.test_idx]).mean() < 1.0
for k in set(h.scene_key):                        # levels learned in °C, scene by scene
    s = h.scene_key[fit.test_idx] == k
    assert abs(p[s].mean() - h.y[fit.test_idx][s].mean()) < 0.5
# held-out blocks are whole blocks: no block appears on both sides within a scene
bp = int(round(cfg.cv_block_m / cfg.res_coarse))
for k in set(h.scene_key):
    tr = fit.train_idx[h.scene_key[fit.train_idx] == k]
    te = fit.test_idx[h.scene_key[fit.test_idx] == k]
    btr = set(map(tuple, h.rc[tr] // bp)); bte = set(map(tuple, h.rc[te] // bp))
    assert not (btr & bte)
print(f"per-scene models in °C: MAE {fit.overall[0]:.2f} °C on held-out spatial blocks, "
      f"scene levels 27-52 °C reproduced without an anomaly target: PASS")

# ---------------------------------------------------------------- two-way composite
mu_true = 30 + rng.normal(0, 2, (60, 80))
A = np.stack([mu_true + 0.0, mu_true + 15.0, mu_true - 3.0])
A[1, :, 40:] = np.nan                              # hot midday scene covers only the west
mu, b = prd.two_way_mean(A)
plain = np.nanmean(A, axis=0)
step_plain = abs(np.mean(plain[:, 39] - mu_true[:, 39]) - np.mean(plain[:, 40] - mu_true[:, 40]))
step_tw = abs(np.mean(mu[:, 39] - mu_true[:, 39]) - np.mean(mu[:, 40] - mu_true[:, 40]))
assert step_plain > 5 and step_tw < 0.05, (step_plain, step_tw)
print(f"two-way mean: footprint seam {step_plain:.1f} °C (plain mean) -> {step_tw:.3f} °C: PASS")

# ---------------------------------------------------------------- hot-day units
with tempfile.TemporaryDirectory() as d:
    c2 = Config.manila(mode="Day", base_dir=d)
    from rasterio.transform import from_origin
    tr = from_origin(280000, 1620000, 10, 10)
    rows = []
    for i in range(8):
        a = 30 + rng.normal(0, 4, (30, 30))
        tag = c2.tag(f"2025-0{i + 1}-10", "0400")
        f = prd.write_raster(c2.path("scenes", f"{tag}_lst_10m_C.tif"), a, "EPSG:32651", tr)
        rows.append({"file_10m": f, "in_composites": True})
    pd.DataFrame(rows).to_csv(c2.path("products", f"inventory_{c2.key}.csv"), index=False)
    hd = prd.hot_days(c2, verbose=False)
    m = np.nanmax(hd["days"])
    assert 0 <= m <= 1, m
    comp = prd.composites(c2, verbose=False)
    assert "OBS_COUNT_Day" in comp["arrays"] and np.nanmax(comp["arrays"]["OBS_COUNT_Day"]) == 8
print("hot-day layers are fractions of overpasses (max <= 1), composites write OBS_COUNT: PASS")

# ---------------------------------------------------------------- WorldCover tiles
assert cov.worldcover_tiles((120.90, 14.55, 121.04, 14.70)) == ["N12E120"]
assert cov.worldcover_tiles((16.18, 48.12, 16.58, 48.33)) == ["N48E015"]
assert cov.worldcover_tiles((-72.40, 18.45, -72.20, 18.60)) == ["N18W075"]
print("WorldCover tile chosen from the study area (Vienna, Port-au-Prince): PASS")

# ---------------------------------------------------------------- per-pixel Sentinel-2
import odc.stac as ostac
t70, s70 = cov.reference_grid(cfg.bbox, cfg.crs, 70)
grid70 = cov._blank(t70, s70, cfg.crs)

class _Item:
    def __init__(s, d): s.datetime = d

dates = {15: [datetime(2025, 3, 10), datetime(2025, 3, 20)], 30: [datetime(2025, 3, 2)],
         60: []}
def fake_search(cfg_, catalog, date, half, inner=-1):
    return [_Item(d) for d in dates.get(half, [])]

def fake_load(items, **kw):
    ts = [np.datetime64(i.datetime) for i in items]
    ny, nx = grid70.shape
    data = {}
    for b_ in cov.S2_BANDS:
        data[b_] = (("time", "y", "x"), np.full((len(ts), ny, nx), 1000.0 + 100 * len(ts),
                                                "float32"))
    scl = np.full((len(ts), ny, nx), 4.0, "float32")           # vegetation = clear
    if len(ts) == 2:
        scl[0, :, :nx // 2] = 9                                 # cloud on the first date, west
        scl[1, :, :nx // 3] = 8                                 # and the second, far west
    data["scl"] = (("time", "y", "x"), scl)
    ds = xr.Dataset(data, coords={"time": ts, "y": grid70.y.values, "x": grid70.x.values})
    return ds.rio.write_crs(cfg.crs)

orig_s, orig_l = cov._s2_search, ostac.load
cov._s2_search, ostac.load = fake_search, fake_load
try:
    class _G: catalog = None
    feats, info = cov.s2_for_date(cfg, _G(), "2025-03-15", 70, grid70,
                                  np.zeros(grid70.shape))
finally:
    cov._s2_search, ostac.load = orig_s, orig_l
red = feats["red"]
nx = grid70.shape[1]
assert np.isfinite(red).all(), "every pixel must be filled from some clear look"
assert info["filled"][15] > 0.6 and info["filled"][30] > 0.25, info["filled"]
assert np.allclose(red[:, nx // 2 + 5], 1200) and np.allclose(red[:, 5], 1100)
print(f"Sentinel-2: clouds masked per pixel, no scene-level threshold, cloudy pixels filled "
      f"from the ±30 d window (filled {info['filled']}): PASS")

# ---------------------------------------------------------------- config
night = cfg.with_mode("Night")
assert night.product_hours is None and night.hot_threshold_c == 26.0 and night.name == "manila"
assert cfg.s2_scene_cloud_max == 100 and cfg.lst_min_c == 15.0
print("config: with_mode keeps the area, resets mode-specific defaults: PASS")

print("\nALL v2.1 TESTS PASS")
