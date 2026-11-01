"""
End-to-end on the real Manila granules, offline: Sentinel-2, DEM and WorldCover are mocked
(this checks that every stage runs and is wired correctly, not the science of the mock).

Run: python tests/test_integration_offline.py
"""
import sys, glob, re, tempfile
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")

DATA = ROOT / "notebooks" / "manila_6_18_ecostress_data"
if not DATA.exists():
    print("skipped: no ECOSTRESS data folder"); sys.exit(0)

from lstpipe import Config, covariates as cov, harvest as hv, model as mdl
from lstpipe import products as prd, diagnostics as dg, session, viz
from lstpipe.acquire import Scene
from scipy import ndimage as ndi

cfg = Config.manila(mode="Day", base_dir=tempfile.mkdtemp(), px_per_scene=4000)
t70, s70 = cov.reference_grid(cfg.bbox, cfg.crs, 70)
t10, s10 = cov.reference_grid(cfg.bbox, cfg.crs, 10)
coarse, fine = cov._blank(t70, s70, cfg.crs), cov._blank(t10, s10, cfg.crs)

# a fake land-cover map: west strip is sea, a park in the middle, the rest built-up
wc_c = np.full(s70, 50.0); wc_c[:, :45] = 80; wc_c[100:130, 100:140] = 10
wc_f = np.kron(wc_c, np.ones((7, 7)))[:s10[0], :s10[1]]
if wc_f.shape != tuple(s10):
    pad = np.full(s10, 50.0); pad[:wc_f.shape[0], :wc_f.shape[1]] = wc_f; wc_f = pad
grids = cov.Grids(coarse=coarse, fine=fine, dem_coarse=np.zeros(s70), dem_fine=np.zeros(s10),
                  wc_coarse=wc_c, wc_fine=wc_f, catalog=None)

def fake_s2(cfg_, grids_, date, resolution, match_to, dem):
    shp = match_to.shape
    yy, xx = np.mgrid[:shp[0], :shp[1]] / (shp[0] / 240)
    veg = (np.sin(xx / 9) * np.cos(yy / 13) + 1) / 2
    wc = grids_.wc_coarse if resolution == 70 else grids_.wc_fine
    veg = np.where(wc == 10, 0.9, veg * 0.6)
    f = {"blue": 0.05 + 0.02 * veg, "green": 0.06 + 0.03 * veg, "red": 0.12 - 0.08 * veg,
         "nir": 0.15 + 0.35 * veg, "swir16": 0.25 - 0.1 * veg, "swir22": 0.2 - 0.1 * veg,
         "dem": dem}
    f["ndvi"] = cov.safe_ratio(f["nir"], f["red"]); f["ndbi"] = cov.safe_ratio(f["swir16"], f["nir"])
    f["ndwi"] = cov.safe_ratio(f["green"], f["nir"])
    return f, {"n_items": 3, "gap_days": 2.0, "gap_p90": 5.0, "cloud_lt": 100, "filled": {15: 1.0}}
cov.s2_for_date = fake_s2

scenes = []
for f in sorted(glob.glob(str(DATA / "*_LST.tif"))):
    st = f[:-8]; ts = re.search(r"_(\d{8}T\d{6})_", f).group(1)
    acq = datetime.strptime(ts, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    scenes.append(Scene(None, f, st + "_water.tif", st + "_QC.tif", st + "_cloud.tif", acq,
                        acq.astimezone(ZoneInfo(cfg.timezone)).hour))
scenes = scenes[:24]

h = hv.collect(cfg, scenes, grids, verbose=False, cache=True)
assert 5 <= h.n_scenes < len(scenes) and np.nanmax(h.y) < 75 and np.nanmin(h.y) >= 15
q = pd.read_csv(cfg.path("products", f"rejected_scenes_{cfg.key}.csv"))
assert len(q) and q.reject_reason.notna().all()
print(f"harvest: {h.n_scenes} of {len(scenes)} overpasses kept, {len(q)} rejected with reasons, "
      f"y {h.y.min():.1f}-{h.y.max():.1f} °C: PASS")

fit = mdl.train(cfg, h, verbose=False)
mdl.save(cfg, fit)
inv = prd.per_scene_maps(cfg, fit, h, grids, verbose=False)
assert len(inv) == len(fit.models)
a10, _, _ = prd.read_raster(inv.file_10m.iloc[0])
assert np.isnan(a10[:, :45 * 7 - 7]).all(), "sea must be blank at 10 m"
assert np.nanmin(a10) > 10
print(f"maps: {len(inv)} per-scene 10 m maps, sea blank, values {np.nanmin(a10):.1f}-"
      f"{np.nanmax(a10):.1f} °C: PASS")

comp = prd.composites(cfg, verbose=False)
cnt = comp["arrays"]["OBS_COUNT_Day"]; mean = comp["arrays"]["MEAN_LST_Day"]
assert np.isnan(mean[(cnt > 0) & (cnt < cfg.min_obs_composite)]).all()
assert set(inv.loc[inv.in_composites, "local_hour"]).issubset(range(9, 17))
hd = prd.hot_days(cfg, verbose=False)
assert np.nanmax(hd["days"]) <= 1
print(f"composites from {comp['n']} clean 09-16 h overpasses; blank below "
      f"{cfg.min_obs_composite} observations; hot layers are fractions: PASS")

tod = dg.by_local_hour(cfg, h, fit, verbose=False)
dg.extreme_coverage(cfg, h, fit, verbose=False); dg.residual_signs(cfg, h, fit, verbose=False)
dg.residual_tails(cfg, h, fit, verbose=False); dg.by_landcover(cfg, h, fit, verbose=False)
sp = dg.sub_pixel_detail(cfg, verbose=False)
assert (sp.closure_mad_C < 0.06).all(), sp
dg.scene_quality(cfg, verbose=False)
viz.time_of_day(cfg, tod); viz.contact_sheet(cfg); viz.planner_products(cfg, comp)
print(f"diagnostics and figures run; 70 m closure median |error| "
      f"{sp.closure_mad_C.median():.3f} °C: PASS")

h2, fit2 = session.load(cfg, verbose=False)
assert abs(fit2.overall[0] - fit.overall[0]) < 1e-12
print("session.load reproduces the run from cache: PASS")
print("\nOFFLINE INTEGRATION: PASS")
