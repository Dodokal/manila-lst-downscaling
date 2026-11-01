"""
Map generation and derived layers.
==================================

Three product families, in the order they must be produced:

**Per-overpass maps** - the operational deliverable. The model supplies the spatial
pattern, the scene's own ECOSTRESS mean supplies the level, and a mass-conserving residual
correction anchors the result so that aggregating it back to 70 m reproduces the original
observation exactly.

**Planner composites** - mean, hot-spot frequency and 95th percentile, obtained by
averaging the per-scene maps. Not by training a second model on blended input: that
approach was tested earlier in the project and failed badly, because a composite target has
no single acquisition to anchor to.

**Hot-day layers** - exceedance counts and their trend. Read the caveat in
:func:`hot_days` before quoting these.

v2.1 changes, each answering a point in MSF's delivery feedback:

* Per-scene maps exist **only where ECOSTRESS observed the surface** (plus at most
  ``residual_fill_px`` coarse pixels of interpolation). v2.0 filled cloud holes and
  unobserved ground with upsampled or column-interpolated values.
* The 10 m maps are cut at the **WorldCover shoreline**, not along the 70 m staircase of
  the ECOSTRESS water mask.
* Composites use only grade A/B scenes inside ``product_hours`` and are blank where fewer
  than ``min_obs_composite`` clean overpasses exist (edges, waterfront). An observation
  count and a standard-error layer are written beside them.
* The mean is the pixel effect of a **two-way additive model** (pixel + overpass). Scenes
  cover different footprints at different hours; a plain per-pixel average jumps wherever
  the set of covering scenes changes - the blocky seams MSF found in the MEAN layers.
* Hot-day layers report the **share of clean overpasses** above the threshold. v2.0
  multiplied that share by 365 and printed it as days per year ("205 days/yr").
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from scipy import ndimage as ndi

from . import covariates as cov
from .config import WC_CLASSES
from .preprocess import nan_gauss


def write_raster(path, array, crs, transform, name="LST", units="C"):
    """Write a single-band float32 GeoTIFF with NaN nodata.

    rasterio is used directly rather than going through xarray, which raises on
    ``_FillValue`` when the array came from a masked read.
    """
    arr = np.asarray(array, "float32")
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs=crs, transform=transform,
                       nodata=float("nan")) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, name)
        dst.update_tags(1, units=units)
    return str(path)


def read_raster(path):
    """Read a single-band raster as float with NaN nodata, plus its georeferencing."""
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(float)
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        return a, ds.crs, ds.transform


def _in_hours(hour, window) -> bool:
    if window is None or hour is None or (isinstance(hour, float) and np.isnan(hour)):
        return True
    a, b = window
    return a <= int(hour) <= b if a <= b else (int(hour) >= a or int(hour) <= b)


def per_scene_maps(cfg, fit, h, grids, verbose: bool = True) -> pd.DataFrame:
    """Produce one 10 m map per overpass, plus land-cover statistics.

    The scene's own model (°C) supplies the 10 m pattern and level; the residual against the
    scene's clean 70 m observation is added back, so aggregating the map to 70 m reproduces
    what ECOSTRESS saw. Pixels ECOSTRESS did not observe stay empty.
    """
    if not h.has_rasters:
        raise RuntimeError(
            "per_scene_maps needs the per-scene rasters, which a cached harvest does not "
            "carry. Re-run harvest.collect(..., rebuild=True), or use run_pipeline.")

    inventory, zonal = [], []
    land10 = grids.land_fine
    if verbose:
        print(f"{cfg.mode}: generating {len(fit.models)} maps "
              f"(land cover {'on' if grids.has_landcover else 'off'})\n")

    for i, (key, sd) in enumerate(sorted(h.scenes.items()), 1):
        if key not in fit.models:
            continue
        sc = sd.scene
        tag = cfg.tag(sc.date, sc.utc)

        feats, info = cov.s2_for_date(cfg, grids, sc.date, cfg.res_fine,
                                      grids.fine, grids.dem_fine)
        if feats is None:
            if verbose:
                print(f"  [{i}/{len(h.scenes)}] {tag}: no optical imagery at {cfg.res_fine} m")
            continue

        mask = cov.valid_mask(feats, cfg.features)
        pred = np.full(mask.shape, np.nan)
        pred[mask] = fit.models[key].predict(cov.stack(feats, cfg.features, mask))
        pred_da = xr.DataArray(pred, dims=["y", "x"],
                               coords={"y": grids.fine.y, "x": grids.fine.x}
                               ).rio.write_crs(cfg.crs).rio.write_nodata(np.nan)

        # residual against the clean observation, interpolated only a short way into gaps
        obs = sd.lst                                       # °C, NaN where not clean
        agg = pred_da.rio.reproject_match(obs, resampling=Resampling.average)
        r = obs.values.astype(float) - agg.values.astype(float)
        have = np.isfinite(r)
        near = ndi.distance_transform_edt(~have) <= cfg.residual_fill_px
        r_fill = np.where(have, r, nan_gauss(r, 1.0))
        r_fill[~near] = np.nan
        observed = np.isfinite(obs.values)
        reach = ndi.distance_transform_edt(~observed) <= cfg.residual_fill_px
        resid10 = (obs.copy(data=r_fill.astype("float32"))
                   .rio.reproject_match(pred_da, resampling=Resampling.bilinear))
        reach10 = (obs.copy(data=reach.astype("float32"))
                   .rio.reproject_match(pred_da, resampling=Resampling.nearest)).values == 1

        final = pred_da + resid10
        # optical gap inside an observed cell: the observation itself, at 70 m
        eco_up = obs.rio.reproject_match(pred_da, resampling=Resampling.bilinear)
        final = final.where(np.isfinite(final), eco_up)
        # a bilinear residual is smooth but does not close each 70 m cell exactly;
        # a few further passes restore closure without reintroducing 70 m blocks
        for _ in range(cfg.closure_iters):
            agg = final.rio.reproject_match(obs, resampling=Resampling.average)
            r2 = np.where(observed, obs.values - agg.values, 0.0)
            r2 = np.nan_to_num(r2)
            final = final + (obs.copy(data=r2.astype("float32"))
                             .rio.reproject_match(pred_da, resampling=Resampling.bilinear))
        fv = final.values.astype(float)
        fv[~reach10] = np.nan
        if land10 is not None and land10.shape == fv.shape:
            fv[~land10] = np.nan
        sharpened = float((np.isfinite(fv) & mask).sum() / max(np.isfinite(fv).sum(), 1))

        write_raster(cfg.path("scenes", f"{tag}_lst_{cfg.res_coarse}m_C.tif"),
                     obs.values, obs.rio.crs, obs.rio.transform())
        p10 = write_raster(cfg.path("scenes", f"{tag}_lst_{cfg.res_fine}m_C.tif"),
                           fv, final.rio.crs, final.rio.transform())

        celsius = fv
        if grids.has_landcover and grids.wc_fine.shape == celsius.shape:
            for code, cls in WC_CLASSES.items():
                sel = (grids.wc_fine == code) & np.isfinite(celsius)
                if sel.sum() < 20:
                    continue
                v = celsius[sel]
                zonal.append({"date": sc.date, "utc": sc.utc, "mode": cfg.mode,
                              "local_hour": sc.local_hour, "lulc": cls,
                              "pixels": int(sel.sum()),
                              "mean_C": round(float(v.mean()), 2),
                              "median_C": round(float(np.median(v)), 2),
                              "std_C": round(float(v.std()), 2),
                              "p95_C": round(float(np.percentile(v, 95)), 2)})

        rep = sd.clean.report if sd.clean is not None else {}
        grade = rep.get("grade", "B")
        land_n = land10.sum() if land10 is not None else fv.size
        coverage = float(np.isfinite(fv).sum() / max(land_n, 1))
        inventory.append({"date": sc.date, "utc": sc.utc, "mode": cfg.mode,
                          "local_hour": sc.local_hour, "grade": grade,
                          "clean_land_frac": rep.get("clean_land_frac"),
                          "in_composites": bool(grade in cfg.product_grades and
                                                _in_hours(sc.local_hour, cfg.product_hours)),
                          "s2_gap_days": sd.s2_info.get("gap_days"),
                          "coverage": round(coverage, 2),
                          "sharpened_frac": round(sharpened, 2),
                          "scene_mean_C": round(float(np.nanmean(obs.values)), 2),
                          "mean_C": round(float(np.nanmean(celsius)), 2),
                          "p95_C": round(float(np.nanpercentile(celsius, 95)), 2),
                          "file_10m": p10})
        if verbose:
            print(f"  [{i}/{len(h.scenes)}] {tag} · grade {grade} · "
                  f"coverage {100 * coverage:.0f}% · sharpened {100 * sharpened:.0f}%")

    inv = pd.DataFrame(inventory)
    inv.to_csv(cfg.path("products", f"inventory_{cfg.key}.csv"), index=False)
    if zonal:
        z = pd.DataFrame(zonal)
        z.to_csv(cfg.path("products", f"zonal_stats_lulc_{cfg.key}.csv"), index=False)
        summ = (z.groupby("lulc")
                .agg(scenes=("date", "nunique"), mean_C=("mean_C", "mean"),
                     p95_C=("p95_C", "mean"))
                .sort_values("mean_C", ascending=False))
        summ.to_csv(cfg.path("products", f"zonal_summary_{cfg.key}.csv"))
        if verbose:
            print("\nmean temperature by land cover:")
            print(summ.round(2).to_string())

    if verbose:
        print(f"\n{cfg.mode}: {len(inv)} maps -> {cfg.dirs['scenes']}")
    return inv


def _selected_maps(cfg):
    """The per-scene maps cleared for composites, from this run's inventory."""
    p = cfg.path("products", f"inventory_{cfg.key}.csv")
    if not p.exists():
        return [], pd.DataFrame()
    inv = pd.read_csv(p)
    if "in_composites" in inv:
        inv = inv[inv.in_composites.astype(bool)]
    inv = inv[[os.path.exists(f) for f in inv.file_10m]]
    return inv.file_10m.tolist(), inv


def _stack(tifs):
    stack, crs, transform, shape = [], None, None, None
    for t in tifs:
        a, c, tr = read_raster(t)
        if crs is None:
            crs, transform, shape = c, tr, a.shape
        stack.append(a if a.shape == shape else np.full(shape, np.nan))
    return np.stack(stack), crs, transform, shape


def two_way_mean(A: np.ndarray, iters: int = 20):
    """Pixel effect of LST(s, p) = mu(p) + b(s) + e, with mean(b) = 0.

    Equals the plain per-pixel mean where every scene covers the pixel; where coverage
    differs it removes the level of the scenes that happen to be present, so footprint
    edges do not appear as steps in the composite.
    """
    ok = np.isfinite(A)
    b = np.zeros(A.shape[0])
    for _ in range(iters):
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(A - b[:, None, None], axis=0)
            b_new = np.array([np.nanmean((A[s] - mu)[ok[s]]) if ok[s].any() else 0.0
                              for s in range(A.shape[0])])
        b_new -= b_new.mean()
        if np.max(np.abs(b_new - b)) < 1e-4:
            b = b_new
            break
        b = b_new
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(A - b[:, None, None], axis=0)
    return mu, b


def composites(cfg, verbose: bool = True) -> dict:
    """Planner layers from the clean per-scene maps."""
    tifs, inv = _selected_maps(cfg)
    if not tifs:
        print("no per-scene maps cleared for composites - run per_scene_maps first, or "
              "relax product_grades / product_hours")
        return {}

    A, crs, transform, shape = _stack(tifs)
    n_obs = np.isfinite(A).sum(axis=0)
    enough = n_obs >= cfg.min_obs_composite
    if verbose:
        hrs = (f"{cfg.product_hours[0]:02d}-{cfg.product_hours[1]:02d} h local"
               if cfg.product_hours else "all hours")
        print(f"{cfg.mode}: compositing {A.shape[0]} clean overpasses ({hrs}) of {shape}")

    mu, b = two_way_mean(A)
    adj = A - b[:, None, None]
    with np.errstate(invalid="ignore"):
        se = np.nanstd(adj, axis=0) / np.sqrt(np.maximum(n_obs, 1))
        scene_p = np.array([np.nanpercentile(a, cfg.hotspot_percentile) for a in A])
        hot = np.where(np.isfinite(A), (A >= scene_p[:, None, None]).astype(float), np.nan)
        freq = np.nanmean(hot, axis=0)
        p95 = np.nanpercentile(adj, 95, axis=0)

    blank = lambda a: np.where(enough, a, np.nan)          # noqa: E731
    out = {f"MEAN_LST_{cfg.mode}": blank(mu),
           f"MEAN_LST_SE_{cfg.mode}": blank(se),
           f"HOTSPOT_FREQ_{cfg.mode}": blank(freq),
           f"P95_LST_{cfg.mode}": blank(p95),
           f"OBS_COUNT_{cfg.mode}": n_obs.astype("float32")}

    paths = {}
    for name, arr in out.items():
        p = cfg.path("products", f"{cfg.name}_{name}_{cfg.res_fine}m.tif")
        units = ("fraction" if "FREQ" in name else "overpasses" if "COUNT" in name
                 else "C")
        paths[name] = write_raster(p, arr, crs, transform, name=name, units=units)
        if verbose:
            print(f"  {os.path.basename(p)}")
    pd.DataFrame({"file_10m": tifs, "scene_offset_C": np.round(b, 2)}).to_csv(
        cfg.path("products", f"composite_scenes_{cfg.key}.csv"), index=False)
    if verbose:
        print(f"  blank where fewer than {cfg.min_obs_composite} clean overpasses: "
              f"{100 * float((~enough & (n_obs > 0)).sum()) / max((n_obs > 0).sum(), 1):.1f} %"
              " of observed area")
        print(f"  hot spot = among the hottest {100 - cfg.hotspot_percentile} % of the city "
              "in that overpass")
    return {"arrays": out, "paths": paths, "crs": crs, "transform": transform,
            "scene_offsets": b, "n": A.shape[0]}


def hot_days(cfg, verbose: bool = True) -> dict:
    """Share of clean overpasses above the hot threshold, per year, and its trend.

    **Units.** The layers are a *fraction of observed overpasses* (0-1), not days. ECOSTRESS
    samples an instantaneous temperature on irregular days; v2.0 multiplied the fraction by
    365, which turned "30 % of 31 overpasses" into "205 days per year".

    **Interpretation caveat.** Year-to-year differences track which overpasses happened to
    be cloud-free, and at which hour, far more strongly than any climate signal. Quote the
    pooled mean layer; treat the trend as indicative only.
    """
    from scipy import stats

    tifs, inv = _selected_maps(cfg)
    if not tifs:
        print("no per-scene maps cleared for composites")
        return {}
    years = np.array([int(re.search(r"(\d{4})-\d{2}-\d{2}", os.path.basename(t)).group(1))
                      for t in tifs])
    A, crs, transform, shape = _stack(tifs)

    uy = np.array(sorted(set(years)))
    days = np.full((len(uy),) + shape, np.nan, "float32")
    for i, yr in enumerate(uy):
        sub = A[years == yr]
        n_obs = np.isfinite(sub).sum(axis=0)
        n_hot = (np.nan_to_num(sub, nan=-999) >= cfg.hot_threshold_c).sum(axis=0)
        rate = np.divide(n_hot, np.maximum(n_obs, 1), dtype="float32")
        days[i] = np.where(n_obs >= cfg.min_obs_per_year, rate, np.nan)
        if verbose:
            print(f"  {yr}: {int((years == yr).sum())} clean overpasses · median pixel above "
                  f"{cfg.hot_threshold_c:.0f} °C in {100 * np.nanmedian(days[i]):.0f} % of them")

    # per-pixel ordinary least squares with a two-sided t-test
    x = uy.astype(float)
    valid = np.isfinite(days)
    n = valid.sum(axis=0)
    Y = np.where(valid, days, 0.0)
    X = np.where(valid, x[:, None, None], 0.0)
    mx = np.divide(X.sum(0), np.maximum(n, 1))
    my = np.divide(Y.sum(0), np.maximum(n, 1))
    dx = np.where(valid, X - mx, 0.0)
    dy = np.where(valid, Y - my, 0.0)
    Sxx, Sxy = (dx ** 2).sum(0), (dx * dy).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = Sxy / Sxx
        sse = (np.where(valid, dy - slope * dx, 0.0) ** 2).sum(0)
        se = np.sqrt(sse / np.maximum(n - 2, 1) / Sxx)
        pval = 2 * (1 - stats.t.cdf(np.abs(slope / se), np.maximum(n - 2, 1)))
    ok = (n >= 3) & np.isfinite(slope) & (Sxx > 0)
    slope = np.where(ok, slope, np.nan)
    sig = np.where(np.where(ok, pval, np.nan) < 0.05, slope, np.nan)

    paths = {}
    for i, yr in enumerate(uy):
        paths[f"HOTFRAC_{yr}"] = write_raster(
            cfg.path("products", f"{cfg.name}_HOTFRAC_{yr}_{cfg.mode}_{cfg.res_fine}m.tif"),
            days[i], crs, transform, name=f"HOTFRAC_{yr}", units="fraction of overpasses")
    for name, arr, units in [("HOTFRAC_MEAN", np.nanmean(days, axis=0),
                              "fraction of overpasses"),
                             ("HOTFRAC_TREND_all", slope, "fraction per year"),
                             ("HOTFRAC_TREND_p05", sig, "fraction per year")]:
        paths[name] = write_raster(
            cfg.path("products", f"{cfg.name}_{name}_{cfg.mode}_{cfg.res_fine}m.tif"),
            arr, crs, transform, name=name, units=units)

    n_sig = int(np.isfinite(sig).sum())
    if verbose:
        testable = max(int(np.isfinite(slope).sum()), 1)
        print(f"\ntrend significant at p<0.05 for {n_sig:,} pixels "
              f"({100 * n_sig / testable:.1f} % of testable)")
        if len(uy) < 5:
            print(f"  only {len(uy)} years - treat the trend as indicative and lead with "
                  "the per-year and mean layers")
    return {"years": uy, "days": days, "slope": slope, "significant": sig,
            "paths": paths, "crs": crs, "transform": transform}


def day_night_layers(cfg, verbose: bool = True) -> dict:
    """Diurnal range, compound risk and no-relief zones. Requires both products.

    Two locations can share a daytime temperature yet carry very different risk: one cools
    overnight, the other does not. Heat that persists after dark leaves no recovery window,
    which is the most direct link between these maps and health outcomes.
    """
    p_day = cfg.path("products", f"{cfg.name}_MEAN_LST_Day_{cfg.res_fine}m.tif")
    p_night = cfg.path("products", f"{cfg.name}_MEAN_LST_Night_{cfg.res_fine}m.tif")
    missing = [p for p in (p_day, p_night) if not p.exists()]
    if missing:
        print("day-night layers need both products; missing:")
        for p in missing:
            print("  ", p.name)
        return {}

    day, crs, transform = read_raster(p_day)
    night, _, _ = read_raster(p_night)
    if day.shape != night.shape:
        hgt = min(day.shape[0], night.shape[0])
        wid = min(day.shape[1], night.shape[1])
        day, night = day[:hgt, :wid], night[:hgt, :wid]

    dtr = day - night
    ok = np.isfinite(dtr)
    night_hot = night >= np.nanpercentile(night, 75)
    day_hot = day >= np.nanpercentile(day, 75)
    low_dtr = dtr <= np.nanpercentile(dtr, 25)

    compound = np.where(ok, (day_hot & night_hot).astype(float), np.nan)
    no_relief = np.where(ok, (night_hot & low_dtr).astype(float), np.nan)

    paths = {}
    for name, arr, units in [("DIURNAL_RANGE", dtr, "C"),
                             ("COMPOUND_RISK_DayNight", compound, "0-1"),
                             ("NO_RELIEF_ZONE", no_relief, "0-1")]:
        paths[name] = write_raster(
            cfg.path("products", f"{cfg.name}_{name}_{cfg.res_fine}m.tif"),
            arr, crs, transform, name=name, units=units)

    if verbose:
        print(f"diurnal range: mean {np.nanmean(dtr):.2f} °C "
              f"(p5 {np.nanpercentile(dtr, 5):.2f}, p95 {np.nanpercentile(dtr, 95):.2f})")
        print(f"  compound risk (hot day and hot night): {100 * np.nanmean(compound):.1f} % of area")
        print(f"  no-relief zones (warm night, small range): {100 * np.nanmean(no_relief):.1f} %")
    return {"day": day, "night": night, "dtr": dtr, "compound": compound,
            "no_relief": no_relief, "paths": paths, "crs": crs, "transform": transform}
