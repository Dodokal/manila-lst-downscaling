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

from . import covariates as cov
from .config import WC_CLASSES


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


def per_scene_maps(cfg, fit, h, grids, verbose: bool = True) -> pd.DataFrame:
    """Produce one 10 m map per overpass, plus land-cover statistics.

    Where optical imagery was cloudy the prediction falls back to upsampled ECOSTRESS, so
    coverage matches the thermal input. ``sharpened_frac`` records how much of each map was
    genuinely model-driven - the honest measure of how much is truly at 10 m.
    """
    if not h.has_rasters:
        raise RuntimeError(
            "per_scene_maps needs the per-scene rasters, which a cached harvest does not "
            "carry. Re-run harvest.collect(..., rebuild=True), or use run_pipeline.")

    inventory, zonal = [], []
    if verbose:
        print(f"{cfg.mode}: generating {h.n_scenes} maps "
              f"(land cover {'on' if grids.has_landcover else 'off'})\n")

    for i, (key, sd) in enumerate(sorted(h.scenes.items()), 1):
        sc = sd.scene
        tag = cfg.tag(sc.date, sc.utc)

        feats, _ = cov.s2_for_date(cfg, grids, sc.date, cfg.res_fine,
                                   grids.fine, grids.dem_fine)
        if feats is None:
            if verbose:
                print(f"  [{i}/{h.n_scenes}] {tag}: no optical imagery at "
                      f"{cfg.res_fine} m")
            continue

        mask = cov.valid_mask(feats, cfg.features)
        level = float(np.nanmean(sd.lst.values.astype(float)))
        pred = np.full(mask.shape, np.nan)
        pred[mask] = fit.predict(cov.stack(feats, cfg.features, mask)) + level

        pred_da = xr.DataArray(pred, dims=["y", "x"],
                               coords={"y": grids.fine.y, "x": grids.fine.x}
                               ).rio.write_crs(cfg.crs)

        # fall back to the thermal observation where optical data was missing
        eco_up = sd.lst.rio.reproject_match(pred_da, resampling=Resampling.bilinear)
        pred_da = pred_da.where(np.isfinite(pred_da), eco_up)
        sharpened = float(mask.mean())

        # mass-conserving residual correction against this scene's own reading
        agg = pred_da.rio.reproject_match(sd.lst, resampling=Resampling.average)
        resid = (sd.lst - agg).rio.reproject_match(pred_da, resampling=Resampling.bilinear)
        final = pred_da + resid

        write_raster(cfg.path("scenes", f"{tag}_lst_{cfg.res_coarse}m_C.tif"),
                     sd.lst.values - 273.15, sd.lst.rio.crs, sd.lst.rio.transform())
        p10 = write_raster(cfg.path("scenes", f"{tag}_lst_{cfg.res_fine}m_C.tif"),
                           final.values - 273.15, final.rio.crs, final.rio.transform())

        celsius = final.values.astype(float) - 273.15
        if grids.has_landcover and grids.wc_fine.shape == celsius.shape:
            for code, cls in WC_CLASSES.items():
                sel = (grids.wc_fine == code) & np.isfinite(celsius)
                if sel.sum() < 20:
                    continue
                v = celsius[sel]
                zonal.append({"date": sc.date, "utc": sc.utc, "mode": cfg.mode,
                              "lulc": cls, "pixels": int(sel.sum()),
                              "mean_C": round(float(v.mean()), 2),
                              "median_C": round(float(np.median(v)), 2),
                              "std_C": round(float(v.std()), 2),
                              "p95_C": round(float(np.percentile(v, 95)), 2)})

        coverage = float(np.isfinite(final.values).mean())
        inventory.append({"date": sc.date, "utc": sc.utc, "mode": cfg.mode,
                          "local_hour": sc.local_hour,
                          "s2_gap_days": sd.s2_info["gap_days"],
                          "coverage": round(coverage, 2),
                          "sharpened_frac": round(sharpened, 2),
                          "scene_mean_C": round(level - 273.15, 2),
                          "mean_C": round(float(np.nanmean(celsius)), 2),
                          "p95_C": round(float(np.nanpercentile(celsius, 95)), 2),
                          "file_10m": p10})
        if verbose:
            print(f"  [{i}/{h.n_scenes}] {tag} · gap {sd.s2_info['gap_days']:.0f} d · "
                  f"coverage {100 * coverage:.0f}% · sharpened {100 * sharpened:.0f}%")

    inv = pd.DataFrame(inventory)
    inv.to_csv(cfg.path("products", f"inventory_{cfg.mode}.csv"), index=False)
    if zonal:
        z = pd.DataFrame(zonal)
        z.to_csv(cfg.path("products", f"zonal_stats_lulc_{cfg.mode}.csv"), index=False)
        summ = (z.groupby("lulc")
                .agg(scenes=("date", "nunique"), mean_C=("mean_C", "mean"),
                     p95_C=("p95_C", "mean"))
                .sort_values("mean_C", ascending=False))
        summ.to_csv(cfg.path("products", f"zonal_summary_{cfg.mode}.csv"))
        if verbose:
            print("\nmean temperature by land cover:")
            print(summ.round(2).to_string())

    if verbose:
        print(f"\n{cfg.mode}: {len(inv)} maps -> {cfg.dirs['scenes']}")
    return inv


def composites(cfg, verbose: bool = True) -> dict:
    """Planner layers, by averaging the per-scene maps already on disk."""
    tifs = sorted(glob.glob(str(cfg.path("scenes", f"*_{cfg.mode}_lst_{cfg.res_fine}m_C.tif"))))
    if not tifs:
        print("no per-scene maps found - run per_scene_maps first")
        return {}

    stack, crs, transform, shape = [], None, None, None
    for t in tifs:
        a, c, tr = read_raster(t)
        if crs is None:
            crs, transform, shape = c, tr, a.shape
        if a.shape == shape:
            stack.append(a)
    A = np.stack(stack)
    if verbose:
        print(f"{cfg.mode}: compositing {A.shape[0]} maps of {shape}")

    threshold = float(np.nanpercentile(A, cfg.hotspot_percentile))
    out = {f"MEAN_LST_{cfg.mode}": np.nanmean(A, axis=0),
           f"HOTSPOT_FREQ_{cfg.mode}": np.nanmean((A >= threshold).astype(float), axis=0),    ## this needs change
           f"P95_LST_{cfg.mode}": np.nanpercentile(A, 95, axis=0)}

    paths = {}
    for name, arr in out.items():
        p = cfg.path("products", f"{cfg.name}_{name}_{cfg.res_fine}m.tif")
        units = "fraction" if "FREQ" in name else "C"
        paths[name] = write_raster(p, arr, crs, transform, name=name, units=units)
        if verbose:
            print(f"  {os.path.basename(p)}")
    if verbose:
        print(f"  hot-spot threshold ({cfg.hotspot_percentile}th percentile): "
              f"{threshold:.1f} °C")
    return {"arrays": out, "paths": paths, "crs": crs, "transform": transform,
            "threshold": threshold, "n": A.shape[0]}


def hot_days(cfg, verbose: bool = True) -> dict:
    """Exceedance counts per year and their per-pixel trend.

    **Interpretation caveat.** ECOSTRESS samples instantaneous overpass temperature at
    varying local hours, not a daily mean, and the archive is irregular. Year-to-year
    differences track which overpasses happened to be cloud-free far more strongly than any
    climate signal. Report these as *"days on which an observed overpass exceeded X °C"*,
    pooled across years, and do not claim a trend from a handful of seasons.
    """
    from scipy import stats

    tifs = sorted(glob.glob(str(cfg.path("scenes", f"*_{cfg.mode}_lst_{cfg.res_fine}m_C.tif"))))
    if not tifs:
        print("no per-scene maps found")
        return {}
    years = np.array([int(re.search(r"(\d{4})-\d{2}-\d{2}", os.path.basename(t)).group(1))
                      for t in tifs])

    stack, crs, transform, shape = [], None, None, None
    for t in tifs:
        a, c, tr = read_raster(t)
        if crs is None:
            crs, transform, shape = c, tr, a.shape
        stack.append(a if a.shape == shape else np.full(shape, np.nan))
    A = np.stack(stack)

    uy = np.array(sorted(set(years)))
    days = np.full((len(uy),) + shape, np.nan, "float32")
    for i, yr in enumerate(uy):
        sub = A[years == yr]
        n_obs = np.isfinite(sub).sum(axis=0)
        n_hot = (np.nan_to_num(sub, nan=-999) >= cfg.hot_threshold_c).sum(axis=0)
        rate = np.divide(n_hot, np.maximum(n_obs, 1), dtype="float32")
        days[i] = np.where(n_obs >= cfg.min_obs_per_year, rate * 365.0, np.nan)        # needs change buit still ok
        if verbose:
            print(f"  {yr}: {int((years == yr).sum())} scenes · "
                  f"mean {np.nanmean(days[i]):.0f} days/yr above {cfg.hot_threshold_c:.0f} °C")

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
        paths[f"HOTDAYS_{yr}"] = write_raster(
            cfg.path("products", f"{cfg.name}_HOTDAYS_{yr}_{cfg.mode}_{cfg.res_fine}m.tif"),
            days[i], crs, transform, name=f"HOTDAYS_{yr}", units="days/yr")
    for name, arr, units in [("HOTDAYS_MEAN", np.nanmean(days, axis=0), "days/yr"),
                             ("HOTDAYS_TREND_all", slope, "days/yr per year"),
                             ("HOTDAYS_TREND_p05", sig, "days/yr per year")]:
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
