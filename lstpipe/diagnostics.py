"""
Validation analyses.
====================

Each function answers one question and returns a table, so results can be compared across
runs rather than only read off a figure. All of them work on held-out **spatial blocks**
(see :mod:`lstpipe.model`) and report temperatures and errors in **°C**.

A recurring point of interpretation: **R² per group tracks that group's own spread.** A
uniform scene or a homogeneous land-cover class offers little to resolve, so a low R² there
is expected rather than diagnostic. Compare MAE across groups instead.

Every output file is scoped by study area *and* mode (``cfg.key``), so running Vienna after
Valenzuela can no longer show Valenzuela's numbers.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

from .config import WC_CLASSES
from .model import metrics
from .products import read_raster


def _held_out(h, fit):
    te = fit.test_idx
    pred = fit.predict(h.X[te], h.scene_key[te])
    return te, pred, h.y[te]


# ---------------------------------------------------------------- sample composition
def extreme_coverage(cfg, h, fit, verbose: bool = True) -> pd.DataFrame:
    """Are the hottest and coldest pixels represented on both sides of the split?"""
    is_tr = np.zeros(len(h.y), bool); is_tr[fit.train_idx] = True
    is_te = np.zeros(len(h.y), bool); is_te[fit.test_idx] = True

    rows = []
    for label, q in [("coldest 1 %", 1), ("coldest 5 %", 5),
                     ("hottest 5 %", 95), ("hottest 1 %", 99)]:
        thr = np.percentile(h.y, q)
        sel = (h.y <= thr) if q <= 50 else (h.y >= thr)
        rows.append({"band": label, "threshold_C": round(float(thr), 2),
                     "n_train": int((sel & is_tr).sum()), "n_test": int((sel & is_te).sum()),
                     "adequate": bool((sel & is_tr).sum() >= 30 and (sel & is_te).sum() >= 30)})
    tab = pd.DataFrame(rows)
    tab.to_csv(cfg.path("products", f"extreme_coverage_{cfg.key}.csv"), index=False)

    if verbose:
        print("temperature extremes in the split:")
        print(tab.to_string(index=False))
        te, p, t = _held_out(h, fit)
        print("\nheld-out error by temperature band:")
        for name, s in [("coldest 5 %", t <= np.percentile(t, 5)),
                        ("middle 90 %", (t > np.percentile(t, 5)) & (t < np.percentile(t, 95))),
                        ("hottest 5 %", t >= np.percentile(t, 95))]:
            if s.sum() < 20:
                continue
            bias = float(np.mean(p[s] - t[s]))
            note = "under-predicts" if bias < -0.3 else "over-predicts" if bias > 0.3 else ""
            print(f"  {name:12s} n={int(s.sum()):7,} MAE {np.mean(np.abs(p[s] - t[s])):.2f} °C "
                  f"bias {bias:+.2f} °C  {note}")
        print("  a cold bias at the hot end is the regression-to-the-mean signature of tree "
              "ensembles; the residual correction restores the observed 70 m value")
    return tab


# ---------------------------------------------------------------- residuals
def residual_signs(cfg, h, fit, verbose: bool = True) -> dict:
    """Is the residual spread symmetric, or does the model lean warm or cold?"""
    te, p, t = _held_out(h, fit)
    res = p - t
    pos, neg = res > 0, res < 0
    imbalance = abs(100 * pos.mean() - 50)

    rows, k_te = [], h.scene_key[te]
    for key in sorted(set(k_te)):
        s = k_te == key
        if s.sum() < 50:
            continue
        rows.append({"date": key[:10], "utc": key[11:], "n_px": int(s.sum()),
                     "bias_C": round(float(res[s].mean()), 2),
                     "pct_positive": round(float(100 * (res[s] > 0).mean()), 1)})
    by_scene = pd.DataFrame(rows).sort_values("bias_C")
    by_scene.to_csv(cfg.path("products", f"residual_bias_by_scene_{cfg.key}.csv"), index=False)

    if verbose:
        print("signed residuals (prediction minus observation, held-out blocks):")
        print(f"  positive {pos.sum():,} ({100 * pos.mean():.1f} %) mean {res[pos].mean():+.2f} °C")
        print(f"  negative {neg.sum():,} ({100 * neg.mean():.1f} %) mean {res[neg].mean():+.2f} °C")
        print(f"  overall bias {res.mean():+.3f} °C · skew {stats.skew(res):+.2f} · "
              f"{'balanced' if imbalance < 5 else 'leans ' + ('warm' if pos.mean() > 0.5 else 'cold')}")
    return {"residuals": res, "by_scene": by_scene, "imbalance_pp": imbalance}


def residual_tails(cfg, h, fit, verbose: bool = True) -> pd.DataFrame:
    """Distribution shape and tail behaviour against a normal reference."""
    _, p, t = _held_out(h, fit)
    res = p - t
    mu, sd = float(res.mean()), float(res.std())
    sk, ku = float(stats.skew(res)), float(stats.kurtosis(res))
    tab = pd.DataFrame([{"percentile": q, "residual_C": round(float(np.percentile(res, q)), 3)}
                        for q in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)])
    tab.to_csv(cfg.path("products", f"residual_tails_{cfg.key}.csv"), index=False)
    if verbose:
        print(f"residual distribution: mean {mu:+.3f} °C · std {sd:.3f} °C · "
              f"skew {sk:+.3f} · excess kurtosis {ku:+.3f}")
        for k in (2, 3, 4):
            obs = 100 * float(np.mean(np.abs(res - mu) > k * sd))
            nrm = 100 * float(2 * (1 - stats.norm.cdf(k)))
            verdict = "heavier" if obs > nrm * 1.3 else "lighter" if obs < nrm * 0.7 else "as expected"
            print(f"  beyond {k}σ: observed {obs:5.2f} % vs normal {nrm:5.2f} %  ({verdict})")
    return tab


def by_landcover(cfg, h, fit, verbose: bool = True, **_ignored) -> pd.DataFrame:
    """Held-out accuracy and bias per land-cover class - one table.

    v2.0 printed this twice (residuals by class, then accuracy by class) and also trained
    class-specific pooled models; with one model per overpass that experiment no longer
    applies, and the two tables are merged here.
    """
    if h.lulc is None:
        print("per-sample land-cover labels unavailable (WorldCover could not be loaded)")
        return pd.DataFrame()
    te, p, t = _held_out(h, fit)
    lc = h.lulc[te]
    rows = []
    for code, cls in WC_CLASSES.items():
        s = lc == code
        if s.sum() < 200:
            continue
        mae, rmse, r2 = metrics(t[s], p[s])
        rows.append({"lulc": cls, "n_test_px": int(s.sum()),
                     "obs_std_C": round(float(t[s].std()), 2),
                     "MAE_C": round(mae, 3), "RMSE_C": round(rmse, 3), "R2": round(r2, 3),
                     "bias_C": round(float((p[s] - t[s]).mean()), 3)})
    tab = pd.DataFrame(rows).sort_values("MAE_C")
    tab.to_csv(cfg.path("products", f"accuracy_by_lulc_{cfg.key}.csv"), index=False)
    if verbose and len(tab):
        print("held-out accuracy by land cover:")
        print(tab.to_string(index=False))
        print("\n  R² tracks each class's own spread (obs_std_C) - compare MAE instead")
    return tab


def sub_pixel_detail(cfg, max_scenes: int = 8, verbose: bool = True) -> pd.DataFrame:
    """How much structure the 10 m product adds beyond the 70 m grid.

    **There is no 10 m thermal ground truth.** What is measured: mass closure of the
    correction, and the standard deviation *within* each 70 m cell. The **detail ratio** -
    that sub-cell spread divided by the scene's total spread - is the most direct single
    measure of whether the output is downscaling or interpolating (a pure upsample is 0).
    """
    tifs = sorted(glob.glob(str(cfg.path("scenes",
                  f"{cfg.name}_*_{cfg.mode}_lst_{cfg.res_fine}m_C.tif"))))[:max_scenes]
    rows = []
    for p10 in tifs:
        p70 = p10.replace(f"_lst_{cfg.res_fine}m_C.tif", f"_lst_{cfg.res_coarse}m_C.tif")
        if not os.path.exists(p70):
            continue
        a70, _, _ = read_raster(p70)
        a10, _, _ = read_raster(p10)
        f = max(1, int(round(a10.shape[0] / a70.shape[0])))
        h2, w2 = (a10.shape[0] // f) * f, (a10.shape[1] // f) * f
        blocks = a10[:h2, :w2].reshape(h2 // f, f, w2 // f, f)
        with np.errstate(invalid="ignore"):
            agg = np.nanmean(blocks, axis=(1, 3))
            sub = np.nanstd(blocks, axis=(1, 3))
        hh, ww = min(a70.shape[0], agg.shape[0]), min(a70.shape[1], agg.shape[1])
        closure = agg[:hh, :ww] - a70[:hh, :ww]
        v = np.isfinite(closure)
        if v.sum() < 50:
            continue
        scene_std = float(np.nanstd(a70[:hh, :ww][np.isfinite(a70[:hh, :ww])]))
        sub_std = float(np.nanmean(sub[np.isfinite(sub)]))
        rows.append({"scene": os.path.basename(p10)[:34],
                     "closure_mean_C": round(float(np.nanmean(closure[v])), 4),
                     "closure_mad_C": round(float(np.nanmedian(np.abs(closure[v]))), 3),
                     "sub70m_std_C": round(sub_std, 3),
                     "scene_std_C": round(scene_std, 2),
                     "detail_ratio": round(sub_std / max(scene_std, 1e-6), 3)})
    tab = pd.DataFrame(rows)
    if len(tab):
        tab.to_csv(cfg.path("products", f"subpixel_detail_{cfg.key}.csv"), index=False)
        if verbose:
            print(tab.to_string(index=False))
            print(f"\ndetail ratio {tab.detail_ratio.mean():.3f} - the share of spatial "
                  "variability that is genuinely sub-pixel (a pure upsample scores 0)")
    return tab


# ---------------------------------------------------------------- time of day
def by_local_hour(cfg, h, fit=None, verbose: bool = True, **_ignored) -> pd.DataFrame:
    """How strongly does the hour of the overpass drive temperature?

    Returns one row per clean overpass with its local hour, median LST and model accuracy.
    :func:`lstpipe.viz.time_of_day` draws the boxplot. This is why ``product_hours``
    restricts the Day composites to 09-16 h: dawn overpasses are a different quantity.
    """
    q = h.quality.copy() if len(h.quality) else pd.DataFrame()
    if not len(q):
        print("no scene-quality table in this harvest - re-run the pipeline")
        return q
    q = q[q.grade.isin(["A", "B"])].copy()
    if fit is not None and len(fit.per_scene):
        acc = fit.per_scene[["date", "utc", "MAE_C", "RMSE_C", "R2"]].copy()
        acc["utc"] = acc.utc.astype(str).str.zfill(4)
        q["utc"] = q.utc.astype(str).str.zfill(4)
        q = q.merge(acc, on=["date", "utc"], how="left")
    q["in_product_hours"] = q.local_hour.apply(
        lambda x: cfg.product_hours is None or cfg.product_hours[0] <= x <= cfg.product_hours[1])
    q.to_csv(cfg.path("products", f"time_of_day_{cfg.key}.csv"), index=False)
    if verbose:
        g = q.groupby("local_hour").agg(scenes=("median_C", "size"),
                                         median_C=("median_C", "median"))
        if "MAE_C" in q:
            g["MAE_C"] = q.groupby("local_hour").MAE_C.median()
        print("clean overpasses by local hour:")
        print(g.round(2).to_string())
        if cfg.product_hours:
            print(f"\ncomposites use {cfg.product_hours[0]:02d}-{cfg.product_hours[1]:02d} h "
                  f"({int(q.in_product_hours.sum())} of {len(q)} clean overpasses)")
    return q


# ---------------------------------------------------------------- scene quality
def scene_quality(cfg, verbose: bool = True) -> pd.DataFrame:
    """Grade table for every overpass considered, with the reason for each rejection."""
    p = cfg.path("products", f"scene_quality_{cfg.key}.csv")
    if not p.exists():
        print("no scene-quality table - run the pipeline first")
        return pd.DataFrame()
    q = pd.read_csv(p)
    inv_p = cfg.path("products", f"inventory_{cfg.key}.csv")
    if inv_p.exists():
        inv = pd.read_csv(inv_p)[["date", "utc", "in_composites", "sharpened_frac", "file_10m"]]
        q["utc"] = q.utc.astype(str).str.zfill(4)
        inv["utc"] = inv.utc.astype(str).str.zfill(4)
        q = q.merge(inv, on=["date", "utc"], how="left")
        keep = inv.loc[inv.in_composites.astype(bool), "file_10m"].tolist()
        with open(cfg.path("products", f"delivery_list_{cfg.key}.txt"), "w") as fh:
            fh.write("\n".join(map(str, keep)))
    q.to_csv(cfg.path("products", f"inventory_graded_{cfg.key}.csv"), index=False)
    if verbose:
        print(f"{cfg.mode}: {len(q)} overpasses considered")
        for g in ("A", "B", "C"):
            s = q.grade == g
            if s.sum():
                print(f"  grade {g}: {s.sum():3d}")
        if "reject_reason" in q:
            r = q.loc[q.grade == "C", "reject_reason"].fillna("").str.split(" ").str[0]
            if len(r):
                print("  rejections by cause: " + ", ".join(f"{k} {v}" for k, v in
                                                       r.value_counts().items()))
    return q


def day_night_deviation(cfg, verbose: bool = True) -> pd.DataFrame:
    """Per-class day-night comparison for *this* study area. Requires both modes."""
    frames = {}
    for m in ("Day", "Night"):
        c = cfg if cfg.mode == m else cfg.with_mode(m)
        p = c.path("products", f"zonal_stats_lulc_{c.key}.csv")
        if p.exists():
            frames[m] = pd.read_csv(p)
    if len(frames) < 2:
        print(f"needs both Day and Night zonal statistics for {cfg.name}")
        return pd.DataFrame()
    d = frames["Day"].groupby("lulc").agg(day_mean=("mean_C", "mean"), day_p95=("p95_C", "mean"))
    n = frames["Night"].groupby("lulc").agg(night_mean=("mean_C", "mean"),
                                            night_p95=("p95_C", "mean"))
    dn = d.join(n, how="inner")
    dn["deviation_C"] = dn.day_mean - dn.night_mean
    dn = dn.sort_values("deviation_C")
    dn.round(3).to_csv(cfg.path("products", f"daynight_deviation_by_lulc_{cfg.name}.csv"))
    if verbose:
        print(dn.round(2).to_string())
    return dn
