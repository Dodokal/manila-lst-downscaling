"""
Validation analyses.
====================

Each function answers one question about the model and returns a table, so results can be
compared across runs rather than only read off a figure. All of them are optional and
independent - run the ones a given review calls for.

A recurring point of interpretation: **R² per group tracks that group's own spread.** A
uniform scene or a homogeneous land-cover class offers little to resolve, so a low or
negative R² there is expected rather than diagnostic. Compare MAE across groups instead.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .config import WC_CLASSES
from .model import metrics
from .products import read_raster


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
        rows.append({"band": label, "threshold_C": round(float(thr - 273.15), 2),
                     "n_train": int((sel & is_tr).sum()), "n_test": int((sel & is_te).sum()),
                     "adequate": bool((sel & is_tr).sum() >= 30 and (sel & is_te).sum() >= 30)})
    tab = pd.DataFrame(rows)
    tab.to_csv(cfg.path("products", f"extreme_coverage_{cfg.mode}.csv"), index=False)

    if verbose:
        print("temperature extremes in the split:")
        print(tab.to_string(index=False))
        pa, ta = fit.predict(h.X[fit.test_idx]), fit.y_anom[fit.test_idx]
        abs_te = h.y[fit.test_idx]
        print("\nheld-out error by temperature band (anomaly space):")
        for name, s in [("coldest 5 %", abs_te <= np.percentile(abs_te, 5)),
                        ("middle 90 %", (abs_te > np.percentile(abs_te, 5)) &
                                        (abs_te < np.percentile(abs_te, 95))),
                        ("hottest 5 %", abs_te >= np.percentile(abs_te, 95))]:
            if s.sum() < 20:
                continue
            bias = float(np.mean(pa[s] - ta[s]))
            note = ("under-predicts" if bias < -0.3 else
                    "over-predicts" if bias > 0.3 else "")
            print(f"  {name:12s} n={int(s.sum()):7,} MAE {np.mean(np.abs(pa[s] - ta[s])):.2f} K "
                  f"bias {bias:+.2f} K  {note}")
        print("  a cold bias at the hot end is the regression-to-the-mean signature of "
              "tree ensembles")
    return tab


def composition(cfg, h, fit, grids=None, verbose: bool = True) -> dict:
    """Sample composition by temperature category and by land cover."""
    is_tr = np.zeros(len(h.y), bool); is_tr[fit.train_idx] = True
    is_te = np.zeros(len(h.y), bool); is_te[fit.test_idx] = True

    q = np.percentile(h.y, [0, 10, 25, 50, 75, 90, 100])
    labels = ["p0-10", "p10-25", "p25-50", "p50-75", "p75-90", "p90-100"]
    cat = np.clip(np.digitize(h.y, q[1:-1]), 0, len(labels) - 1)
    rows = []
    for i, name in enumerate(labels):
        s = cat == i
        rows.append({"category": name,
                     "range_C": f"{h.y[s].min() - 273.15:.1f} to {h.y[s].max() - 273.15:.1f}",
                     "n_train": int((s & is_tr).sum()), "n_test": int((s & is_te).sum()),
                     "share_pct": round(100 * s.mean(), 1)})
    by_lst = pd.DataFrame(rows)
    by_lst.to_csv(cfg.path("products", f"composition_by_LST_{cfg.mode}.csv"), index=False)

    by_lulc = pd.DataFrame()
    if h.lulc is not None:
        rows = []
        area = grids.wc_coarse if grids is not None and grids.has_landcover else None
        for code, cls in WC_CLASSES.items():
            s = h.lulc == code
            if s.sum() == 0:
                continue
            r = {"lulc": cls, "n_train": int((s & is_tr).sum()),
                 "n_test": int((s & is_te).sum()),
                 "sample_share_pct": round(100 * s.mean(), 2),
                 "mean_LST_C": round(float(h.y[s].mean() - 273.15), 2)}
            if area is not None:
                r["area_share_pct"] = round(
                    100 * float((area == code).sum()) / max(np.isfinite(area).sum(), 1), 2)
            rows.append(r)
        by_lulc = pd.DataFrame(rows).sort_values("sample_share_pct", ascending=False)
        by_lulc.to_csv(cfg.path("products", f"composition_by_LULC_{cfg.mode}.csv"), index=False)

    if verbose:
        print("samples by temperature category:")
        print(by_lst.to_string(index=False))
        if len(by_lulc):
            print("\nsamples by land cover:")
            print(by_lulc.to_string(index=False))
    return {"by_lst": by_lst, "by_lulc": by_lulc}


# ---------------------------------------------------------------- residuals
def residual_signs(cfg, h, fit, verbose: bool = True) -> dict:
    """Is the residual spread symmetric, or does the model lean warm or cold?

    A systematic lean means a whole class of pixels is mis-stated, and that bias propagates
    into every map.
    """
    res = fit.predict(h.X[fit.test_idx]) - fit.y_anom[fit.test_idx]
    pos, neg = res > 0, res < 0
    imbalance = abs(100 * pos.mean() - 50)

    rows, k_te = [], h.scene_key[fit.test_idx]
    for key in sorted(set(k_te)):
        s = k_te == key
        if s.sum() < 50:
            continue
        rows.append({"date": key[:10], "utc": key[11:], "n_px": int(s.sum()),
                     "bias_K": round(float(res[s].mean()), 2),
                     "pct_positive": round(float(100 * (res[s] > 0).mean()), 1)})
    by_scene = pd.DataFrame(rows).sort_values("bias_K")
    by_scene.to_csv(cfg.path("products", f"residual_bias_by_scene_{cfg.mode}.csv"), index=False)

    if verbose:
        print("signed residuals (prediction minus observation, anomaly space):")
        print(f"  positive {pos.sum():,} ({100 * pos.mean():.1f} %) mean {res[pos].mean():+.2f} K")
        print(f"  negative {neg.sum():,} ({100 * neg.mean():.1f} %) mean {res[neg].mean():+.2f} K")
        print(f"  overall bias {res.mean():+.3f} K · skew {stats.skew(res):+.2f} · "
              f"{'balanced' if imbalance < 5 else 'leans ' + ('warm' if pos.mean() > 0.5 else 'cold')}")
        if len(by_scene):
            print(f"\nper-scene bias, most negative and most positive:")
            print(pd.concat([by_scene.head(3), by_scene.tail(3)]).to_string(index=False))
    return {"residuals": res, "by_scene": by_scene, "imbalance_pp": imbalance}


def residual_tails(cfg, h, fit, verbose: bool = True) -> pd.DataFrame:
    """Distribution shape and tail behaviour against a normal reference."""
    res = fit.predict(h.X[fit.test_idx]) - fit.y_anom[fit.test_idx]
    mu, sd = float(res.mean()), float(res.std())
    sk, ku = float(stats.skew(res)), float(stats.kurtosis(res))

    rows = [{"percentile": q, "residual_K": round(float(np.percentile(res, q)), 3)}
            for q in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)]
    tab = pd.DataFrame(rows)
    tab.to_csv(cfg.path("products", f"residual_tails_{cfg.mode}.csv"), index=False)

    if verbose:
        print(f"residual distribution: mean {mu:+.3f} K · std {sd:.3f} K · "
              f"skew {sk:+.3f} · excess kurtosis {ku:+.3f}")
        print(f"  {'heavy-tailed' if ku > 1 else 'near-normal'} tails, "
              f"{'right-skewed' if sk > 0.2 else 'left-skewed' if sk < -0.2 else 'symmetric'}")
        for k in (2, 3, 4):
            obs = 100 * float(np.mean(np.abs(res - mu) > k * sd))
            nrm = 100 * float(2 * (1 - stats.norm.cdf(k)))
            verdict = "heavier" if obs > nrm * 1.3 else "lighter" if obs < nrm * 0.7 else "as expected"
            print(f"  beyond {k}σ: observed {obs:5.2f} % vs normal {nrm:5.2f} %  ({verdict})")
        print("\n" + tab.to_string(index=False))
    return tab


def residual_by_landcover(cfg, h, fit, verbose: bool = True) -> pd.DataFrame:
    """Residual statistics per land-cover class.

    A class-dependent bias means the model systematically mis-states that surface type.
    """
    if h.lulc is None:
        print("per-sample land-cover labels unavailable")
        return pd.DataFrame()

    res = fit.predict(h.X[fit.test_idx]) - fit.y_anom[fit.test_idx]
    lulc_te = h.lulc[fit.test_idx]
    rows = []
    for code, cls in WC_CLASSES.items():
        s = lulc_te == code
        if s.sum() < 100:
            continue
        r = res[s]
        rows.append({"lulc": cls, "n_px": int(s.sum()),
                     "bias_K": round(float(r.mean()), 3),
                     "MAE_K": round(float(np.abs(r).mean()), 3),
                     "RMSE_K": round(float(np.sqrt((r ** 2).mean())), 3),
                     "pct_positive": round(float(100 * (r > 0).mean()), 1),
                     "p05_K": round(float(np.percentile(r, 5)), 2),
                     "p95_K": round(float(np.percentile(r, 95)), 2)})
    tab = pd.DataFrame(rows).sort_values("bias_K")
    tab.to_csv(cfg.path("products", f"residual_by_lulc_{cfg.mode}.csv"), index=False)
    if verbose and len(tab):
        print(tab.to_string(index=False))
        print(f"\n  most under-predicted: {tab.lulc.iloc[0]} ({tab.bias_K.iloc[0]:+.2f} K) · "
              f"most over-predicted: {tab.lulc.iloc[-1]} ({tab.bias_K.iloc[-1]:+.2f} K)")
    return tab


def sub_pixel_detail(cfg, max_scenes: int = 8, verbose: bool = True) -> pd.DataFrame:
    """How much structure the 10 m product adds beyond the 70 m grid.

    **There is no 10 m thermal ground truth.** ECOSTRESS observes at 70 m, so a true 10 m
    residual cannot be computed and any claim otherwise would be circular. What is measured
    here instead: mass closure of the correction, and the standard deviation *within* each
    70 m cell, which is precisely the detail the product claims to add.

    The **detail ratio** - that sub-cell spread divided by the scene's total spread - is
    the most direct single measure of whether the output is downscaling or interpolating.
    A pure upsample scores zero.
    """
    tifs = sorted(glob.glob(str(cfg.path("scenes",
                  f"*_{cfg.mode}_lst_{cfg.res_fine}m_C.tif"))))[:max_scenes]
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
                     "closure_mean_K": round(float(np.nanmean(closure[v])), 4),
                     "sub70m_std_K": round(sub_std, 3),
                     "scene_std_K": round(scene_std, 2),
                     "detail_ratio": round(sub_std / max(scene_std, 1e-6), 3)})
    tab = pd.DataFrame(rows)
    if len(tab):
        tab.to_csv(cfg.path("products", f"subpixel_detail_{cfg.mode}.csv"), index=False)
        if verbose:
            print(tab.to_string(index=False))
            worst = tab.closure_mean_K.abs().max()
            print(f"\nmass closure worst |mean| {worst:.4f} K -> "
                  f"{'conserved' if worst < 0.01 else 'check the correction'}")
            print(f"detail ratio {tab.detail_ratio.mean():.3f} - the share of spatial "
                  "variability that is genuinely sub-pixel (a pure upsample scores 0)")
    return tab


# ---------------------------------------------------------------- stratified performance
def by_landcover(cfg, h, fit, try_class_models: bool = True,
                 min_px: int = 2000, verbose: bool = True) -> dict:
    """Accuracy per land-cover class, and whether class-specific models would help."""
    if h.lulc is None:
        print("per-sample land-cover labels unavailable")
        return {}

    pred, true = fit.predict(h.X[fit.test_idx]), fit.y_anom[fit.test_idx]
    lulc_te = h.lulc[fit.test_idx]
    rows = []
    for code, cls in WC_CLASSES.items():
        s = lulc_te == code
        if s.sum() < 200:
            continue
        mae, rmse, r2 = metrics(true[s], pred[s])
        rows.append({"lulc": cls, "n_test_px": int(s.sum()),
                     "obs_std_K": round(float(true[s].std()), 2),
                     "MAE_K": round(mae, 3), "RMSE_K": round(rmse, 3), "R2": round(r2, 3),
                     "bias_K": round(float((pred[s] - true[s]).mean()), 3)})
    per_class = pd.DataFrame(rows).sort_values("R2", ascending=False)
    per_class.to_csv(cfg.path("products", f"accuracy_by_lulc_{cfg.mode}.csv"), index=False)

    gains = pd.DataFrame()
    if try_class_models:
        rows = []
        for code, cls in WC_CLASSES.items():
            s = h.lulc == code
            if s.sum() < min_px:
                continue
            Xc, yc = h.X[s], fit.y_anom[s]
            i1, i2 = train_test_split(np.arange(len(yc)), test_size=cfg.test_size,
                                      random_state=cfg.seed_split)
            mc = RandomForestRegressor(n_estimators=80, max_depth=cfg.rf_max_depth,
                                       min_samples_leaf=cfg.rf_min_leaf,
                                       random_state=cfg.seed_model, n_jobs=-1)
            mc.fit(Xc[i1], yc[i1])
            c_mae, _, c_r2 = metrics(yc[i2], mc.predict(Xc[i2]))
            p_mae, _, p_r2 = metrics(yc[i2], fit.predict(Xc[i2]))
            rows.append({"lulc": cls, "n_px": int(s.sum()),
                         "pooled_MAE_K": round(p_mae, 3), "class_MAE_K": round(c_mae, 3),
                         "pooled_R2": round(p_r2, 3), "class_R2": round(c_r2, 3),
                         "R2_gain": round(c_r2 - p_r2, 3)})
        gains = pd.DataFrame(rows).sort_values("R2_gain", ascending=False)
        gains.to_csv(cfg.path("products", f"class_model_gain_{cfg.mode}.csv"), index=False)

    if verbose:
        print("held-out accuracy by land cover (anomaly space):")
        print(per_class.to_string(index=False))
        print("\n  R² tracks each class's own spread (obs_std_K) - compare MAE instead")
        if len(gains):
            print("\nwould class-specific models help?")
            print(gains.to_string(index=False))
            g = float(gains.R2_gain.mean())
            print(f"  mean gain {g:+.3f} -> " +
                  ("worth adopting" if g > 0.03 else
                   "one pooled model is adequate; the covariates already encode land cover"))
    return {"per_class": per_class, "class_models": gains}


def by_local_hour(cfg, h, fit, bands=None, verbose: bool = True) -> dict:
    """Does the spatial pattern depend on time of day, not just the level?

    The acquisition window spans several hours, and scene level varies widely across it.
    If band-specific models outperform a single all-window model, the pattern itself is
    time-dependent - surfaces warming unevenly in the morning behave differently from
    surfaces cooling in the afternoon.
    """
    if bands is None:
        bands = ([("morning", 6, 9), ("midday", 9, 15), ("afternoon", 15, 19)]
                 if cfg.mode == "Day" else
                 [("evening", 18, 21), ("late_night", 21, 24), ("pre_dawn", 0, 6)])

    hour_of = {k: sd.scene.local_hour for k, sd in h.scenes.items()}
    def band_of(hour):
        for name, a, b in bands:
            if a <= hour < b:
                return name
        return "other"
    band_key = {k: band_of(v) for k, v in hour_of.items()}
    band_all = np.array([band_key.get(k, "other") for k in h.scene_key], dtype=object)

    rows, models = [], {}
    for name, a, b in bands:
        s = band_all == name
        n_scenes = len({k for k, v in band_key.items() if v == name})
        if s.sum() < 500 or n_scenes < 3:
            continue
        Xb, yb = h.X[s], fit.y_anom[s]
        p_mae, _, p_r2 = metrics(yb, fit.predict(Xb))
        strata = None
        try:
            from .model import temperature_strata
            strata = temperature_strata(yb, cfg.n_strata)
            i1, i2 = train_test_split(np.arange(len(yb)), test_size=cfg.test_size,
                                      random_state=cfg.seed_split, stratify=strata)
        except ValueError:
            i1, i2 = train_test_split(np.arange(len(yb)), test_size=cfg.test_size,
                                      random_state=cfg.seed_split)
        mb = RandomForestRegressor(n_estimators=cfg.rf_estimators, max_depth=cfg.rf_max_depth,
                                   min_samples_leaf=cfg.rf_min_leaf,
                                   random_state=cfg.seed_model, n_jobs=-1)
        mb.fit(Xb[i1], yb[i1])
        b_mae, _, b_r2 = metrics(yb[i2], mb.predict(Xb[i2]))
        models[name] = mb
        rows.append({"band": name, "local_hours": f"{a:02d}-{b:02d}", "scenes": n_scenes,
                     "samples": int(s.sum()),
                     "pooled_MAE_K": round(p_mae, 2), "pooled_R2": round(p_r2, 3),
                     "band_MAE_K": round(b_mae, 2), "band_R2": round(b_r2, 3),
                     "R2_gain": round(b_r2 - p_r2, 3)})

    tab = pd.DataFrame(rows)
    if len(tab):
        tab.to_csv(cfg.path("products", f"localhour_bands_{cfg.mode}.csv"), index=False)
        if verbose:
            print(tab.to_string(index=False))
            g = float(tab.R2_gain.mean())
            print(f"\nmean R² gain from stratifying: {g:+.3f}")
            if g > 0.03:
                print("  band-specific models are measurably better - the spatial pattern "
                      "itself depends on time of day, not only the temperature level")
            else:
                print("  stratifying does not change skill, but still fixes reporting: "
                      "quote accuracy per band and build exceedance statistics within one band")
    return {"bands": tab, "models": models, "band_of_scene": band_key}


def scene_quality(cfg, good: float = 0.70, poor: float = 0.40,
                  verbose: bool = True) -> pd.DataFrame:
    """Grade each map by how much of it was genuinely downscaled.

    ``sharpened_frac`` is the share where optical imagery existed and the model drove the
    detail. The remainder was filled from upsampled ECOSTRESS - real temperature, but at
    70 m effective resolution, which is what makes some maps look blocky.
    """
    path = cfg.path("products", f"inventory_{cfg.mode}.csv")
    if not path.exists():
        print("no inventory found - run per_scene_maps first")
        return pd.DataFrame()
    inv = pd.read_csv(path)
    if "sharpened_frac" not in inv:
        print("inventory has no sharpened_frac column")
        return inv

    inv["quality"] = pd.cut(inv.sharpened_frac, [-0.01, poor, good, 1.01],
                            labels=["COARSE", "PARTIAL", "GOOD"])
    inv = inv.sort_values("sharpened_frac", ascending=False)
    inv.to_csv(cfg.path("products", f"inventory_graded_{cfg.mode}.csv"), index=False)
    keep = inv.loc[inv.quality != "COARSE", "file_10m"].tolist()
    with open(cfg.path("products", f"delivery_list_{cfg.mode}.txt"), "w") as fh:
        fh.write("\n".join(map(str, keep)))

    if verbose:
        print(f"{cfg.mode}: {len(inv)} scenes graded")
        for g in ("GOOD", "PARTIAL", "COARSE"):
            s = inv.quality == g
            if s.sum():
                print(f"  {g:8s} {s.sum():3d} ({100 * s.mean():4.1f} %) "
                      f"sharpened {inv.sharpened_frac[s].min():.2f}–{inv.sharpened_frac[s].max():.2f}")
        print(f"  {len(keep)} cleared for delivery")
        if "s2_gap_days" in inv and inv.s2_gap_days.notna().sum() > 5:
            r = float(np.corrcoef(inv.s2_gap_days.fillna(inv.s2_gap_days.median()),
                                  inv.sharpened_frac)[0, 1])
            print(f"  corr(optical gap, sharpened) {r:+.2f} -> " +
                  ("widen s2_window_days" if r < -0.3 else
                   "driven by cloud on the day itself, not by the pairing window"))
    return inv


def day_night_deviation(cfg, verbose: bool = True) -> pd.DataFrame:
    """Per-class day-night comparison. Requires both products to exist."""
    frames = {}
    for m in ("Day", "Night"):
        p = cfg.path("products", f"zonal_stats_lulc_{m}.csv")
        if p.exists():
            frames[m] = pd.read_csv(p)
    if len(frames) < 2:
        print("needs both Day and Night zonal statistics")
        return pd.DataFrame()

    d = frames["Day"].groupby("lulc").agg(day_mean=("mean_C", "mean"),
                                          day_p95=("p95_C", "mean"))
    n = frames["Night"].groupby("lulc").agg(night_mean=("mean_C", "mean"),
                                            night_p95=("p95_C", "mean"))
    dn = d.join(n, how="inner")
    dn["deviation_K"] = dn.day_mean - dn.night_mean
    dn = dn.sort_values("deviation_K")
    dn.round(3).to_csv(cfg.path("products", "daynight_deviation_by_lulc.csv"))

    if verbose:
        print(dn.round(2).to_string())
        print(f"\n  {dn.index[0]} cools least overnight ({dn.deviation_K.iloc[0]:.1f} K), "
              f"{dn.index[-1]} cools most ({dn.deviation_K.iloc[-1]:.1f} K)")
        print("  the class with the warmest nights and smallest range is where heat "
              "accumulates without relief")
    return dn
