"""
Model training, accuracy and persistence.
=========================================

**v2.1: one Random Forest per overpass, trained on absolute land surface temperature in
degrees Celsius.** No anomaly target, no Kelvin anywhere.

Why per overpass. Sentinel-2 cannot see weather, season or hour of day, so a single model
pooled across overpasses cannot know whether a roof is at 30 °C or 55 °C - v2.0 worked
around that by training on the anomaly (pixel minus scene mean). The standard alternative,
and the design of the NASA ARSET ECOSTRESS downscaling method this project follows, is to
fit the relationship *within* each overpass: every scene's model learns its own level and
its own surface-temperature-to-reflectance relationship directly in °C, from that scene's
pixels at 70 m, and is then applied to the same scene's 10 m Sentinel-2 predictors. This is
the scale-invariance assumption behind TsHARP, DisTrad and the data-mining sharpener.

Accuracy is measured on **spatial blocks** (``cv_block_m``, default ~1 km) held out within
each scene, not on random pixels: neighbouring 70 m pixels are strongly correlated, and a
random pixel split would let the model be scored on near-copies of its training data.

What to quote: per-scene MAE and RMSE in °C, and the median per-scene R². A pooled R² across
scenes is not reported, because between-scene level differences would inflate it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(true, pred) -> tuple[float, float, float]:
    """(MAE, RMSE, R²)."""
    return (float(mean_absolute_error(true, pred)),
            float(np.sqrt(mean_squared_error(true, pred))),
            float(r2_score(true, pred)))


def block_split(rc: np.ndarray, block_px: int, test_size: float, seed: int):
    """Train/test indices with whole spatial blocks held out.

    Parameters
    ----------
    rc : (n, 2) int array of row, column on the coarse grid.
    """
    blk = (rc[:, 0] // block_px) * 100_000 + (rc[:, 1] // block_px)
    ub = np.unique(blk)
    rng = np.random.default_rng(seed)
    test_blocks = rng.choice(ub, size=max(1, int(round(test_size * len(ub)))), replace=False)
    te = np.isin(blk, test_blocks)
    return np.where(~te)[0], np.where(te)[0]


def _hour(h, key):
    sd = h.scenes.get(key) if hasattr(h, "scenes") else None
    return getattr(getattr(sd, "scene", sd), "local_hour", None) if sd is not None else None


def _rf(cfg, n_estimators=None):
    return RandomForestRegressor(n_estimators=n_estimators or cfg.rf_estimators,
                                 max_depth=cfg.rf_max_depth,
                                 min_samples_leaf=cfg.rf_min_leaf,
                                 random_state=cfg.seed_model, n_jobs=-1)


@dataclass
class Fit:
    """One model per overpass, with held-out accuracy.

    ``train_idx`` and ``test_idx`` index the pooled harvest arrays, so the diagnostics can
    work across all scenes at once; :meth:`predict` routes each sample to its own scene's
    model.
    """
    models: dict                        # scene key -> RandomForestRegressor
    y: np.ndarray                       # observed LST, °C (the training target)
    train_idx: np.ndarray
    test_idx: np.ndarray
    overall: tuple                      # (MAE, RMSE, median per-scene R²) on held-out blocks
    per_scene: pd.DataFrame
    importance: pd.DataFrame
    units: str = "°C"

    def predict(self, X, keys) -> np.ndarray:
        """Predicted LST in °C. ``keys`` gives the overpass of each row."""
        X = np.asarray(X)
        keys = np.asarray(keys)
        out = np.full(len(X), np.nan)
        for k in np.unique(keys):
            if k in self.models:
                s = keys == k
                out[s] = self.models[k].predict(X[s])
        return out

    def scene_model(self, key):
        return self.models[key]

    def summary(self) -> str:
        return (f"{len(self.models)} per-overpass models · held-out spatial blocks: "
                f"MAE {self.overall[0]:.2f} °C · RMSE {self.overall[1]:.2f} °C · "
                f"median per-scene R² {self.overall[2]:.3f}")


def train(cfg, h, verbose: bool = True, **_ignored) -> Fit:
    """Train one model per overpass on absolute LST (°C) and evaluate on held-out blocks."""
    block_px = max(1, int(round(cfg.cv_block_m / cfg.res_coarse)))
    models, rows, imps = {}, [], []
    tr_all, te_all = [], []
    pred_te = np.full(len(h.y), np.nan)

    keys = [k for k in dict.fromkeys(h.scene_key)]          # preserves order
    for k in keys:
        idx = np.where(h.scene_key == k)[0]
        if len(idx) < cfg.min_train_px:
            continue
        a, b = block_split(h.rc[idx], block_px, cfg.test_size, cfg.seed_split)
        i_tr, i_te = idx[a], idx[b]
        rf = _rf(cfg)
        rf.fit(h.X[i_tr], h.y[i_tr])
        models[k] = rf
        p = rf.predict(h.X[i_te])
        pred_te[i_te] = p
        tr_all.append(i_tr)
        te_all.append(i_te)
        mae, rmse, r2 = metrics(h.y[i_te], p)
        rows.append({"date": k[:10], "utc": k[11:], "mode": cfg.mode,
                     "local_hour": _hour(h, k),
                     "n_train_px": len(i_tr), "n_test_px": len(i_te),
                     "scene_mean_C": round(float(h.y[idx].mean()), 2),
                     "scene_std_C": round(float(h.y[idx].std()), 2),
                     "MAE_C": round(mae, 2), "RMSE_C": round(rmse, 2), "R2": round(r2, 3),
                     "bias_C": round(float((p - h.y[i_te]).mean()), 3)})
        imps.append(rf.feature_importances_)

    if not models:
        raise RuntimeError("no scene had enough clean pixels to train a model")

    tr, te = np.concatenate(tr_all), np.concatenate(te_all)
    per_scene = pd.DataFrame(rows)
    mae_all, rmse_all, _ = metrics(h.y[te], pred_te[te])
    overall = (mae_all, rmse_all, float(per_scene.R2.median()))
    per_scene.to_csv(cfg.path("products", f"accuracy_by_scene_{cfg.key}.csv"), index=False)

    imp = (pd.DataFrame({"feature": list(cfg.features),
                         "importance": np.mean(imps, axis=0),
                         "importance_sd": np.std(imps, axis=0)})
           .sort_values("importance", ascending=False))
    imp.to_csv(cfg.path("products", f"feature_importance_{cfg.key}.csv"), index=False)

    fit = Fit(models=models, y=h.y, train_idx=tr, test_idx=te, overall=overall,
              per_scene=per_scene, importance=imp)
    if verbose:
        print(f"{cfg.mode}: {len(models)} per-overpass models, target LST in °C")
        print(f"  hold-out: {int(cfg.test_size * 100)} % of ~{cfg.cv_block_m} m spatial "
              f"blocks in every scene ({len(te):,} pixels)\n")
        print("=" * 68)
        print(f"ACCURACY ({cfg.mode}, held-out spatial blocks)")
        print(f"  MAE {overall[0]:.2f} °C · RMSE {overall[1]:.2f} °C · "
              f"median per-scene R² {overall[2]:.3f}")
        print(f"  per-scene RMSE {per_scene.RMSE_C.min():.2f}–{per_scene.RMSE_C.max():.2f} °C")
        print("=" * 68)
        print("  read R² beside scene_std_C: a uniform scene offers little to resolve, so a "
              "low R² there is expected - compare MAE instead")
        print("\nfeature importance (mean across scenes):")
        print(imp.round(3).to_string(index=False))
    return fit


def save(cfg, fit: Fit, include_models: bool = False) -> str:
    """Write the run's model metadata; optionally the per-scene forests too.

    Per-overpass forests are cheap to refit from the harvest cache (seconds per scene) and
    large on disk (tens of MB each), so by default only the metadata is written and
    :func:`lstpipe.session.load` refits them. Pass ``include_models=True`` to keep them.
    """
    meta = {"feature_names": list(cfg.features), "n_features": len(cfg.features),
            "run_key": cfg.key, "mode": cfg.mode, "study_area": cfg.name,
            "trained": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "scheme": "one Random Forest per overpass",
            "target": "absolute land surface temperature, degrees Celsius",
            "split": f"{int(cfg.test_size * 100)} % of {cfg.cv_block_m} m spatial blocks "
                     f"held out per scene",
            "n_scenes": len(fit.models), "scenes": sorted(fit.models),
            "MAE_C": round(fit.overall[0], 3), "RMSE_C": round(fit.overall[1], 3),
            "median_scene_R2": round(fit.overall[2], 4),
            "params": {"n_estimators": cfg.rf_estimators, "max_depth": cfg.rf_max_depth,
                       "min_samples_leaf": cfg.rf_min_leaf}}
    with open(cfg.cache("model_meta"), "w") as fh:
        json.dump(meta, fh, indent=2)
    if include_models:
        joblib.dump(fit.models, cfg.cache("model"), compress=3)
        print(f"models saved: {cfg.cache('model').name}")
    print(f"model metadata saved: {cfg.cache('model_meta').name}")
    return str(cfg.cache("model_meta"))


def load(cfg, path: str | None = None):
    """Load saved per-scene models and verify they belong to this run."""
    path = path or str(cfg.cache("model"))
    with open(cfg.cache("model_meta")) as fh:
        meta = json.load(fh)
    if meta.get("run_key") and meta["run_key"] != cfg.key:
        raise ValueError(f"model belongs to {meta['run_key']}, not {cfg.key}: {path}")
    if tuple(meta["feature_names"]) != tuple(cfg.features):
        raise ValueError("feature mismatch between the saved model and this configuration:\n"
                         f"  model:  {meta['feature_names']}\n  config: {list(cfg.features)}")
    if "scheme" not in meta:
        raise ValueError("this model was written by v2.0 (pooled anomaly model, Kelvin). "
                         "It is not compatible with v2.1 - re-run 01_pipeline.")
    models = joblib.load(path)
    print(f"loaded {len(models)} per-overpass models trained {meta['trained']}")
    return models, meta
