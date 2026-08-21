"""
Model training, accuracy and persistence.
=========================================

One Random Forest is trained across all overpasses, predicting the **anomaly** - each
pixel's departure from its own scene mean - rather than absolute temperature.

Why the anomaly. Individual scenes over a dense tropical city are internally uniform, often
varying by under 1 K across the whole area, while the *level* differs by tens of degrees
between overpasses. Asking the model for absolute temperature forces it to infer that level
from optical reflectance alone, which is impossible: weather, season and hour of day leave
no signature in Sentinel-2. Being 6 K wrong on a scene with 1 K of internal spread drives
R² to around -40 even when the spatial pattern is entirely correct. The level is already
available from ECOSTRESS itself and is restored by the residual correction at map time, so
the model is trained only on what it can actually learn.

Two accuracy figures are reported. **Pattern** accuracy, in anomaly space, is the honest
measure of downscaling skill and the one to quote. **Absolute** accuracy, after the scene
level is added back, carries *identical* MAE and RMSE and differs only because the variance
denominator grows - it is not additional skill.
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
from sklearn.model_selection import GroupKFold, train_test_split


def metrics(true, pred) -> tuple[float, float, float]:
    """(MAE, RMSE, R²)."""
    return (float(mean_absolute_error(true, pred)),
            float(np.sqrt(mean_squared_error(true, pred))),
            float(r2_score(true, pred)))


def temperature_strata(y, n_strata: int = 10) -> np.ndarray:
    """Quantile bins used to stratify the split.

    A purely random split leaves representation of the rare hot tail to chance;
    stratifying guarantees hot and cold pixels appear proportionally on both sides.
    Bins too small to stratify are merged into their neighbour.
    """
    edges = np.unique(np.quantile(y, np.linspace(0, 1, n_strata + 1)))
    s = np.clip(np.digitize(y, edges[1:-1]), 0, max(len(edges) - 2, 0))
    for b in np.where(np.bincount(s, minlength=max(len(edges) - 1, 1)) < 2)[0]:
        s[s == b] = max(b - 1, 0)
    return s


@dataclass
class Fit:
    """A trained model with its accuracy and the indices behind it."""
    rf: RandomForestRegressor
    y_anom: np.ndarray
    scene_mean: dict
    train_idx: np.ndarray
    test_idx: np.ndarray
    pattern: tuple                      # (MAE, RMSE, R²) in anomaly space
    absolute: tuple                     # (MAE, RMSE, R²) after restoring scene level
    per_scene: pd.DataFrame
    importance: pd.DataFrame
    cv: pd.DataFrame = field(default_factory=pd.DataFrame)

    def predict(self, X) -> np.ndarray:
        """Predicted anomaly, in kelvin."""
        return self.rf.predict(X)

    def summary(self) -> str:
        return (f"pattern  MAE {self.pattern[0]:.2f} K · RMSE {self.pattern[1]:.2f} K · "
                f"R² {self.pattern[2]:.3f}\n"
                f"absolute MAE {self.absolute[0]:.2f} K · RMSE {self.absolute[1]:.2f} K · "
                f"R² {self.absolute[2]:.3f}   (same error, larger denominator)")


def train(cfg, h, cross_validate: bool = True, verbose: bool = True) -> Fit:
    """Train one model on the pooled samples and evaluate it."""
    scene_mean = {k: h.y[h.scene_key == k].mean() for k in set(h.scene_key)}
    sm = np.array([scene_mean[k] for k in h.scene_key])
    y_anom = h.y - sm

    strata = temperature_strata(y_anom, cfg.n_strata)
    idx = np.arange(len(y_anom))
    i_tr, i_te = train_test_split(idx, test_size=cfg.test_size,
                                  random_state=cfg.seed_split, stratify=strata)

    rf = RandomForestRegressor(n_estimators=cfg.rf_estimators, max_depth=cfg.rf_max_depth,
                               min_samples_leaf=cfg.rf_min_leaf,
                               random_state=cfg.seed_model, n_jobs=-1)
    rf.fit(h.X[i_tr], y_anom[i_tr])

    pa, ta = rf.predict(h.X[i_te]), y_anom[i_te]
    pb, tb = pa + sm[i_te], h.y[i_te]
    pattern, absolute = metrics(ta, pa), metrics(tb, pb)

    if verbose:
        levels = np.array(list(scene_mean.values())) - 273.15
        print(f"{cfg.mode}: anomaly target | scene levels {levels.min():.1f}–"
              f"{levels.max():.1f} °C | anomaly std {y_anom.std():.2f} K")
        print(f"  train {len(i_tr):,} · test {len(i_te):,} "
              f"({int((1 - cfg.test_size) * 100)}/{int(cfg.test_size * 100)}, "
              f"stratified by temperature)\n")
        print("=" * 68)
        print(f"ACCURACY ({cfg.mode}, {h.n_scenes} scenes pooled, held-out "
              f"{int(cfg.test_size * 100)} %)")
        print(f"  SPATIAL PATTERN (anomaly)    MAE {pattern[0]:.2f} K  "
              f"RMSE {pattern[1]:.2f} K  R² {pattern[2]:.3f}")
        print(f"  ABSOLUTE LST (reconstructed) MAE {absolute[0]:.2f} K  "
              f"RMSE {absolute[1]:.2f} K  R² {absolute[2]:.3f}")
        print("=" * 68)

    # per-scene accuracy, with each scene's own spread for context
    rows, k_te = [], h.scene_key[i_te]
    for key in sorted(set(k_te)):
        s = k_te == key
        if s.sum() < 50:
            continue
        ma, ra, r2a = metrics(ta[s], pa[s])
        mb, rb, _ = metrics(tb[s], pb[s])
        rows.append({"date": key[:10], "utc": key[11:], "mode": cfg.mode,
                     "n_test_px": int(s.sum()),
                     "scene_mean_C": round(float(scene_mean[key] - 273.15), 2),
                     "scene_std_K": round(float(h.y[h.scene_key == key].std()), 2),
                     "MAE_pattern_K": round(ma, 2), "RMSE_pattern_K": round(ra, 2),
                     "R2_pattern": round(r2a, 3),
                     "MAE_abs_K": round(mb, 2), "RMSE_abs_K": round(rb, 2)})
    per_scene = pd.DataFrame(rows)
    per_scene.to_csv(cfg.path("products", f"accuracy_by_scene_{cfg.mode}.csv"), index=False)
    if verbose and len(per_scene):
        print(f"\nper-scene: median pattern R² {per_scene.R2_pattern.median():.3f} · "
              f"RMSE {per_scene.RMSE_pattern_K.median():.2f} K")
        print("  read R² beside scene_std_K - a uniform scene offers little to resolve, "
              "so a low value there is expected")

    cv = pd.DataFrame()
    if cross_validate and len(set(h.scene_key)) >= 3:
        gkf = GroupKFold(n_splits=min(5, len(set(h.scene_key))))
        folds = []
        for f, (a, b) in enumerate(gkf.split(h.X, y_anom, h.scene_key), 1):
            m_ = RandomForestRegressor(n_estimators=60, max_depth=cfg.rf_max_depth,
                                       min_samples_leaf=cfg.rf_min_leaf,
                                       random_state=cfg.seed_model, n_jobs=-1)
            m_.fit(h.X[a], y_anom[a])
            ma, ra, r2a = metrics(y_anom[b], m_.predict(h.X[b]))
            folds.append({"fold": f, "n_test": len(b), "MAE_K": round(ma, 2),
                          "RMSE_K": round(ra, 2), "R2": round(r2a, 3)})
        cv = pd.DataFrame(folds)
        cv.to_csv(cfg.path("products", f"cv_scene_grouped_{cfg.mode}.csv"), index=False)
        if verbose:
            print(f"\nleave-scene-out CV ({len(cv)} folds): mean R² {cv.R2.mean():.3f} · "
                  f"RMSE {cv.RMSE_K.mean():.2f} K")

    imp = (pd.DataFrame({"feature": list(cfg.features)[:rf.n_features_in_],
                         "importance": rf.feature_importances_})
           .sort_values("importance", ascending=False))
    imp.to_csv(cfg.path("products", f"feature_importance_{cfg.mode}.csv"), index=False)
    if verbose:
        print("\nfeature importance:")
        print(imp.to_string(index=False))

    return Fit(rf=rf, y_anom=y_anom, scene_mean=scene_mean, train_idx=i_tr, test_idx=i_te,
               pattern=pattern, absolute=absolute, per_scene=per_scene,
               importance=imp, cv=cv)


def save(cfg, fit: Fit) -> str:
    """Persist the model with the metadata needed to use it safely later.

    A fitted Random Forest records only how many columns it expects, not their names or
    order, so the feature list is stored alongside it. Loading a model without checking
    that list against the current configuration risks predictions that are silently wrong
    rather than obviously broken.
    """
    path = cfg.cache("model")
    joblib.dump(fit.rf, path, compress=3)
    meta = {"feature_names": list(cfg.features),
            "n_features": int(fit.rf.n_features_in_),
            "run_key": cfg.key, "mode": cfg.mode, "study_area": cfg.name,
            "trained": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "target": "anomaly (pixel minus scene mean), kelvin",
            "split": f"{int((1 - cfg.test_size) * 100)}/{int(cfg.test_size * 100)} "
                     f"stratified by temperature quantile",
            "pattern_MAE_K": round(fit.pattern[0], 3),
            "pattern_R2": round(fit.pattern[2], 4),
            "params": {"n_estimators": cfg.rf_estimators, "max_depth": cfg.rf_max_depth,
                       "min_samples_leaf": cfg.rf_min_leaf}}
    with open(cfg.cache("model_meta"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"model saved: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return str(path)


def load(cfg, path: str | None = None):
    """Load a saved model and verify it belongs to this run and expects these features."""
    path = path or str(cfg.cache("model"))
    rf = joblib.load(path)
    meta_path = (cfg.cache("model_meta") if path == str(cfg.cache("model"))
                 else path.replace(".joblib", "_meta.json"))
    with open(meta_path) as fh:
        meta = json.load(fh)
    if meta.get("run_key") and meta["run_key"] != cfg.key:
        raise ValueError(f"model belongs to {meta['run_key']}, not {cfg.key}: {path}")
    if tuple(meta["feature_names"]) != tuple(cfg.features):
        raise ValueError("feature mismatch between the saved model and this configuration:\n"
                         f"  model:  {meta['feature_names']}\n  config: {list(cfg.features)}")
    print(f"loaded model trained {meta['trained']} ({meta['n_features']} features)")
    return rf, meta
