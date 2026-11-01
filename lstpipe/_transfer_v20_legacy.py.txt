"""
Transferability testing.
========================

Applies a trained model, unchanged, to a different study area and scores it against real
observations there. One function covers every test city; the interpretation depends on
which city and why it was chosen.

**Two tests read together.** A cross-climate failure on its own is ambiguous - it could
mean the environment was wrong, or that the method does not generalise at all. Pairing it
with a *near-transfer control*, a city that is climatically and topographically almost
identical to the training area, distinguishes the two:

===========================  ==========================  ============================
Cross-climate (Addis)        Near-transfer (Valenzuela)   Conclusion
===========================  ==========================  ============================
fails                        transfers                    environmental mismatch; the
                                                          method is portable within a
                                                          climate zone
fails                        fails                        over-fitted to the training
                                                          area itself; retrain per city
transfers                    transfers                    the model generalises broadly
===========================  ==========================  ============================

Always read the feature-overlap table before the accuracy figures. A Random Forest cannot
extrapolate: a predictor whose values fall outside the trained range is clamped to the edge
of that range, and the resulting error says more about coverage than about the method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from . import acquire, covariates as cov, harvest as hv
from .model import metrics


@dataclass
class TransferResult:
    city: str
    as_is: tuple                 # (MAE, RMSE, R²) - source model unchanged
    without_dem: tuple           # source model retrained without elevation
    local: tuple                 # model fitted on the target city
    home: tuple                  # source model on its own held-out data, for scale
    overlap: pd.DataFrame
    n_samples: int
    n_scenes: int

    @property
    def retention(self) -> float:
        """Share of home skill retained, as a percentage."""
        return (self.as_is[2] / self.home[2] * 100) if self.home[2] > 0 else float("nan")

    def table(self, mode: str) -> pd.DataFrame:
        rows = [("home (source city)", self.home), ("model as-is", self.as_is),
                ("model without elevation", self.without_dem), ("retrained locally", self.local)]
        return pd.DataFrame([{"mode": mode, "city": self.city, "test": n,
                              "MAE_K": round(v[0], 2), "RMSE_K": round(v[1], 2),
                              "R2": round(v[2], 3)} for n, v in rows])

    def verdict(self) -> str:
        if self.as_is[2] > 0 and self.as_is[2] >= 0.5 * max(self.local[2], 1e-9):
            return ("TRANSFERS - the model works in this city without retraining, so the "
                    "pipeline is portable to comparable environments.")
        if self.as_is[2] > 0:
            return ("PARTIAL - positive skill but below the local ceiling. Usable for a "
                    "first pass; retrain locally for a final product.")
        return ("DOES NOT TRANSFER - check the feature overlap first. A predictor outside "
                "the trained range explains this without implicating the method.")


def evaluate(cfg_target, source_fit, source_X, source_y_anom, source_test_idx,
             since: str | None = None, rebuild: bool = False,
             verbose: bool = True) -> TransferResult:
    """Run a full transfer test against ``cfg_target``.

    Parameters
    ----------
    cfg_target : Config
        The city to test. Use ``Config.valenzuela()`` or ``Config.addis()``.
    source_fit : model.Fit
        The trained model from the source city.
    source_X, source_y_anom, source_test_idx
        Source-city arrays, used to report home skill on the same scale and to define the
        trained feature range.
    rebuild : bool
        Re-screen the target city instead of reusing its cached manifest.
    """
    if verbose:
        print("=" * 74)
        print(f"TRANSFER TEST · {cfg_target.name.upper()} ({cfg_target.mode})")
        print(f"  {cfg_target.bbox} | {cfg_target.crs} | {cfg_target.start} to {cfg_target.end}")
        print("=" * 74)

    scenes = acquire.screen_and_download(cfg_target, rebuild=rebuild)
    if not scenes:
        raise RuntimeError(f"no usable scenes for {cfg_target.name}")
    grids = cov.build_grids(cfg_target, load_landcover=False)
    h = hv.collect(cfg_target, scenes, grids, since=since, verbose=verbose,
                   cache=True, rebuild=rebuild)

    scene_mean = {k: h.y[h.scene_key == k].mean() for k in set(h.scene_key)}
    y_anom = h.y - np.array([scene_mean[k] for k in h.scene_key])

    # feature-space overlap: read this before any accuracy number
    rows = []
    for j, name in enumerate(cfg_target.features):
        lo, hi = np.nanpercentile(source_X[:, j], [1, 99])
        inside = float(np.mean((h.X[:, j] >= lo) & (h.X[:, j] <= hi)))
        rows.append({"feature": name, "source_p1": round(float(lo), 4),
                     "source_p99": round(float(hi), 4),
                     "target_median": round(float(np.nanmedian(h.X[:, j])), 4),
                     "inside_pct": round(100 * inside, 1)})
    overlap = pd.DataFrame(rows)
    overlap.to_csv(cfg_target.path("transfer",
                   f"feature_overlap_{cfg_target.name}_{cfg_target.mode}.csv"), index=False)

    as_is = metrics(y_anom, source_fit.predict(h.X))

    j_dem = list(cfg_target.features).index("dem")
    keep = [j for j in range(len(cfg_target.features)) if j != j_dem]
    nodem = RandomForestRegressor(n_estimators=cfg_target.rf_estimators,
                                  max_depth=cfg_target.rf_max_depth,
                                  min_samples_leaf=cfg_target.rf_min_leaf,
                                  random_state=cfg_target.seed_model, n_jobs=-1)
    nodem.fit(source_X[:, keep], source_y_anom)
    without_dem = metrics(y_anom, nodem.predict(h.X[:, keep]))

    i1, i2 = train_test_split(np.arange(len(y_anom)), test_size=cfg_target.test_size,
                              random_state=cfg_target.seed_split)
    loc = RandomForestRegressor(n_estimators=cfg_target.rf_estimators,
                                max_depth=cfg_target.rf_max_depth,
                                min_samples_leaf=cfg_target.rf_min_leaf,
                                random_state=cfg_target.seed_model, n_jobs=-1)
    loc.fit(h.X[i1], y_anom[i1])
    local = metrics(y_anom[i2], loc.predict(h.X[i2]))

    home = metrics(source_y_anom[source_test_idx], source_fit.predict(source_X[source_test_idx]))

    res = TransferResult(city=cfg_target.name, as_is=as_is, without_dem=without_dem,
                         local=local, home=home, overlap=overlap,
                         n_samples=h.X.shape[0], n_scenes=h.n_scenes)
    res.table(cfg_target.mode).to_csv(
        cfg_target.path("transfer",
                        f"transferability_{cfg_target.name}_{cfg_target.mode}.csv"), index=False)

    if verbose:
        print("\nfeature-space overlap with the trained range:")
        print(overlap.to_string(index=False))
        dem_ov = float(overlap.loc[overlap.feature == "dem", "inside_pct"].iloc[0])
        print(f"  elevation overlap {dem_ov:.0f} %")
        print("\n" + "=" * 74)
        print(f"{cfg_target.name.upper()} ({cfg_target.mode}) - spatial pattern, anomaly space")
        for label, v in [("home (source city)", home), ("model as-is", as_is),
                         ("model without elevation", without_dem),
                         ("retrained locally", local)]:
            print(f"  {label:26s} MAE {v[0]:5.2f} K  RMSE {v[1]:5.2f} K  R² {v[2]:+.3f}")
        print("=" * 74)
        print(f"  retains {res.retention:.0f} % of home skill")
        print(f"\n{res.verdict()}")
    return res
