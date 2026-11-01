"""
Assembling the pooled training set.
===================================

Walks every screened overpass, cleans it, pairs it with its own optical imagery, and draws
a sample of valid pixels. The result is one pooled dataset tagged by scene, which is what
allows a single model to be trained across all overpasses.

Two details matter and are easy to lose in a refactor:

* Each scene's cleaned 70 m array is retained in :class:`Harvest.scenes`, so the map stage
  does not have to re-read and re-clean every granule.
* Only pixels that survive the v2.1 cleaning (``preprocess.prepare_scene``) are sampled,
  and shoreline pixels are excluded. Grade C scenes are skipped and logged with the reason
  in ``rejected_scenes_<area>_<mode>.csv``.
* Temperatures are in **degrees Celsius** from the moment a granule is read.
* Each sample keeps its row and column on the coarse grid, so accuracy can be measured on
  held-out spatial blocks rather than on random (spatially correlated) pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import covariates as cov
from . import preprocess as pre


@dataclass
class SceneData:
    """One overpass: its cleaned thermal array and acquisition metadata."""
    lst: object                 # xr.DataArray, °C, on the coarse grid, NaN where not clean
    scene: object               # acquire.Scene
    s2_info: dict
    clean: object = None        # preprocess.SceneClean (masks and quality report)


@dataclass
class SceneStub:
    """Placeholder used when a harvest is restored from cache.

    The cache stores the pooled arrays, not the per-scene rasters, because the
    diagnostics only need the former. This carries the acquisition metadata they do use
    (local hour, date) so a cached harvest behaves identically for them. ``lst`` is None,
    so :func:`lstpipe.products.per_scene_maps` requires a full harvest - it needs the
    rasters and will say so rather than fail obscurely.
    """
    key: str
    local_hour: int
    s2_info: dict
    lst: object = None

    @property
    def scene(self):
        return self

    @property
    def date(self) -> str:
        return self.key[:10]

    @property
    def utc(self) -> str:
        return self.key[11:]


@dataclass
class Harvest:
    """The pooled training set and the per-scene material behind it."""
    X: np.ndarray
    y: np.ndarray               # °C
    scene_key: np.ndarray       # which overpass each sample came from
    lulc: np.ndarray | None     # land-cover code per sample, when available
    scenes: dict = field(default_factory=dict)
    gaps: pd.DataFrame = field(default_factory=pd.DataFrame)
    rc: np.ndarray | None = None            # (n, 2) row, column on the coarse grid
    quality: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_scenes(self) -> int:
        return len(self.scenes)

    @property
    def has_rasters(self) -> bool:
        """False when restored from cache: the pooled arrays are present, the per-scene
        rasters are not. Map generation needs the rasters; diagnostics do not."""
        return any(getattr(sd, "lst", None) is not None for sd in self.scenes.values())

    def save(self, path, key: str = "") -> str:
        """Write the pooled arrays and scene metadata to a single compressed file."""
        keys = list(self.scenes)
        np.savez_compressed(
            path,
            run_key=np.array([key]), units=np.array(["degC"]),
            X=self.X, y=self.y, scene_key=self.scene_key.astype(str),
            rc=self.rc if self.rc is not None else np.zeros((0, 2), int),
            lulc=self.lulc if self.lulc is not None else np.array([]),
            has_lulc=np.array([self.lulc is not None]),
            scene_keys=np.array(keys, dtype=str),
            local_hours=np.array([self.scenes[k].scene.local_hour for k in keys]),
            gap_days=np.array([self.scenes[k].s2_info.get("gap_days", np.nan) for k in keys]),
            n_items=np.array([self.scenes[k].s2_info.get("n_items", 0) for k in keys]))
        return str(path)

    @classmethod
    def load(cls, path, expect_key: str = "") -> "Harvest":
        """Restore a cached harvest. Scene rasters are not included - see `has_rasters`.

        Raises if the file was written for a different run, which protects against a
        cache from another study area being read as though it belonged to this one.
        """
        z = np.load(path, allow_pickle=False)
        if "units" not in z or "rc" not in z:
            raise ValueError(
                f"{path} was written by v2.0 (Kelvin, no pixel coordinates). It cannot be "
                "used by v2.1 - re-run 01_pipeline with rebuild=True.")
        if expect_key and "run_key" in z:
            found = str(z["run_key"][0])
            if found and found != expect_key:
                raise ValueError(
                    f"harvest cache belongs to {found}, not {expect_key}: {path}")
        scenes = {k: SceneStub(key=k, local_hour=int(h),
                               s2_info={"gap_days": float(g), "n_items": int(n)})
                  for k, h, g, n in zip(z["scene_keys"], z["local_hours"],
                                        z["gap_days"], z["n_items"])}
        gaps = pd.DataFrame({"date": [k[:10] for k in z["scene_keys"]],
                             "utc": [k[11:] for k in z["scene_keys"]],
                             "s2_gap_days": z["gap_days"], "s2_items": z["n_items"]})
        quality = pd.DataFrame()
        qp = Path(str(path)).with_name(Path(str(path)).name.replace("harvest_", "scene_quality_")
                                       .replace(".npz", ".csv"))
        if qp.exists():
            quality = pd.read_csv(qp)
        return cls(X=z["X"], y=z["y"],
                   scene_key=z["scene_key"].astype(object),
                   lulc=z["lulc"] if bool(z["has_lulc"][0]) else None,
                   scenes=scenes, gaps=gaps, rc=z["rc"], quality=quality)

    def summary(self) -> str:
        years = pd.Series([k[:4] for k in self.scene_key]).value_counts().sort_index()
        per_year = " · ".join(f"{y}: {n:,}" for y, n in years.items())
        return (f"{self.X.shape[0]:,} samples × {self.X.shape[1]} features "
                f"from {self.n_scenes} scenes\n  {per_year}")


def cache_path(cfg):
    """Where a cached harvest for this configuration lives."""
    return cfg.cache("harvest")


def collect(cfg, scenes=None, grids=None, since: str | None = None, verbose: bool = True,
            cache: bool = False, rebuild: bool = False) -> Harvest:
    """Build the pooled training set.

    Parameters
    ----------
    scenes : list[acquire.Scene], optional
        Required unless a cache hit satisfies the request.
    grids : covariates.Grids, optional
        Required unless a cache hit satisfies the request.
    since : str, optional
        Ignore overpasses before this date, ``YYYY-MM-DD``.
    cache : bool
        Save the result, and reuse it on a later call instead of rebuilding. Worth using
        for the validation notebook: harvesting re-fetches a Sentinel-2 composite from AWS
        for every overpass, which dominates the runtime, and the diagnostics only need the
        pooled arrays. Map generation still needs a full harvest.
    rebuild : bool
        Ignore an existing cache and harvest again.
    """
    if cache and not rebuild:
        cached = cache_path(cfg)
        if cached.exists():
            h = Harvest.load(cached, expect_key=cfg.key)
            if verbose:
                print(f"restored cached harvest: {h.summary()}")
                print(f"  from {cached.name} - pass rebuild=True to harvest again")
            return h

    if scenes is None or grids is None:
        raise ValueError(
            "harvest.collect needs scenes and grids when no cache is available. "
            "Run the pipeline notebook first, or pass cache=True after a completed run.")
    rng = np.random.default_rng(0)
    X_parts, y_parts, k_parts, l_parts, rc_parts, gap_rows = [], [], [], [], [], []
    kept: dict[str, SceneData] = {}
    reports, rejected = [], []

    selected = [s for s in scenes
                if since is None or s.date >= since]
    if verbose:
        print(f"{cfg.mode}: harvesting from {len(selected)} overpasses\n")

    def reject(sc, reason, rep=None):
        r = dict(rep or {"date": sc.date, "utc": sc.utc, "local_hour": sc.local_hour})
        r["grade"], r["reject_reason"] = "C", reason
        rejected.append(r)
        reports.append(r)
        if verbose:
            print(f"  [{i}/{len(selected)}] {sc.key}: rejected - {reason}")

    for i, sc in enumerate(selected, 1):
        clean = pre.prepare_scene(cfg, sc, grids.coarse, grids.land_coarse)
        if clean is None:
            reject(sc, "does not overlap the study area")
            continue
        if clean.grade == "C":
            reject(sc, clean.report.get("reject_reason", "grade C"), clean.report)
            continue

        lst = clean.lst
        values = lst.values.astype(float)

        feats, info = cov.s2_for_date(cfg, grids, sc.date, cfg.res_coarse,
                                      grids.coarse, grids.dem_coarse)
        if feats is None:
            reject(sc, "no clear Sentinel-2 pixel within "
                       f"±{cfg.s2_max_window_days} d", clean.report)
            continue

        # train only on clean, observed, non-shoreline pixels with complete predictors
        m = cov.valid_mask(feats, cfg.features, extra=clean.train)
        n = int(m.sum())
        if n < cfg.min_train_px:
            reject(sc, f"only {n} clean pixels with optical predictors", clean.report)
            continue

        idx = np.arange(n)
        if n > cfg.px_per_scene:
            idx = rng.choice(n, cfg.px_per_scene, replace=False)
        rows_, cols_ = np.nonzero(m)

        X_parts.append(cov.stack(feats, cfg.features, m)[idx])
        y_parts.append(values[m][idx])
        k_parts.append(np.full(len(idx), sc.key, dtype=object))
        rc_parts.append(np.c_[rows_[idx], cols_[idx]])
        if grids.has_landcover:
            l_parts.append(grids.wc_coarse[m][idx])

        kept[sc.key] = SceneData(lst=lst, scene=sc, s2_info=info, clean=clean)
        rep = dict(clean.report, s2_gap_days=info["gap_days"], s2_gap_p90=info.get("gap_p90"),
                   s2_items=info["n_items"])
        reports.append(rep)
        gap_rows.append({"date": sc.date, "utc": sc.utc, "mode": cfg.mode,
                         "s2_gap_days": info["gap_days"], "s2_items": info["n_items"],
                         "cloud_lt": info["cloud_lt"]})
        if verbose:
            print(f"  [{i}/{len(selected)}] {sc.key}: grade {clean.grade} · "
                  f"{len(idx):,} samples · clean {100 * clean.report['clean_land_frac']:.0f}% · "
                  f"S2 gap {info['gap_days']:.0f} d")

    quality = pd.DataFrame(reports)
    quality.to_csv(cfg.path("products", f"scene_quality_{cfg.key}.csv"), index=False)
    pd.DataFrame(rejected).to_csv(cfg.path("products", f"rejected_scenes_{cfg.key}.csv"),
                                  index=False)
    if verbose:
        print(f"\n{len(kept)} scenes kept · {len(rejected)} rejected "
              f"(reasons in rejected_scenes_{cfg.key}.csv)")

    if not X_parts:
        raise RuntimeError("no usable scenes - every overpass was rejected; see "
                           f"rejected_scenes_{cfg.key}.csv")

    h = Harvest(X=np.vstack(X_parts), y=np.concatenate(y_parts),
                scene_key=np.concatenate(k_parts),
                lulc=np.concatenate(l_parts) if l_parts else None,
                scenes=kept, gaps=pd.DataFrame(gap_rows),
                rc=np.vstack(rc_parts).astype(int), quality=quality)
    h.gaps.to_csv(cfg.path("products", f"s2_gap_perscene_{cfg.key}.csv"), index=False)
    if cache:
        h.save(cache_path(cfg), key=cfg.key)
        if verbose:
            print(f"cached to {cache_path(cfg).name}")

    if verbose:
        print(f"\n{h.summary()}")
        print(f"optical pairing gap: median {h.gaps.s2_gap_days.median():.1f} d · "
              f"max {h.gaps.s2_gap_days.max():.1f} d")
    return h
