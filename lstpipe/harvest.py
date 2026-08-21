"""
Assembling the pooled training set.
===================================

Walks every screened overpass, cleans it, pairs it with its own optical imagery, and draws
a sample of valid pixels. The result is one pooled dataset tagged by scene, which is what
allows a single model to be trained across all overpasses.

Two details matter and are easy to lose in a refactor:

* Each scene's cleaned 70 m array is retained in :class:`Harvest.scenes`, so the map stage
  does not have to re-read and re-clean every granule.
* Gap-filled pixels are **excluded from training** through ``~filled_mask`` while remaining
  in the delivered maps. The model is never fitted to interpolated values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import covariates as cov
from . import preprocess as pre


@dataclass
class SceneData:
    """One overpass: its cleaned thermal array and acquisition metadata."""
    lst: object                 # xr.DataArray, kelvin, on the coarse grid
    scene: object               # acquire.Scene
    s2_info: dict


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
    y: np.ndarray               # kelvin
    scene_key: np.ndarray       # which overpass each sample came from
    lulc: np.ndarray | None     # land-cover code per sample, when available
    scenes: dict = field(default_factory=dict)
    gaps: pd.DataFrame = field(default_factory=pd.DataFrame)

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
            run_key=np.array([key]),
            X=self.X, y=self.y, scene_key=self.scene_key.astype(str),
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
        return cls(X=z["X"], y=z["y"],
                   scene_key=z["scene_key"].astype(object),
                   lulc=z["lulc"] if bool(z["has_lulc"][0]) else None,
                   scenes=scenes, gaps=gaps)

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
    X_parts, y_parts, k_parts, l_parts, gap_rows = [], [], [], [], []
    kept: dict[str, SceneData] = {}

    selected = [s for s in scenes
                if since is None or s.date >= since]
    if verbose:
        print(f"{cfg.mode}: harvesting from {len(selected)} overpasses\n")

    for i, sc in enumerate(selected, 1):
        masked = pre.qc_mask_lst(sc.lst_file, sc.qc_file)
        if masked is None:
            continue
        clipped = pre.clip_and_mask(masked, sc.water_file, sc.cloud_file,
                                    cfg.bbox, cfg.crs, cfg.max_cloud_cover)
        if clipped is None:
            continue

        lst = clipped.rio.reproject_match(grids.coarse).rio.write_nodata(np.nan)
        cleaned, filled_mask, rep = pre.clean_scene(lst.values.astype(float))
        lst = lst.copy(data=cleaned)

        feats, info = cov.s2_for_date(cfg, grids, sc.date, cfg.res_coarse,
                                      grids.coarse, grids.dem_coarse)
        if feats is None:
            if verbose:
                print(f"  [{i}/{len(selected)}] {sc.key}: no optical imagery in window")
            continue

        # train only on observed pixels: finite, plausible, and not interpolated
        m = cov.valid_mask(feats, cfg.features,
                           extra=np.isfinite(cleaned) & (cleaned > 100) & ~filled_mask)
        n = int(m.sum())
        if n < 500:
            if verbose:
                print(f"  [{i}/{len(selected)}] {sc.key}: only {n} valid pixels")
            continue

        idx = np.arange(n)
        if n > cfg.px_per_scene:
            idx = rng.choice(n, cfg.px_per_scene, replace=False)

        X_parts.append(cov.stack(feats, cfg.features, m)[idx])
        y_parts.append(cleaned[m][idx])
        k_parts.append(np.full(len(idx), sc.key, dtype=object))
        if grids.has_landcover:
            l_parts.append(grids.wc_coarse[m][idx])

        kept[sc.key] = SceneData(lst=lst, scene=sc, s2_info=info)
        gap_rows.append({"date": sc.date, "utc": sc.utc, "mode": cfg.mode,
                         "s2_gap_days": info["gap_days"], "s2_items": info["n_items"],
                         "cloud_lt": info["cloud_lt"]})
        if verbose:
            rows = rep.get("missing_rows", 0)
            extra = f" · {rows} dead lines filled" if rows else ""
            print(f"  [{i}/{len(selected)}] {sc.key}: {len(idx):,} samples · "
                  f"S2 gap {info['gap_days']:.0f} d{extra}")

    if not X_parts:
        raise RuntimeError("no usable scenes - relax the quality threshold or widen the dates")

    h = Harvest(X=np.vstack(X_parts), y=np.concatenate(y_parts),
                scene_key=np.concatenate(k_parts),
                lulc=np.concatenate(l_parts) if l_parts else None,
                scenes=kept, gaps=pd.DataFrame(gap_rows))
    h.gaps.to_csv(cfg.path("products", f"s2_gap_perscene_{cfg.mode}.csv"), index=False)
    if cache:
        h.save(cache_path(cfg), key=cfg.key)
        if verbose:
            print(f"cached to {cache_path(cfg).name}")

    if verbose:
        print(f"\n{h.summary()}")
        print(f"optical pairing gap: median {h.gaps.s2_gap_days.median():.1f} d · "
              f"max {h.gaps.s2_gap_days.max():.1f} d")
    return h
