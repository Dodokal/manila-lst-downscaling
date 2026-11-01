"""
Reference grids, static layers and per-scene optical pairing.
=============================================================

Everything the model uses as input, and the spatial frame they share.

The central decision lives in :func:`s2_for_date`. Each thermal scene is paired with its
**own** Sentinel-2 composite from a short window around its acquisition. An earlier design
reused one multi-year composite for every scene, which made the predictors identical across
overpasses - the model then had no way to distinguish scenes and pooled skill collapsed to
near zero. Pairing per scene reduced the median optical-thermal gap from 57 days to about
two and is the single largest improvement in the project's history.

**v2.1 - every usable Sentinel-2 pixel, no scene-level cloud threshold.** v2.0 searched
with a scene cloud-cover ladder (< 10, 30, 50, 80 %) and then *averaged all bands over time
without any per-pixel cloud mask*, so cloudy acquisitions were blended into the predictors.
Now no scene is discarded for its cloud percentage (``s2_scene_cloud_max = 100``). Each
pixel is screened with the Scene Classification Layer (no data, defective, cloud shadow,
medium/high cloud, cirrus removed) and the median of its clear observations is taken.
Pixels with no clear look within ±``s2_window_days`` are filled from progressively wider
windows up to ±``s2_max_window_days``, and the per-pixel gap is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import odc.stac
import pystac_client
import rioxarray as rxr
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

STAC_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
DEM_COLLECTION = "cop-dem-glo-30"
WC_TEMPLATE = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
               "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
S2_BANDS = ["red", "green", "blue", "nir", "swir16", "swir22"]
# Scene Classification Layer classes that are not a clear view of the surface:
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8 cloud medium, 9 cloud high, 10 cirrus
SCL_NOT_CLEAR = (0, 1, 3, 8, 9, 10)


def worldcover_tiles(bbox) -> list[str]:
    """ESA WorldCover 3x3 degree tile names covering ``bbox``.

    v2.0 hard-coded the Manila tile (N12E120), so every other city reported "land cover
    unavailable" and lost its zonal statistics.
    """
    import math
    w, s, e, n = bbox
    tiles = []
    for lat in range(int(math.floor(s / 3) * 3), int(math.floor(n / 3) * 3) + 1, 3):
        for lon in range(int(math.floor(w / 3) * 3), int(math.floor(e / 3) * 3) + 1, 3):
            ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
            tiles.append(ns + ew)
    return tiles


def safe_ratio(a, b):
    """Normalised difference (a - b) / (a + b), guarding a vanishing denominator."""
    d = a + b
    ok = np.abs(d) > 1e-9
    return np.where(ok, (a - b) / np.where(ok, d, 1), np.nan)


def reference_grid(bbox, crs: str, resolution: float):
    """Affine transform and shape spanning the whole study area.

    Built from the study area rather than from any granule footprint, so a date with
    partial coverage cannot shrink the output extent.
    """
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", crs, *bbox)
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    return Affine(resolution, 0, minx, 0, -resolution, maxy), (height, width)


def _blank(transform: Affine, shape, crs: str) -> xr.DataArray:
    ys = transform.f + transform.e * (np.arange(shape[0]) + 0.5)
    xs = transform.c + transform.a * (np.arange(shape[1]) + 0.5)
    return (xr.DataArray(np.full(shape, np.nan, "float32"), dims=["y", "x"],
                         coords={"y": ys, "x": xs})
            .rio.write_crs(crs).rio.write_transform(transform))


@dataclass
class Grids:
    """Shared spatial frame plus the layers that do not change between scenes."""
    coarse: xr.DataArray            # reference grid at res_coarse
    fine: xr.DataArray              # reference grid at res_fine
    dem_coarse: np.ndarray
    dem_fine: np.ndarray
    wc_coarse: np.ndarray | None    # ESA WorldCover class codes
    wc_fine: np.ndarray | None
    catalog: object

    @property
    def has_landcover(self) -> bool:
        return self.wc_coarse is not None

    @property
    def land_coarse(self) -> np.ndarray | None:
        """Land pixels of the study area at the coarse grid (WorldCover not water)."""
        if self.wc_coarse is None:
            return None
        return np.isfinite(self.wc_coarse) & (self.wc_coarse != 80)

    @property
    def land_fine(self) -> np.ndarray | None:
        """Land pixels at 10 m - used to cut the delivered maps at the real shoreline
        rather than along the 70 m ECOSTRESS water mask."""
        if self.wc_fine is None:
            return None
        return np.isfinite(self.wc_fine) & (self.wc_fine != 80)


def build_grids(cfg, load_landcover: bool = True) -> Grids:
    """Construct the reference grids and load DEM and land cover onto them."""
    catalog = pystac_client.Client.open(STAC_URL)

    t_c, s_c = reference_grid(cfg.bbox, cfg.crs, cfg.res_coarse)
    t_f, s_f = reference_grid(cfg.bbox, cfg.crs, cfg.res_fine)
    coarse, fine = _blank(t_c, s_c, cfg.crs), _blank(t_f, s_f, cfg.crs)
    print(f"grids: {cfg.res_coarse} m {s_c} | {cfg.res_fine} m {s_f}")

    dem_items = catalog.search(collections=DEM_COLLECTION, bbox=cfg.bbox).item_collection()
    def _dem(res, match):
        ds = odc.stac.load(dem_items, bbox=cfg.bbox, bands=["data"], resolution=res,
                           chunks={}, crs=cfg.crs, resampling="bilinear")
        return (ds["data"].squeeze().rio.write_crs(cfg.crs)
                .rio.reproject_match(match, resampling=Resampling.bilinear)
                .values.astype(float))
    dem_c = _dem(cfg.res_coarse, coarse)
    dem_f = _dem(cfg.res_fine, fine)
    print(f"DEM: {np.nanmin(dem_c):.0f}–{np.nanmax(dem_c):.0f} m")

    wc_c = wc_f = None
    if load_landcover:
        try:
            # Clip before reading: the source tile is 36000 x 36000 px and reading it
            # whole exhausts memory.
            w, s, e, n = cfg.bbox
            parts = [(rxr.open_rasterio(WC_TEMPLATE.format(tile=t), masked=True, chunks=True)
                      .squeeze().rio.clip_box(minx=w, miny=s, maxx=e, maxy=n).compute())
                     for t in worldcover_tiles(cfg.bbox)]
            if len(parts) == 1:
                wc = parts[0]
            else:
                from rioxarray.merge import merge_arrays
                wc = merge_arrays(parts)
            wc_c = wc.rio.reproject_match(coarse, resampling=Resampling.nearest).values.astype(float)
            wc_f = wc.rio.reproject_match(fine, resampling=Resampling.nearest).values.astype(float)
            print(f"land cover: {wc.shape} -> {wc_c.shape} / {wc_f.shape}")
        except Exception as exc:                        # noqa: BLE001
            print(f"land cover unavailable ({type(exc).__name__}); zonal statistics disabled")

    return Grids(coarse=coarse, fine=fine, dem_coarse=dem_c, dem_fine=dem_f,
                 wc_coarse=wc_c, wc_fine=wc_f, catalog=catalog)


def _s2_search(cfg, catalog, date: str, half: int, inner: int = -1):
    """Sentinel-2 items within ±``half`` days of ``date`` but outside ±``inner``."""
    t0 = datetime.strptime(date, "%Y-%m-%d")
    d0 = (t0 - timedelta(days=half)).strftime("%Y-%m-%d")
    d1 = (t0 + timedelta(days=half)).strftime("%Y-%m-%d")
    query = ({"eo:cloud_cover": {"lt": cfg.s2_scene_cloud_max}}
             if cfg.s2_scene_cloud_max < 100 else None)
    items = catalog.search(collections=S2_COLLECTION, bbox=cfg.bbox,
                           datetime=f"{d0}/{d1}", query=query).item_collection()
    if inner >= 0:
        items = [i for i in items
                 if abs((i.datetime.replace(tzinfo=None) - t0).days) > inner]
    return list(items)


def _windows(cfg):
    w = [cfg.s2_window_days]
    while w[-1] < cfg.s2_max_window_days:
        w.append(min(w[-1] * 2, cfg.s2_max_window_days))
    return w


def s2_for_date(cfg, grids: Grids, date: str, resolution: int, match_to, dem):
    """Per-scene optical covariates from every clear Sentinel-2 pixel, snapped to ``match_to``.

    Returns
    -------
    (features, info)
        ``features`` is a dict keyed by the names in ``cfg.features``. ``info`` reports the
        number of acquisitions used, the per-pixel gap to the clear observation (median and
        90th percentile, days), and the share of pixels each window filled.
        ``(None, info)`` if no clear imagery exists within ±``s2_max_window_days``.
    """
    import pandas as pd

    shape = match_to.shape
    comp = {b: np.full(shape, np.nan) for b in S2_BANDS}
    gap = np.full(shape, np.nan)
    t0 = pd.Timestamp(date)
    n_items, filled, inner = 0, {}, -1

    for half in _windows(cfg):
        need = ~np.isfinite(comp["red"])
        if not need.any():
            break
        items = _s2_search(cfg, grids.catalog, date, half, inner)
        inner = half
        if not items:
            continue
        n_items += len(items)
        ds = odc.stac.load(items, bbox=cfg.bbox, bands=S2_BANDS + ["scl"],
                           resolution=resolution, chunks={"x": 1024, "y": 1024},
                           groupby="solar_day", crs=cfg.crs, dtype="float32",
                           resampling={"scl": "nearest", "*": "bilinear"})
        clear = ~ds["scl"].isin(SCL_NOT_CLEAR) & np.isfinite(ds["scl"])
        bands = ds[S2_BANDS].where(clear & (ds["red"] > 0))
        med = bands.median(dim="time", skipna=True)
        days = xr.DataArray(np.abs((pd.to_datetime(ds.time.values) - t0).days).astype(float),
                            dims="time", coords={"time": ds.time})
        dgap = days.where(clear).min(dim="time", skipna=True)

        out = xr.merge([med, dgap.rename("gap")]).compute()
        out = out.rio.write_crs(cfg.crs).rio.reproject_match(match_to,
                                                             resampling=Resampling.average)
        new = need & np.isfinite(out["red"].values)
        for b in S2_BANDS:
            comp[b][new] = out[b].values[new]
        gap[new] = out["gap"].values[new]
        filled[half] = round(float(new.mean()), 3)

    if not np.isfinite(comp["red"]).any():
        return None, {"n_items": n_items, "gap_days": np.nan, "gap_p90": np.nan,
                      "cloud_lt": cfg.s2_scene_cloud_max, "filled": filled}

    blue, green, red = comp["blue"], comp["green"], comp["red"]
    nir, swir16, swir22 = comp["nir"], comp["swir16"], comp["swir22"]
    features = {
        "blue": blue, "green": green, "red": red, "nir": nir,
        "swir16": swir16, "swir22": swir22, "dem": dem,
        "ndvi": safe_ratio(nir, red),        # vegetation
        "ndbi": safe_ratio(swir16, nir),     # built-up
        "ndwi": safe_ratio(green, nir),      # moisture
    }
    g = gap[np.isfinite(gap)]
    return features, {"n_items": n_items,
                      "gap_days": float(np.median(g)) if g.size else np.nan,
                      "gap_p90": float(np.percentile(g, 90)) if g.size else np.nan,
                      "cloud_lt": cfg.s2_scene_cloud_max, "filled": filled}


def stack(features: dict, order, mask=None) -> np.ndarray:
    """Design matrix in ``order``, optionally restricted to ``mask``.

    Both training and inference call this, which is what keeps their columns aligned.
    """
    if mask is None:
        return np.column_stack([features[k].ravel() for k in order])
    return np.column_stack([features[k][mask] for k in order])


def valid_mask(features: dict, order, extra=None) -> np.ndarray:
    """Pixels where every predictor is finite, optionally combined with ``extra``."""
    m = np.ones(np.asarray(features[order[0]]).shape, bool)
    for k in order:
        m &= np.isfinite(features[k])
    if extra is not None:
        m &= extra
    return m
