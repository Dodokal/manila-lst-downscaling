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
WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_N12E120_Map.tif")
S2_BANDS = ["red", "green", "blue", "nir", "swir16", "swir22"]


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
            wc = (rxr.open_rasterio(WC_URL, masked=True, chunks=True).squeeze()
                  .rio.clip_box(minx=w, miny=s, maxx=e, maxy=n).compute())
            wc_c = wc.rio.reproject_match(coarse, resampling=Resampling.nearest).values.astype(float)
            wc_f = wc.rio.reproject_match(fine, resampling=Resampling.nearest).values.astype(float)
            print(f"land cover: {wc.shape} -> {wc_c.shape} / {wc_f.shape}")
        except Exception as exc:                        # noqa: BLE001
            print(f"land cover unavailable ({type(exc).__name__}); zonal statistics disabled")

    return Grids(coarse=coarse, fine=fine, dem_coarse=dem_c, dem_fine=dem_f,
                 wc_coarse=wc_c, wc_fine=wc_f, catalog=catalog)


def _s2_items(cfg, catalog, date: str):
    """Sentinel-2 items near a date, relaxing the cloud filter until something is found."""
    d0 = (datetime.strptime(date, "%Y-%m-%d")
          - timedelta(days=cfg.s2_window_days)).strftime("%Y-%m-%d")
    d1 = (datetime.strptime(date, "%Y-%m-%d")
          + timedelta(days=cfg.s2_window_days)).strftime("%Y-%m-%d")
    for cloud in cfg.s2_cloud_ladder:
        items = catalog.search(collections=S2_COLLECTION, bbox=cfg.bbox,
                               datetime=f"{d0}/{d1}",
                               query={"eo:cloud_cover": {"lt": cloud}}).item_collection()
        if len(items):
            return items, cloud
    return [], None


def s2_for_date(cfg, grids: Grids, date: str, resolution: int, match_to, dem):
    """Per-scene optical covariates, snapped to ``match_to``.

    Returns
    -------
    (features, info)
        ``features`` is a dict keyed by the names in ``cfg.features``; ``info`` reports the
        gap in days to the nearest acquisition, how many items were used, and the cloud
        threshold that produced them. ``(None, info)`` if no imagery is available.
    """
    import pandas as pd

    items, cloud = _s2_items(cfg, grids.catalog, date)
    if len(items) == 0:
        return None, {"n_items": 0, "gap_days": np.nan, "cloud_lt": None}

    ds = odc.stac.load(items, bbox=cfg.bbox, bands=S2_BANDS, resolution=resolution,
                       chunks={}, groupby="solar_day", crs=cfg.crs, resampling="bilinear")
    ds = ds.rio.reproject_match(match_to, resampling=Resampling.average)
    band = lambda k: ds[k].mean(dim="time").values.astype(float)   # noqa: E731

    blue, green, red = band("blue"), band("green"), band("red")
    nir, swir16, swir22 = band("nir"), band("swir16"), band("swir22")
    features = {
        "blue": blue, "green": green, "red": red, "nir": nir,
        "swir16": swir16, "swir22": swir22, "dem": dem,
        "ndvi": safe_ratio(nir, red),        # vegetation
        "ndbi": safe_ratio(swir16, nir),     # built-up
        "ndwi": safe_ratio(green, nir),      # moisture
    }
    gaps = [abs((pd.Timestamp(i.datetime.date()) - pd.Timestamp(date)).days) for i in items]
    return features, {"n_items": len(items), "gap_days": float(min(gaps)), "cloud_lt": cloud}


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
