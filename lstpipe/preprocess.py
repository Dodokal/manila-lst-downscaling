"""
Scene cleaning - quality masking, water and cloud removal, detector gap filling.
================================================================================

Turns a raw ECOSTRESS granule into an analysis-ready array. Two distinct problems are
handled here and it is worth keeping them separate in your head:

**Quality masking** uses the QC band. Bits 0-1 carry the primary flag: 00 good,
01 acceptable, 10 poor, 11 bad. Anything below 2 is kept.

**Missing data** appears as striping, and per JPL documentation has three causes, only two
of which are defects:

===========  ===============================================================  ==============
Code         Cause                                                            Treatment
===========  ===============================================================  ==============
``-9998``    detectors damaged before launch (8 lines in every 128)           gap-filled
``-9997``    band 1 observes ground the other bands do not                    gap-filled
*none*       push-whisk scan overlap - a visualisation effect that            left untouched
             geolocation already corrects; the values are valid
===========  ===============================================================  ==============

So the task is detection and interpolation of missing values, not radiometric de-striping.
Filled pixels are tracked in a mask so they can be excluded from model training while still
appearing in delivered maps.
"""

from __future__ import annotations

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from rioxarray.exceptions import NoDataInBounds, OneDimensionalRaster

ECO_FILL_CODES = (-9998.0, -9997.0)
FILL_MAX_GAP = 12          # documented defect is 8 lines; allow a little margin
LST_VALID_K = (200.0, 400.0)
MAD_K = 6.0                # conservative: fill codes are handled separately


# ---------------------------------------------------------------- quality masking
def good_quality_mask(qc: xr.DataArray) -> xr.DataArray:
    """True where QC bits 0-1 indicate good (00) or acceptable (01) quality.

    Missing QC values are filled with 255, which resolves to 11 and is discarded.
    """
    qc_int = qc.fillna(255).astype(np.uint16)
    return (qc_int & 0b11) < 2


def qc_mask_lst(lst_file: str, qc_file: str, verbose: bool = False):
    """Load an LST granule and apply its QC mask. Returns None if nothing survives."""
    lst = rioxarray.open_rasterio(lst_file, masked=True).squeeze()
    qc = rioxarray.open_rasterio(qc_file, masked=True).squeeze()
    qc = qc.rio.reproject_match(lst, resampling=rasterio.enums.Resampling.nearest)

    masked = lst.where(good_quality_mask(qc))
    frac = float(masked.notnull().mean())
    if verbose:
        print(f"    QC mask: {frac:.1%} retained")
    return None if frac == 0 else masked


def clip_and_mask(lst_masked, water_file, cloud_file, bbox, crs,
                  max_cloud_cover: int = 20, verbose: bool = False):
    """Remove water and cloud, reproject, clip to the study area.

    Returns None if the granule misses the area, is too cloudy over it, or retains no
    valid pixels.
    """
    water = rioxarray.open_rasterio(water_file, masked=True).squeeze()
    water = water.rio.reproject_match(lst_masked, resampling=rasterio.enums.Resampling.nearest)
    cloud = rioxarray.open_rasterio(cloud_file, masked=True).squeeze()
    cloud = cloud.rio.reproject_match(lst_masked, resampling=rasterio.enums.Resampling.nearest)

    land = lst_masked.where((water != 1) & (cloud != 1))
    reproj = land.rio.reproject(crs)

    west, south, east, north = bbox
    try:
        clipped = reproj.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north,
                                      crs="EPSG:4326")
    except (NoDataInBounds, OneDimensionalRaster):
        if verbose:
            print("    granule does not meaningfully overlap the study area")
        return None

    if max_cloud_cover is not None:
        total = clipped.size
        cloudy = (cloud.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north,
                                     crs="EPSG:4326") == 1).sum().item()
        pct = (cloudy / total) * 100 if total else 100
        if pct > max_cloud_cover:
            if verbose:
                print(f"    cloud cover {pct:.1f}% over the area, above {max_cloud_cover}%")
            return None

    return None if float(clipped.notnull().mean()) == 0 else clipped


# ---------------------------------------------------------------- gap filling
def detect_fill(arr) -> np.ndarray:
    """Boolean mask of documented missing-data pixels: error codes, non-finite,
    and physically impossible temperatures."""
    a = np.asarray(arr, float)
    bad = np.zeros(a.shape, bool)
    for code in ECO_FILL_CODES:
        bad |= np.isclose(a, code, atol=0.5)
    bad |= ~np.isfinite(a)
    bad |= (a < LST_VALID_K[0]) | (a > LST_VALID_K[1])
    return bad


def fill_missing_lines(arr, max_gap: int = FILL_MAX_GAP):
    """Interpolate missing across-track lines along each column.

    Interpolation runs down the columns because that is the direction in which the valid
    data either side of a dead detector block relates. Gaps longer than ``max_gap`` are
    left as NaN rather than invented.

    Returns
    -------
    (filled, filled_mask, report)
    """
    a = np.asarray(arr, float).copy()
    bad = detect_fill(a)
    a[bad] = np.nan
    filled_mask = np.zeros(a.shape, bool)
    if not bad.any():
        return a, filled_mask, {"missing_px": 0, "filled_px": 0,
                                "missing_rows": 0, "unfilled_px": 0}

    row_missing = bad.mean(axis=1) > 0.5      # near-empty rows = dead detector lines
    H, W = a.shape
    idx = np.arange(H)
    for c in range(W):
        col = a[:, c]
        m = np.isfinite(col)
        if m.sum() < 2:
            continue
        gaps, start = [], None
        for i in range(H):
            if not m[i] and start is None:
                start = i
            elif m[i] and start is not None:
                gaps.append((start, i - 1))
                start = None
        if start is not None:
            gaps.append((start, H - 1))
        interp = np.interp(idx, idx[m], col[m])
        for g0, g1 in gaps:
            if (g1 - g0 + 1) <= max_gap and g0 > 0 and g1 < H - 1:   # interior only
                col[g0:g1 + 1] = interp[g0:g1 + 1]
                filled_mask[g0:g1 + 1, c] = True
        a[:, c] = col

    return a, filled_mask, {"missing_px": int(bad.sum()),
                            "filled_px": int(filled_mask.sum()),
                            "missing_rows": int(row_missing.sum()),
                            "unfilled_px": int(bad.sum() - filled_mask.sum())}


def remove_outliers(arr, mad_k: float = MAD_K):
    """Mask statistical outliers using a robust (MAD) threshold."""
    a = np.asarray(arr, float).copy()
    fin = np.isfinite(a)
    if fin.sum() < 100:
        return a, 0
    med = np.nanmedian(a[fin])
    mad = np.nanmedian(np.abs(a[fin] - med)) * 1.4826 or 1e-6
    bad = fin & (np.abs(a - med) > mad_k * mad)
    a[bad] = np.nan
    return a, int(bad.sum())


def clean_scene(arr):
    """Full cleaning chain: gap-fill documented missing data, then remove outliers.

    Returns
    -------
    (clean, filled_mask, report)
        ``filled_mask`` marks interpolated pixels so callers can exclude them from
        training while keeping them in delivered maps.
    """
    filled, fmask, rep = fill_missing_lines(arr)
    clean, n_out = remove_outliers(filled)
    rep["outlier_px"] = n_out
    return clean, fmask, rep
