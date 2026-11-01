"""
Scene cleaning - QC decoding, stripe and cloud removal, outlier screening, grading.
===================================================================================

Turns one ECOSTRESS Collection 2 tiled granule (``ECO_L2T_LSTE`` v002) into an analysis-ready
array **in degrees Celsius** on the study-area grid, plus a training mask and a quality grade.

Everything here was rebuilt in v2.1 after a pixel-level audit of the 66 Manila granules
(see ``docs/CLEANING_EVIDENCE.md``). What the audit found, and what is done about it:

**1. Stripes are flagged in the QC band, and v2.0 kept them.**
QC bits 0-1 (mandatory QA) take 00 *best* / 01 *nominal* / 10 cloud / 11 not produced.
QC bits 2-3 (L1B data quality) take 00 good / 01 *missing-stripe pixel* / 11 bad data.
v2.0 kept every pixel with bits 0-1 < 2 and never read bits 2-3, so interpolated stripe
pixels went into training and into the maps. In the 2024-07-07 22:16 UTC granule they sit
**+10.6 °C** above clean pixels of the same scene. They are now removed, together with a
``qc_stripe_buffer_px`` halo. Bits 0-1 = 01 is *kept*: in humid scenes it covers up to 90 %
of the land surface and is not biased there.

**2. The v2.0 "gap filler" created the cloud-shaped artefacts.**
It interpolated *down image columns* every NaN gap of up to 12 pixels. In the tiled product
scan lines run diagonally and clouds, rivers and masked stripes are all NaN - so cloud holes
were filled with vertical streaks. It also looked for the -9998 / -9997 swath fill codes,
which the tiled product does not use. Gap filling of ECOSTRESS is removed entirely: gaps
stay gaps, and the 10 m map comes from Sentinel-2 plus the observed residual (products.py).

**3. Residual detector striping is removed along the scan direction only.**
The scan direction is measured from the QC stripe lines themselves (it differs between
ascending and descending passes). Line offsets are estimated from a local high-pass, kept
only where significant (z > 4), and subtracted. A wrong-angle control on 42 scenes removes
0.03-0.06 K, against 0.10 K median at the true angle - the correction is selective.
Scan-block *levelling* was tested and rejected: across a 12-pixel gap, ordinary urban
contrast already reaches 2-5 °C, so block steps cannot be separated from real structure.
Blocks are excluded through grading instead of being "repaired".

**4. Clouds the ECOSTRESS mask misses.** Cloud and internal NaN holes get a buffer; cold
outliers (median − k·MAD) and a physical floor ``lst_min_c`` remove the 2-12 °C pixels that
MSF found in July scenes; cold outliers are buffered like clouds.

**5. Shoreline pixels** (70 m cells touching water) mix water and land. They are mapped, but
never used for training.

Every scene receives a report and a grade: **A** ≥ 60 % of the land area clean, **B** ≥ 30 %,
**C** otherwise. Grade C scenes are rejected - no model, no map, not in any composite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from rioxarray.exceptions import NoDataInBounds, OneDimensionalRaster
from scipy import ndimage as ndi

LST_VALID_K = (200.0, 400.0)
CLIP_PAD_DEG = 0.03          # context around the AOI for angle detection and destriping


# ---------------------------------------------------------------- QC decoding
def qc_bits(qc, lo: int, hi: int) -> np.ndarray:
    """Extract QC bits ``lo``..``hi`` (inclusive) as integers."""
    q = np.asarray(qc).astype(np.int64)
    return (q >> lo) & ((1 << (hi - lo + 1)) - 1)


def good_quality_mask(qc) -> np.ndarray:
    """Mandatory QA best (00) or nominal (01). Kept for backwards compatibility.

    On its own this is **not** sufficient: stripe pixels also carry 01. Use
    :func:`stripe_flag` as well.
    """
    q = np.nan_to_num(np.asarray(qc, float), nan=255).astype(np.int64)
    return qc_bits(q, 0, 1) < 2


def stripe_flag(qc) -> np.ndarray:
    """L1B data-quality bits 2-3 not 00: interpolated missing-stripe pixel or bad data."""
    q = np.nan_to_num(np.asarray(qc, float), nan=255).astype(np.int64)
    return qc_bits(q, 2, 3) != 0


# ---------------------------------------------------------------- scan geometry
def stripe_angle(flag: np.ndarray, step: float = 0.25) -> float | None:
    """Orientation of the stripe lines, as the angle of their normal in degrees.

    Projects flagged pixels onto candidate normals and keeps the angle at which they pile
    onto the fewest lines. Returns None when too few pixels are flagged to tell.
    """
    yy, xx = np.nonzero(flag)
    if len(yy) < 200:
        return None
    best, score = None, -1.0
    for th in np.arange(0.0, 180.0, step):
        t = np.deg2rad(th)
        u = xx * np.cos(t) + yy * np.sin(t)
        h = np.bincount((u - u.min()).astype(int)).astype(float)
        s = float((h ** 2).sum() / h.sum() ** 2)
        if s > score:
            best, score = float(th), s
    return best


def scan_lines(shape, angle: float) -> np.ndarray:
    """Integer index of the scan line each pixel belongs to."""
    t = np.deg2rad(angle)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    u = xx * np.cos(t) + yy * np.sin(t)
    return np.round(u - u.min()).astype(int)


def nan_gauss(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaN (normalised convolution)."""
    m = np.isfinite(a)
    num = ndi.gaussian_filter(np.where(m, a, 0.0), sigma)
    den = ndi.gaussian_filter(m.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.05] = np.nan
    return out


def _line_medians(r, u, min_n):
    ok = np.isfinite(r)
    n = int(u.max()) + 1
    P, N = np.full(n, np.nan), np.zeros(n, int)
    o = np.argsort(u[ok], kind="stable")
    uu, rr = u[ok][o], r[ok][o]
    b = np.searchsorted(uu, np.arange(n + 1))
    for k in range(n):
        seg = rr[b[k]:b[k + 1]]
        N[k] = len(seg)
        if len(seg) >= min_n:
            P[k] = np.median(seg)
    return P, N


def destripe_lines(x: np.ndarray, angle: float, iters: int = 2, sigma: float = 3.0,
                   hp: int = 15, z: float = 4.0, min_n: int = 30):
    """Remove significant detector-line offsets along the scan direction.

    Each line's median departure from a local (σ = 3 px) background is high-passed with a
    ``hp``-line running mean, so gradients wider than about a kilometre cannot leak into
    the correction, and applied only where it exceeds ``z`` standard errors. (A running
    *median* was tried first and rejected: over 9 lines it returns the majority sign of a
    periodic stripe pattern and so removes the signal it should keep.)

    Returns (corrected, applied_sd_C).
    """
    u = scan_lines(x.shape, angle)
    out = x.copy()
    total = np.zeros(int(u.max()) + 1)
    for _ in range(iters):
        r = out - nan_gauss(out, sigma)
        P, N = _line_medians(r, u, min_n)
        Pf = np.where(np.isfinite(P), P, 0.0)
        hpP = Pf - ndi.uniform_filter1d(Pf, size=hp, mode="nearest")
        se = 1.2533 * np.nanstd(r) / np.sqrt(np.maximum(N, 1))
        d = np.where(np.isfinite(P) & (np.abs(hpP) > z * se), hpP, 0.0)
        out = out - d[u]
        total += d
    valid = np.isfinite(x)
    sd = float(np.std(total[u][valid])) if valid.any() else 0.0
    return out, sd


# ---------------------------------------------------------------- scene preparation
@dataclass
class SceneClean:
    """One cleaned overpass on the study-area grid."""
    lst: xr.DataArray                     # °C, NaN where not usable
    train: np.ndarray                     # bool: usable for training (excludes shoreline)
    shore: np.ndarray                     # bool: 70 m cell touches water
    report: dict = field(default_factory=dict)

    @property
    def grade(self) -> str:
        return self.report.get("grade", "C")


def _open(path, bbox):
    da = rioxarray.open_rasterio(path, masked=False).squeeze(drop=True)
    w, s, e, n = bbox
    p = CLIP_PAD_DEG
    return da.rio.clip_box(minx=w - p, miny=s - p, maxx=e + p, maxy=n + p, crs="EPSG:4326")


def _grade(cfg, frac: float, stripe_sd: float) -> str:
    g = "A" if frac >= cfg.grade_a_frac else "B" if frac >= cfg.grade_b_frac else "C"
    if g == "A" and stripe_sd > cfg.stripe_downgrade_k:
        g = "B"
    if stripe_sd > cfg.stripe_reject_k:
        g = "C"
    return g


def prepare_scene(cfg, scene, grid: xr.DataArray, land_ref: np.ndarray | None = None,
                  verbose: bool = False) -> SceneClean | None:
    """Clean one granule and place it on ``grid``. Returns None if it misses the area.

    Parameters
    ----------
    scene : acquire.Scene
    grid : xr.DataArray
        The coarse reference grid (``Grids.coarse``).
    land_ref : bool array on ``grid``, optional
        Land pixels of the study area (from ESA WorldCover). The denominator of the clean
        fraction used for grading; without it the whole study area is used.
    """
    try:
        lst = _open(scene.lst_file, cfg.bbox)
        qc = _open(scene.qc_file, cfg.bbox)
        cloud = _open(scene.cloud_file, cfg.bbox)
        water = _open(scene.water_file, cfg.bbox)
    except (NoDataInBounds, OneDimensionalRaster):
        return None

    L = lst.values.astype(float)
    Q = np.nan_to_num(qc.values.astype(float), nan=255).astype(np.int64)
    Cl = np.nan_to_num(cloud.values.astype(float), nan=0) == 1
    W = np.nan_to_num(water.values.astype(float), nan=0) == 1

    finite = np.isfinite(L) & (L > LST_VALID_K[0]) & (L < LST_VALID_K[1])
    land = ~W
    base = land & finite & (qc_bits(Q, 0, 1) < 2)

    # 1. QC stripe lines and their halo
    sflag = stripe_flag(Q)
    stripe_zone = ndi.binary_dilation(sflag, iterations=cfg.qc_stripe_buffer_px) \
        if cfg.qc_stripe_buffer_px else sflag

    # 2. clouds and internal holes (C2 already sets most cloud pixels to NaN) + buffer
    holes = land & ~finite
    cloud_zone = Cl | holes
    if cfg.cloud_buffer_px:
        cloud_zone = ndi.binary_dilation(cloud_zone, iterations=cfg.cloud_buffer_px)

    good = base & ~stripe_zone & ~cloud_zone
    C = np.where(good, L - 273.15, np.nan)

    # 3. detector-line destriping along the measured scan direction
    angle, stripe_sd = stripe_angle(sflag), 0.0
    if cfg.destripe and angle is not None and np.isfinite(C).sum() > 500:
        C, stripe_sd = destripe_lines(C, angle)

    # 4. undetected cloud and implausible values
    n_out = 0
    v = C[np.isfinite(C)]
    if v.size > 100:
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)) * 1.4826) or 0.5
        cold = np.isfinite(C) & (C < med - cfg.outlier_mad_cold * mad)
        hot = np.isfinite(C) & (C > med + cfg.outlier_mad_hot * mad)
        if cfg.lst_min_c is not None:
            cold |= np.isfinite(C) & (C < cfg.lst_min_c)
        if cfg.lst_max_c is not None:
            hot |= np.isfinite(C) & (C > cfg.lst_max_c)
        if cfg.cloud_buffer_px and cold.any():
            cold = ndi.binary_dilation(cold, iterations=cfg.cloud_buffer_px)
        bad = (cold | hot) & np.isfinite(C)
        n_out = int(bad.sum())
        C[cold | hot] = np.nan

    # 5. shoreline
    shore = land & ndi.binary_dilation(W, iterations=cfg.shore_buffer_px) \
        if cfg.shore_buffer_px else np.zeros_like(W)

    # onto the study-area grid
    native = lst.copy(data=C.astype("float32")).rio.write_nodata(np.nan)
    on_grid = native.rio.reproject_match(grid, resampling=Resampling.nearest)
    shore_g = (lst.copy(data=shore.astype("float32")).rio.write_nodata(np.nan)
               .rio.reproject_match(grid, resampling=Resampling.nearest).values == 1)
    cloud_g = (lst.copy(data=Cl.astype("float32")).rio.write_nodata(np.nan)
               .rio.reproject_match(grid, resampling=Resampling.nearest).values == 1)

    vals = on_grid.values.astype(float)
    obs = np.isfinite(vals)
    denom_mask = land_ref if land_ref is not None else np.ones(vals.shape, bool)
    denom = max(int(denom_mask.sum()), 1)
    if obs.sum() == 0:
        return None

    # ECOSTRESS cloud cover over the area: scenes above the limit are not used
    cloud_pct = 100.0 * float((cloud_g & denom_mask).sum()) / denom

    frac = float((obs & denom_mask).sum()) / denom
    train = obs & ~shore_g
    report = {
        "date": scene.date, "utc": scene.utc, "local_hour": scene.local_hour,
        "clean_land_frac": round(frac, 3),
        "stripe_px_native": int((base & stripe_zone).sum()),
        "cloud_px_native": int((base & cloud_zone & ~stripe_zone).sum()),
        "outlier_px": n_out,
        "scan_angle_deg": angle,
        "destripe_sd_C": round(stripe_sd, 3),
        "cloud_pct_aoi": round(cloud_pct, 1),
        "train_px": int(train.sum()),
        "median_C": round(float(np.nanmedian(vals)), 2),
    }
    grade = _grade(cfg, frac, stripe_sd)
    if cfg.max_cloud_cover is not None and cloud_pct > cfg.max_cloud_cover:
        grade, report["reject_reason"] = "C", f"cloud {cloud_pct:.0f}% > {cfg.max_cloud_cover}%"
    elif grade == "C" and stripe_sd > cfg.stripe_reject_k:
        report["reject_reason"] = f"striping {stripe_sd:.2f} °C > {cfg.stripe_reject_k} °C"
    elif grade == "C":
        report["reject_reason"] = f"clean land {100 * frac:.0f}% < {100 * cfg.grade_b_frac:.0f}%"
    report["grade"] = grade

    if verbose:
        print(f"    {scene.key}: grade {grade} · clean {100 * frac:.0f}% · "
              f"angle {angle} · destripe {stripe_sd:.2f} °C · outliers {n_out}")
    return SceneClean(lst=on_grid.rio.write_nodata(np.nan), train=train, shore=shore_g,
                      report=report)


# ---------------------------------------------------------------- legacy entry points
def qc_mask_lst(lst_file: str, qc_file: str, verbose: bool = False):
    """v2.0 entry point, kept so old notebooks fail loudly rather than silently."""
    raise RuntimeError("qc_mask_lst was replaced in v2.1 by preprocess.prepare_scene, "
                       "which also removes QC-flagged stripes. Re-run the pipeline.")


def clean_scene(arr):
    raise RuntimeError("clean_scene (column gap filling) was removed in v2.1: it created "
                       "vertical streaks in cloud holes. Use preprocess.prepare_scene.")
