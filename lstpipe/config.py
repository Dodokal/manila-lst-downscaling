"""
Configuration - study area, paths and every tunable constant.
=============================================================

One dataclass holds everything a run depends on, so a study area is described in a single
place rather than scattered across notebook cells. `Config.manila()` and
`Config.for_city()` cover the common cases.

Directories are created on construction, using absolute paths, so a run cannot silently
write somewhere unexpected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Feature order is the single source of truth for the design matrix. Training and inference
# both build their columns by iterating this list, which is what keeps them aligned; earlier
# versions listed the predictors separately in each place and drifted out of order.
FEATURE_ORDER: tuple[str, ...] = (
    "blue", "green", "red", "nir", "swir16", "swir22",
    "dem", "ndvi", "ndbi", "ndwi",
)

# ESA WorldCover class codes.
WC_CLASSES: dict[int, str] = {
    10: "Tree", 20: "Shrub", 30: "Grass", 40: "Crop", 50: "Built-up", 60: "Bare",
    70: "Snow", 80: "Water", 90: "Wetland", 95: "Mangrove", 100: "Moss",
}

# The hour test in the granule filter is an inclusive range and cannot wrap midnight, so
# night is requested as two windows and the results concatenated.
DAY_WINDOWS: tuple[tuple[int, int], ...] = ((6, 18),)
NIGHT_WINDOWS: tuple[tuple[int, int], ...] = ((18, 23), (0, 5))


@dataclass
class Config:
    """Everything a pipeline run needs to know.

    Parameters
    ----------
    name : str
        Short identifier used in every output filename.
    bbox : tuple
        (west, south, east, north) in EPSG:4326.
    crs : str
        Projected CRS for all analysis, e.g. ``"EPSG:32651"``.
    timezone : str
        IANA zone used to convert acquisition times to local hours.
    mode : str
        ``"Day"`` or ``"Night"``. Sets the acquisition windows and tags all outputs.
    start, end : str
        Date range, ``YYYY-MM-DD``.
    base_dir : str
        Root for outputs. Defaults to the working directory.
    """

    name: str
    bbox: tuple[float, float, float, float]
    crs: str
    timezone: str
    mode: str = "Day"
    start: str = "2024-01-01"
    end: str = "2026-12-31"
    base_dir: str = ""

    # --- acquisition -------------------------------------------------------------
    valid_pixel_threshold: float = 0.30   # minimum valid fraction to download a granule
    max_cloud_cover: int = 20             # per-scene AOI cloud limit, %
    download_retries: int = 4             # LP DAAC returns transient 502/503 errors

    # --- covariates --------------------------------------------------------------
    res_coarse: int = 70                  # ECOSTRESS working resolution, m
    res_fine: int = 10                    # output resolution, m
    s2_window_days: int = 15              # optical pairing window, ± days
    s2_cloud_ladder: Sequence[int] = (10, 30, 50, 80)

    # --- sampling and model ------------------------------------------------------
    px_per_scene: int = 20_000
    test_size: float = 0.30
    n_strata: int = 10                    # temperature strata for the stratified split
    rf_estimators: int = 100
    rf_max_depth: int = 15
    rf_min_leaf: int = 5
    seed_model: int = 79                  # kept from the original implementation
    seed_split: int = 11

    # --- products ----------------------------------------------------------------
    hot_threshold_c: float | None = None  # defaults by mode in __post_init__
    hotspot_percentile: int = 90
    min_obs_per_year: int = 3

    dirs: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- constructors
    @classmethod
    def manila(cls, mode: str = "Day", **kw) -> "Config":
        """Tondo, Manila - the primary study area."""
        return cls(name="manila", bbox=(120.90, 14.55, 121.04, 14.70),
                   crs="EPSG:32651", timezone="Asia/Manila", mode=mode, **kw)

    @classmethod
    def valenzuela(cls, mode: str = "Day", **kw) -> "Config":
        """Valenzuela City - near-transfer control.

        Same metropolitan area, climate and terrain as Tondo. The southern edge is set to
        14.70 degrees, the northern edge of the Manila study area, so the two do not
        overlap and the model is never scored on ground it learned from.
        """
        return cls(name="valenzuela", bbox=(120.94, 14.70, 121.03, 14.78),
                   crs="EPSG:32651", timezone="Asia/Manila", mode=mode, **kw)

    @classmethod
    def addis(cls, mode: str = "Day", **kw) -> "Config":
        """Addis Ababa - cross-climate transfer test, roughly 2,300 m elevation."""
        kw.setdefault("start", "2024-10-01")
        kw.setdefault("end", "2025-05-31")
        return cls(name="addis", bbox=(38.68, 8.90, 38.88, 9.08),
                   crs="EPSG:32637", timezone="Africa/Addis_Ababa", mode=mode, **kw)

    @classmethod
    def for_city(cls, name, bbox, crs, timezone, **kw) -> "Config":
        """Any other study area."""
        return cls(name=name, bbox=bbox, crs=crs, timezone=timezone, **kw)

    # ---------------------------------------------------------------- setup
    def __post_init__(self):
        if self.mode not in ("Day", "Night"):
            raise ValueError(f"mode must be 'Day' or 'Night', got {self.mode!r}")

        if self.hot_threshold_c is None:
            # Night surfaces rarely approach the daytime threshold, so a 35 C night map
            # would be almost empty. 26 C instead flags warm nights, which is the
            # health-relevant signal: heat that does not release overnight.
            self.hot_threshold_c = 35.0 if self.mode == "Day" else 26.0

        if self.base_dir:
            base = Path(self.base_dir)
        else:
            # Resolve to the project root rather than the working directory, so that a
            # notebook in notebooks/ and a script at the root write to the same place.
            # Without this, outputs land in notebooks/products/ and the later notebooks
            # cannot find what the pipeline produced.
            cwd = Path.cwd()
            base = cwd.parent if (cwd.name == "notebooks"
                                  and (cwd.parent / "lstpipe").is_dir()) else cwd
        base.mkdir(parents=True, exist_ok=True)
        self.base_dir = str(base.resolve())

        self.dirs = {
            "products":  base / "products",
            "scenes":    base / "products" / f"per_scene_{self.mode.lower()}",
            "raw":       base / "products" / "raw_70m",
            "quicklooks": base / "quicklooks",
            "heatmaps":  base / "heatmaps",
            "models":    base / "models",
            "transfer":  base / "transfer_tests",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # Public S3 buckets (Copernicus DEM, ESA WorldCover) are unsigned.
        os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

    # ---------------------------------------------------------------- helpers
    @property
    def windows(self) -> tuple[tuple[int, int], ...]:
        """Local-hour windows to request, as (start, end) inclusive pairs."""
        return DAY_WINDOWS if self.mode == "Day" else NIGHT_WINDOWS

    @property
    def features(self) -> tuple[str, ...]:
        return FEATURE_ORDER

    def path(self, kind: str, filename: str) -> Path:
        """Absolute path inside one of the output directories."""
        return self.dirs[kind] / filename

    @property
    def key(self) -> str:
        """Identifier for this run: study area and mode.

        Every cache and product filename is scoped by this, so runs over different areas
        can share an output root without overwriting one another. Scoping by mode alone
        was not enough: a transferability test reuses the source area's directories, and
        an area-agnostic name let it load the wrong manifest and silently harvest the
        wrong granules.
        """
        return f"{self.name}_{self.mode}"

    def cache(self, what: str) -> Path:
        """Absolute path to one of this run's cache files.

        Parameters
        ----------
        what : {"manifest", "harvest", "model", "model_meta"}
        """
        names = {"manifest":   ("products", f"scenes_{self.key}.json"),
                 "harvest":    ("products", f"harvest_{self.key}.npz"),
                 "model":      ("models",   f"{self.key}_rf.joblib"),
                 "model_meta": ("models",   f"{self.key}_rf_meta.json")}
        if what not in names:
            raise KeyError(f"unknown cache {what!r}; expected one of {sorted(names)}")
        kind, fname = names[what]
        return self.dirs[kind] / fname

    def tag(self, date: str, utc: str) -> str:
        """Standard filename stem for one overpass."""
        return f"{self.name}_{date}_{utc}_{self.mode}"

    def summary(self) -> str:
        w = ", ".join(f"{a:02d}-{b:02d}h" for a, b in self.windows)
        return (f"{self.name} · {self.mode} · {self.start} to {self.end}\n"
                f"  bbox {self.bbox} | {self.crs}\n"
                f"  local windows {w} | quality threshold {self.valid_pixel_threshold}\n"
                f"  {self.res_coarse} m -> {self.res_fine} m | S2 window ±{self.s2_window_days} d\n"
                f"  outputs {self.base_dir}")
