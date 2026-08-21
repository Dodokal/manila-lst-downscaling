"""
Granule discovery, quality screening and resilient download.
============================================================

Selects which ECOSTRESS L2T_LSTE granules are worth retrieving, before spending bandwidth
on them. Each candidate's LST raster is opened *over HTTPS* and its valid-pixel fraction
measured; only granules above the threshold are downloaded. Over a multi-year archive this
is the difference between hours and days of transfer.

Downloads retry with exponential backoff and, if a granule still fails, it is skipped
rather than aborting the run. LP DAAC intermittently returns 502 and 503, and losing an
entire run to one transient server error after fifty successful granules is not acceptable.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import earthaccess
import rasterio
import rioxarray


@dataclass
class Scene:
    """One screened, downloaded overpass and the local paths to its rasters."""
    granule: Any
    lst_file: str
    water_file: str
    qc_file: str
    cloud_file: str
    acquired: datetime          # UTC
    local_hour: int

    @property
    def date(self) -> str:
        return self.acquired.strftime("%Y-%m-%d")

    @property
    def utc(self) -> str:
        return self.acquired.strftime("%H%M")

    @property
    def key(self) -> str:
        return f"{self.date}_{self.utc}"


def ensure_login(persist: bool = True, strategy: str = "all") -> bool:
    """Authenticate with NASA Earthdata if the session is not already active.

    Searching the catalogue is unauthenticated, but streaming and downloading files are
    not, so a missing login surfaces late and confusingly - the granule search succeeds and
    the first file access raises ``AttributeError`` on an uninitialised store. This checks
    first and logs in only when needed, so the failure cannot happen mid-run.

    On first use you are prompted for credentials; ``persist=True`` writes them to
    ``~/.netrc`` so later sessions authenticate silently.
    """
    auth = getattr(earthaccess, "__auth__", None)
    if auth is not None and getattr(auth, "authenticated", False):
        return True
    try:
        earthaccess.login(strategy=strategy, persist=persist)
    except TypeError:                      # older signatures take no strategy
        earthaccess.login(persist=persist)
    auth = getattr(earthaccess, "__auth__", None)
    ok = bool(auth is not None and getattr(auth, "authenticated", False))
    print("Earthdata login: " + ("active" if ok else "FAILED - check your credentials"))
    return ok


def _acquired(granule) -> datetime:
    ts = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def local_hour(granule, timezone: str) -> int:
    return _acquired(granule).astimezone(ZoneInfo(timezone)).hour


def manifest_path(cfg):
    """Where the screening manifest for this configuration lives."""
    return cfg.cache("manifest")


def save_manifest(cfg, scenes: list) -> str:
    """Record which granules passed screening and where their files are.

    Screening streams every candidate granule over HTTPS to measure its valid-pixel
    fraction. That is worth doing once; repeating it on every notebook is the single
    largest avoidable cost in the workflow.
    """
    import json
    payload = [{"lst_file": s.lst_file, "water_file": s.water_file,
                "qc_file": s.qc_file, "cloud_file": s.cloud_file,
                "acquired": s.acquired.isoformat(), "local_hour": s.local_hour}
               for s in scenes]
    p = manifest_path(cfg)
    with open(p, "w") as fh:
        json.dump({"mode": cfg.mode, "study_area": cfg.name, "bbox": list(cfg.bbox),
                   "start": cfg.start, "end": cfg.end, "n": len(payload),
                   "scenes": payload}, fh, indent=1)
    return str(p)


def load_manifest(cfg) -> list | None:
    """Restore screened scenes from the manifest, verifying the files still exist.

    Returns None if the manifest is absent or any referenced file has been moved, so a
    stale manifest degrades into a normal screening run rather than a confusing failure.
    """
    import json
    p = manifest_path(cfg)
    if not p.exists():
        return None
    with open(p) as fh:
        data = json.load(fh)

    # Refuse a manifest written for a different area or window. Filenames are scoped, so
    # this should not occur, but a mismatched manifest would harvest the wrong granules
    # against the wrong grid and fail far from the cause.
    if data.get("study_area") != cfg.name or data.get("mode") != cfg.mode:
        print(f"manifest is for {data.get('study_area')} {data.get('mode')}, "
              f"not {cfg.name} {cfg.mode} - re-screening")
        return None
    if data.get("bbox") and tuple(data["bbox"]) != tuple(cfg.bbox):
        print("manifest was written for a different bounding box - re-screening")
        return None

    scenes = []
    for r in data["scenes"]:
        if not all(os.path.exists(r[k]) for k in
                   ("lst_file", "water_file", "qc_file", "cloud_file")):
            print(f"manifest references missing files - re-screening")
            return None
        scenes.append(Scene(granule=None, lst_file=r["lst_file"],
                            water_file=r["water_file"], qc_file=r["qc_file"],
                            cloud_file=r["cloud_file"],
                            acquired=datetime.fromisoformat(r["acquired"]),
                            local_hour=r["local_hour"]))
    scenes.sort(key=lambda s: s.acquired)
    return scenes


def search(cfg) -> list:
    """Search ECOSTRESS L2T_LSTE over the study area and date range."""
    ensure_login()
    granules = earthaccess.search_data(
        short_name="ECO_L2T_LSTE", version="002",
        temporal=(cfg.start, cfg.end), bounding_box=cfg.bbox)
    print(f"granules intersecting {cfg.name}: {len(granules)}")
    return granules


def valid_fraction(url: str, fs) -> float:
    """Fraction of valid pixels, measured by streaming the raster over HTTPS."""
    with rioxarray.open_rasterio(fs.open(url), masked=True) as src:
        return float(src.squeeze().load().notnull().mean())


def _download(urls, directory: str, retries: int = 4, wait: float = 5.0):
    """Download with exponential backoff. Raises only after every attempt fails."""
    last = None
    for attempt in range(retries):
        try:
            return earthaccess.download(urls, local_path=directory)
        except Exception as exc:                       # noqa: BLE001 - server-side variety
            last = exc
            if attempt < retries - 1:
                print(f"      {type(exc).__name__}, retry {attempt + 1}/{retries - 1} "
                      f"in {wait:.0f}s")
                time.sleep(wait)
                wait *= 2
    raise last


def screen_and_download(cfg, granules=None, cache: bool = True,
                        rebuild: bool = False) -> list[Scene]:
    """Screen granules on quality and download those that pass.

    Night runs request two windows (evening and pre-dawn) because the hour test is an
    inclusive range and cannot wrap midnight.

    Parameters
    ----------
    cache : bool
        Reuse a previous screening manifest when one exists. Screening streams every
        candidate granule over HTTPS, so repeating it costs minutes to hours for no gain.
        Set ``rebuild=True`` after changing the date range or quality threshold.
    """
    if cache and not rebuild:
        cached = load_manifest(cfg)
        if cached:
            print(f"{cfg.mode}: {len(cached)} scenes restored from manifest "
                  f"({manifest_path(cfg).name}) - pass rebuild=True to re-screen")
            return cached

    if granules is None:
        granules = search(cfg)
    else:
        ensure_login()

    fs = earthaccess.get_fsspec_https_session()
    scenes: list[Scene] = []
    seen: set[str] = set()
    skipped_quality = skipped_download = 0

    for start_hour, end_hour in cfg.windows:
        directory = f"{cfg.name}_{start_hour}_{end_hour}_ecostress_data"
        print(f"\nwindow {start_hour:02d}-{end_hour:02d}h local")

        for granule in granules:
            hour = local_hour(granule, cfg.timezone)
            if not (start_hour <= hour <= end_hour):
                continue

            lst_url = next(l for l in granule.data_links() if l.endswith("_LST.tif"))
            gid = lst_url.split("/")[-1]
            if gid in seen:
                continue
            seen.add(gid)

            frac = None
            for attempt in range(2):
                try:
                    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                                      GDAL_HTTP_MAX_RETRY=10, GDAL_HTTP_RETRY_DELAY=0.5):
                        frac = valid_fraction(lst_url, fs)
                    break
                except Exception:                      # noqa: BLE001
                    if attempt == 0:                   # session probably expired
                        earthaccess.login(strategy="netrc")
                        fs = earthaccess.get_fsspec_https_session()
                    else:
                        frac = None
            if frac is None or frac < cfg.valid_pixel_threshold:
                skipped_quality += 1
                continue

            water_url = next(l for l in granule.data_links() if l.endswith("_water.tif"))
            qc_url = next(l for l in granule.data_links() if l.endswith("_QC.tif"))
            cloud_url = next((l for l in granule.data_links()
                              if l.endswith("_cloud.tif")), None)

            os.makedirs(directory, exist_ok=True)
            try:
                paths = _download([lst_url, water_url, qc_url, cloud_url],
                                  f"./{directory}", retries=cfg.download_retries)
            except Exception as exc:                   # noqa: BLE001
                skipped_download += 1
                print(f"    skipped {gid[-42:]}: {type(exc).__name__} after retries")
                continue

            acq = _acquired(granule)
            scenes.append(Scene(granule=granule, lst_file=str(paths[0]),
                                water_file=str(paths[1]), qc_file=str(paths[2]),
                                cloud_file=str(paths[3]), acquired=acq,
                                local_hour=hour))
            print(f"    {acq:%Y-%m-%d %H%M}UTC ({hour:02d}h local) valid {frac:.2f}")

    scenes.sort(key=lambda s: s.acquired)
    if cache and scenes:
        save_manifest(cfg, scenes)
    print(f"\n{cfg.mode}: {len(scenes)} scenes downloaded "
          f"({skipped_quality} below quality threshold, {skipped_download} download failures)")
    if skipped_download:
        print("  re-run this cell later to pick up the failures; existing files are skipped")
    return scenes
