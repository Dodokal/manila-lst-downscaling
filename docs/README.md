# High-Resolution Urban Heat Mapping - Tondo, Manila

Downscaling ECOSTRESS land surface temperature from 70 m to 10 m, one map per satellite
overpass, for neighbourhood-scale heat-vulnerability assessment.

Geo-Humanitarian Research (Z_GIS), Paris Lodron University of Salzburg
in partnership with Médecins Sans Frontières

---

## Layout

```
project/
├── lstpipe/              the pipeline - 11 modules, ~2,400 lines
├── notebooks/            5 notebooks - ~160 lines of code in total
├── tests/                equivalence and unit tests
├── docs/                 this file, MODULES.md, OUTPUTS.md
└── requirements.txt
```

Processing lives in the package; the notebooks are a thin, readable interface to it. A
production run is five function calls.

## Quick start

```python
from lstpipe import Config, run_pipeline

cfg = Config.manila(mode="Day")     # or "Night"
result = run_pipeline(cfg)
```

Run `notebooks/01_pipeline.ipynb` twice - once per mode - then the others in order.

| Notebook | Purpose | Needs |
|---|---|---|
| `01_pipeline` | produce all maps and composites | Earthdata login |
| `01b_pipeline_stepwise` | the same, with intermediates exposed | for development |
| `02_validation` | diagnostics and accuracy analysis | `01` for that mode |
| `03_transferability` | Addis Ababa and Valenzuela tests | `01` |
| `04_synthesis` | day–night layers, delivery inventory | `01` for **both** modes |

## Requirements

Python 3.11+, a NASA Earthdata account, and `requirements.txt`. Authenticate once with
`earthaccess.login(persist=True)`. `AWS_NO_SIGN_REQUEST` is set automatically for the
public Copernicus DEM and WorldCover buckets.

## Method

1. **Screen** every ECOSTRESS granule over HTTPS; download only those that pass.
2. **Mask** with the QC bitmask plus water and cloud masks.
3. **Repair** documented detector gaps (error codes −9998 / −9997) by along-track
   interpolation. Filled pixels are excluded from training, retained in maps.
4. **Pair** each thermal scene with its own Sentinel-2 composite from a ±15-day window.
5. **Train** one Random Forest on the pooled samples, predicting the anomaly.
6. **Apply** per overpass, restore that scene's level, and apply a mass-conserving residual
   correction.
7. **Validate** on a held-out 30 % stratified by temperature, per scene, by leave-scene-out
   cross-validation, and by transfer to two other cities.

## Two decisions that determine the results

**Per-scene Sentinel-2 pairing.** An earlier version reused one multi-year optical
composite for every thermal scene, so the predictors were identical across overpasses and
the model could not distinguish them - pooled R² ≈ 0.002. Pairing per scene cut the median
optical–thermal gap from 57 days to about two and is the single largest improvement in the
project's history.

**Anomaly target.** Scenes here are internally uniform, often under 1 K of spread, while
the level differs by tens of degrees between overpasses. Asking for absolute temperature
forces the model to infer that level from reflectance alone - impossible, since weather,
season and hour leave no optical signature. Being 6 K wrong on a scene with 1 K of internal
spread drives R² to about −40 even when the pattern is correct. The level comes from
ECOSTRESS instead, through the residual correction.

## Accuracy (Day, 31 overpasses, 2024–2026)

| Metric | Value |
|---|---|
| Mean absolute error | **2.34 K** |
| RMSE | **3.60 K** |
| Spatial-pattern R² | **0.157** |
| Median per-scene pattern R² | 0.066 |
| Leave-scene-out CV R² | 0.012 |
| Median optical–thermal gap | 2.0 days |

**Quote the pattern R².** A second figure near 0.8 exists for reconstructed absolute
temperature; it carries identical MAE and RMSE and differs only because the variance
denominator grows once the scene level is added back. It is not additional skill.

## Known limits

- **Land surface temperature is not air temperature.** Radiometric surface readings; 50 °C
  on a midday roof is normal and says nothing directly about ambient conditions.
- **Hot-day counts are not a climate trend.** Per-year exceedance correlates at +0.96 with
  the mean temperature of whichever scenes were cloud-free. Report as observed-overpass
  exceedance, pooled across years.
- **Skill scales with scene contrast** (r = +0.50). On near-uniform scenes there is little
  to resolve, so R² goes negative by construction, not by failure.
- **Elevation acts as a location proxy** in flat terrain, which is why cross-climate
  transfer fails.
- **Per-scene coverage 52–63 %**, set by the ECOSTRESS swath and masking.
- **Time of day changes the spatial pattern**, not only the level: a single all-day model
  scores −0.06 on afternoon scenes where a band-specific model scores +0.17.

## Transferability

| Test | City | Result |
|---|---|---|
| Cross-climate | Addis Ababa, 2,300 m | as-is **R² −2.16**; without elevation −0.02; retrained **+0.32** |
| Near-transfer | Valenzuela City | same climate and terrain, no overlap with training |

Addis fails because elevation there lies entirely outside the trained range (0 % feature
overlap) and a Random Forest cannot extrapolate. Valenzuela isolates whether that was
environmental mismatch or a limitation of the method.

**The method transfers; the fitted model does not.** Retraining takes hours; the pipeline
is unchanged.

## Tests

```bash
python tests/test_equivalence.py    # refactor reproduces the original numerics exactly
python tests/test_units.py          # component behaviour
python tests/test_session.py        # restoring from disk gives identical metrics
python tests/test_paths.py          # output paths independent of working directory
python tests/test_isolation.py      # study areas cannot read each other's caches
```

`test_equivalence.py` compares the package against an inline reimplementation of each
original notebook cell and requires agreement to 1e-12. It is the guarantee that
reorganising the code changed no result.

## Documentation

| File | Contents |
|---|---|
| `docs/README.md` | this file |
| `docs/MODULES.md` | every module, its public interface, and why it exists |
| `docs/OUTPUTS.md` | catalogue of every file the pipeline writes |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Transfer test reports "no usable scenes" | Caused by an area-agnostic cache name in versions before 2.0.1; each area now has its own. Run `session.migrate(cfg)`, or pass `rebuild=True` to `transfer.evaluate`. |
| `no cached run for ... Day` in notebook 02 | `01_pipeline` has not completed for that mode, or the two notebooks resolved different output roots. Every notebook prints `outputs -> ...`; the paths must match. |
| `AttributeError: 'NoneType' object has no attribute 'get_fsspec_session'` | Not authenticated. The catalogue search succeeds without credentials, so this appears only at the first file access. Run `earthaccess.login(persist=True)`; the pipeline also calls it automatically. |
| `ModuleNotFoundError: lstpipe` | The package is not on the path. Notebooks add the parent directory themselves; if importing elsewhere, keep `lstpipe/` beside the notebook or one level up. |
| `DownloadFailure ... 502` | LP DAAC returned a transient server error. Downloads retry with backoff and skip a granule that still fails; re-run later to pick it up, since existing files are skipped. |
| `MemoryError` while loading land cover | Only occurs if the tile is read before clipping. `covariates.build_grids` clips first. |
| `per_scene_maps needs the per-scene rasters` | The harvest came from cache. Pass `rebuild=True`, or use `run_pipeline`. |
| Blocky, low-detail maps | Low `sharpened_frac` - optical imagery was cloudy for those scenes. Check `diagnostics.scene_quality`; widening `s2_window_days` may help. |
| No scenes pass screening | Lower `valid_pixel_threshold`, widen the date range, or raise `max_cloud_cover`. |
