# Module reference

Eleven modules, each with one responsibility. Import order reflects the pipeline order.

| Module | Responsibility | Key entry points |
|---|---|---|
| `config` | study area, paths, every tunable constant | `Config`, `FEATURE_ORDER`, `WC_CLASSES` |
| `acquire` | granule search, streaming quality screen, resilient download | `screen_and_download`, `Scene` |
| `preprocess` | QC masking, water/cloud removal, detector gap filling | `qc_mask_lst`, `clip_and_mask`, `clean_scene` |
| `covariates` | reference grids, DEM, land cover, per-scene optical pairing | `build_grids`, `s2_for_date`, `stack`, `valid_mask` |
| `harvest` | pooled training set across overpasses | `collect`, `Harvest` |
| `model` | training on the anomaly target, accuracy, persistence | `train`, `save`, `load`, `metrics` |
| `products` | per-scene maps, composites, hot-day and day–night layers | `per_scene_maps`, `composites`, `hot_days`, `day_night_layers` |
| `diagnostics` | validation analyses | `extreme_coverage`, `residual_signs`, `sub_pixel_detail`, `by_landcover`, `by_local_hour`, `scene_quality` |
| `transfer` | transferability testing against another city | `evaluate`, `TransferResult` |
| `viz` | figures | `planner_products`, `residual_panel`, `landcover_boxplots`, `transfer_panel` |
| `pipeline` | the whole run as one call | `run_pipeline`, `Result` |

## Dependency order

```
config      (no internal dependencies)
   ↓
preprocess ─┐
acquire ────┤
covariates ─┴→ harvest → model → products → diagnostics
                  ↓                              ↓
               transfer                         viz
                  ↓
              pipeline
```

## What replaced what

The previous version had five imported modules plus eleven notebook cells kept as loose
`.py` files, and the analysis logic lived in the notebook itself - 2,944 lines across 91
cells, duplicated between the Day and Night notebooks.

| Was | Now |
|---|---|
| `granules_filter.py` | `acquire` - plus retry and skip on transient server errors |
| `preprocess_lst.py` + `eco_stripe_fill.py` | `preprocess` - both are scene cleaning |
| `build_composite.py` | `covariates.reference_grid` and `products.composites` |
| `comparison_plots.py` | `viz.scene_comparison` |
| `cell_lulc.py` | `covariates.build_grids` |
| `cell_train3.py` | `model.train` |
| `cell_apply3.py` | `products.per_scene_maps` |
| `cell_composite.py` | `products.composites` |
| `fb_cellE/F/G`, `cellH/I/J` | `diagnostics` |
| `cellK_valenzuela.py` + the Addis cell | `transfer.evaluate` - one function, any city |
| `finisher_*.py` | removed; superseded |

The Addis and Valenzuela tests were near-identical 200-line cells differing only in
configuration. They are now one function called twice, which is also what makes adding a
third city a two-line change.

## Three things to understand before editing

**`FEATURE_ORDER` is load-bearing.** Training and inference both build their design matrices
through `covariates.stack`, which iterates this one tuple. That is what keeps their columns
aligned; earlier versions listed the predictors separately in each place and drifted out of
order more than once, producing silently wrong predictions rather than an error.

**`filled_mask` separates training from delivery.** Pixels repaired by gap filling are
excluded from training but retained in delivered maps. The model is never fitted to
interpolated values, while the products still have complete coverage.

**Diagnostics return tables, not just figures.** Every analysis writes a CSV so results can
be compared across runs and quoted in a report. Plotting is deliberately separate, in
`viz`, so a figure is an optional view of a table rather than the only record.

## Adding a study area

```python
cfg = Config.for_city(
    name="cebu",
    bbox=(123.85, 10.25, 123.95, 10.35),
    crs="EPSG:32651",
    timezone="Asia/Manila",
    mode="Day",
)
result = run_pipeline(cfg)
```

Nothing else changes. For a transfer test against an existing model, pass the new config to
`transfer.evaluate` instead.

## Caching the harvest

Harvesting re-fetches a Sentinel-2 composite from AWS for every overpass, and that network
cost dominates the runtime - the ECOSTRESS granules themselves are already local.

The diagnostics only need the pooled arrays (`X`, `y`, `scene_key`, `lulc`), so those can be
cached and the harvest skipped entirely:

```python
h = harvest.collect(cfg, scenes, grids, cache=True)   # builds, then reuses
h = harvest.collect(cfg, scenes, grids, cache=True, rebuild=True)   # force a rebuild
```

The cache is a single `.npz` in `products/`, roughly 0.1 MB per 1,000 samples. It carries
the pooled arrays plus each scene's local hour, which is what `diagnostics.by_local_hour`
needs - so a cached harvest behaves identically for every diagnostic.

It deliberately does **not** carry the per-scene rasters, which are far larger and only
needed for map generation. `products.per_scene_maps` checks `harvest.has_rasters` and
raises a clear message if handed a cached harvest, rather than failing obscurely on a
missing array.

Rebuild after changing anything upstream: the date range, the quality threshold, the
pairing window, or the feature list.

## What is cached, and why

A pipeline run downloads granules, screens them over HTTPS, fetches optical imagery per
scene and trains a model. None of that needs repeating to analyse the result, but calling
the same functions again in a second notebook does exactly that. Three caches prevent it.

| Cache | File | Prevents |
|---|---|---|
| Screening manifest | `products/scenes_<area>_<mode>.json` | re-streaming every candidate granule to re-measure valid-pixel fraction |
| Harvest | `products/harvest_<area>_<mode>.npz` | re-fetching a Sentinel-2 composite for every overpass |
| Model | `models/<area>_<mode>_rf.joblib` | retraining |

Every name is scoped by **study area and mode**, built in one place (`Config.cache`). This
matters for transferability testing, where a second area shares the source area's output
root: an area-agnostic name previously let a transfer test load the source city's manifest
and harvest the wrong granules, failing with a misleading "no usable scenes". Each file
also carries its own area stamp, so a cache renamed or copied by hand is rejected rather
than silently misread.

Caches written by an earlier version are adopted automatically:

```python
session.migrate(cfg)     # renames legacy files; safe to call every time
```

`session.load(cfg)` uses all three:

```python
from lstpipe import Config, session
cfg = Config.manila(mode="Day")
h, fit = session.load(cfg)      # disk only - no network, no retraining
```

The train/test split is regenerated with the same seed and stratification used during
training, so every metric from a restored session matches the original run to machine
precision. This is verified in `tests/test_session.py`.

Only `01_pipeline.ipynb` downloads. Notebooks 02, 03 and 04 restore.

**Rebuild after changing the configuration.** The caches cannot detect a changed date
range, quality threshold, pairing window or feature list:

```python
scenes = acquire.screen_and_download(cfg, rebuild=True)
h      = harvest.collect(cfg, scenes, grids, cache=True, rebuild=True)
```
