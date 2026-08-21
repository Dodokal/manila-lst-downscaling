# Output catalogue

Every path derives from `BASE_DIR`, set in Part 1. `<mode>` is `Day` or `Night`.

---

## Raster products

### Per-overpass - `products/per_scene_<mode>/`

| File | Content |
|---|---|
| `manila_<date>_<utc>_<mode>_lst_10m_C.tif` | Downscaled 10 m LST, °C - **the primary deliverable** |
| `manila_<date>_<utc>_<mode>_lst_70m_C.tif` | The ECOSTRESS observation it was derived from, °C |

Single band, float32, NaN nodata, band described as `LST` with a `units` tag. `<utc>` is
the acquisition time in UTC; local time is UTC+8.

Each 10 m map is mass-conserving: aggregated back to 70 m it reproduces its paired
observation. That property is verified by the residual diagnostics.

### Composites - `products/`

| File | Content |
|---|---|
| `manila_MEAN_LST_<mode>_10m.tif` | Mean across all overpasses - chronic heat |
| `manila_HOTSPOT_FREQ_<mode>_10m.tif` | Fraction of overpasses in the hottest 10 %, 0–1 |
| `manila_P95_LST_<mode>_10m.tif` | 95th percentile - extreme conditions |
| `manila_HOTDAYS_<year>_<mode>_10m.tif` | Days per year exceeding the threshold |
| `manila_HOTDAYS_MEAN_<mode>_10m.tif` | Mean across years |
| `manila_HOTDAYS_TREND_all_<mode>_10m.tif` | Per-pixel slope, days/yr per year |
| `manila_HOTDAYS_TREND_p05_<mode>_10m.tif` | The same, masked to p < 0.05 |

Hot-day layers describe **observed overpasses**, not daily means, and their year-to-year
differences are driven by overpass sampling. See README §7.

### Day–night layers - `products/` (both modes required)

| File | Content |
|---|---|
| `manila_DIURNAL_RANGE_10m.tif` | Mean day minus mean night, K |
| `manila_COMPOUND_RISK_DayNight_10m.tif` | Hot by day **and** by night, flag |
| `manila_NO_RELIEF_ZONE_10m.tif` | Warm night with a small diurnal range, flag |
| `manila_HOTSPOT_FREQ_BOTH_10m.tif` | Minimum of the day and night frequencies |

`NO_RELIEF_ZONE` is the layer most directly tied to health risk: heat that persists
overnight leaves no recovery window.

---

## Tables

### Inventory and quality - `products/`

| File | Content |
|---|---|
| `inventory_<mode>.csv` | One row per scene: date, optical gap, coverage, `sharpened_frac`, mean, p95, path |
| `inventory_graded_<mode>.csv` | The same, graded GOOD / PARTIAL / COARSE |
| `delivery_list_<mode>.txt` | Paths cleared for delivery |
| `s2_gap_perscene_<mode>.csv` | Optical–thermal gap per scene, in days |

`sharpened_frac` is the share of each map genuinely driven by the model rather than
gap-filled from upsampled ECOSTRESS. It is the honest measure of how much of a map is
truly 10 m.

### Accuracy - `products/`

| File | Content |
|---|---|
| `accuracy_by_scene_<mode>.csv` | Per-scene MAE, RMSE, R² for pattern and absolute, plus each scene's own spread |
| `cv_scene_grouped_<mode>.csv` | Leave-scene-out cross-validation folds |
| `feature_importance_<mode>.csv` | Predictor importances |
| `localhour_stratification_<mode>.csv` | Pooled versus band-specific skill by time of day |
| `localhour_composition_<mode>.csv` | Scenes and level spread per band |

Always read `scene_std_K` beside a per-scene R². A near-uniform scene offers little to
resolve, so a low or negative R² there is expected rather than diagnostic.

### Land cover - `products/`

| File | Content |
|---|---|
| `zonal_stats_lulc_<mode>.csv` | Per scene per class: mean, median, std, p95 |
| `zonal_summary_<mode>.csv` | Averaged across scenes |
| `lulc_boxplot_summary.csv` | Quartiles per class per mode |
| `daynight_by_lulc.csv` | Day, night and diurnal range per class |
| `daynight_deviation_by_lulc.csv` | Per-class deviation, sorted by cooling |
| `inference_by_lulc_<mode>.csv` | Held-out accuracy per class |
| `class_model_gain_<mode>.csv` | Whether class-specific models would help |
| `detail_by_lulc_<mode>.csv` | Sub-70 m structure added, per class |
| `residual_by_lulc_<mode>.csv` | Residual bias per class |

### Residual diagnostics - `products/`

| File | Content |
|---|---|
| `residual_10m_<mode>.csv` | Mass closure, sub-70 m std, detail ratio per scene |
| `residual_tails_<mode>.csv` | Residual percentiles |
| `residual_bias_by_scene_<mode>.csv` | Signed bias per scene |
| `mass_conservation_<mode>.csv` | Aggregation check against the observation |
| `strata_balance_<mode>.csv` | Train/test counts per temperature stratum |
| `hist_by_LST_<mode>.csv`, `hist_by_LULC_<mode>.csv` | Sample composition |

**Detail ratio** is the sub-70 m standard deviation divided by the scene's total spread. A
pure upsample scores 0. It is the most direct single measure of whether the product is
downscaling or interpolating.

### Transfer tests - `addis_test/`, `valenzuela_test/`

| File | Content |
|---|---|
| `transferability_<mode>.csv` | Model as-is, without elevation, and retrained locally |
| `feature_overlap_<mode>.csv` | Share of test-city pixels inside the trained range, per predictor |

Read `feature_overlap` before the accuracy figures. A predictor at 0 % overlap is being
extrapolated, and a Random Forest cannot extrapolate - it clamps to the edge of its
training range.

---

## Figures - `quicklooks/`

| File | Content |
|---|---|
| `<tag>_70m_vs_10m.png` | Side-by-side comparison per scene |
| `planner_products_<mode>.png` | Mean, hot-spot frequency, p95 |
| `hotdays_trend_<mode>.png` | Hot days per year and the significant trend |
| `residual_hist_and_cdf.png` | Residual histogram, CDF, and absolute-error CDF |
| `residual_10m_lulc_tails_<mode>.png` | Sub-70 m detail, distribution, Q–Q, bias by class |
| `lulc_zonal_boxplots.png` | Temperature by land cover across overpasses |
| `daynight_deviation.png`, `daynight_compound_risk.png` | Day/night structure |
| `localhour_stratification_<mode>.png` | Scene level against overpass hour |
| `scene_quality_<mode>.png`, `contact_GOOD_<mode>.png` | Scene grading and a contact sheet |
| `hotcold_coverage_<mode>.png`, `histograms_<mode>.png` | Sample composition |
| `inference_by_lulc_<mode>.png` | Error and added detail by class |

`heatmaps/` holds interactive HTML versions of the same maps.

---

## Models - `models/`

| File | Content |
|---|---|
| `manila_lst_rf.joblib` | The trained Random Forest |
| `manila_lst_rf_meta.json` | Feature names, count, training date, split description |
| `rf_<mode>_<band>.joblib` | Band-specific models, if local-hour stratification was run |

The metadata file exists because a fitted Random Forest stores only the *number* of expected
columns, not their names or order. Loading a model without checking the recorded
`feature_names` against `FEATURE_ORDER` risks predictions that are silently wrong rather
than obviously broken.

```python
import joblib, json
rf   = joblib.load("models/manila_lst_rf.joblib")
meta = json.load(open("models/manila_lst_rf_meta.json"))
assert meta["feature_names"] == FEATURE_ORDER
```

---

## Reading the products in GIS

All rasters are EPSG:32651 (UTM 51N), float32, NaN nodata. Temperature layers are °C;
frequency and flag layers are 0–1.

A suggested display stretch for LST is the 2nd–98th percentile with a diverging blue-to-red
ramp. Applying the *same* stretch to a 70 m and 10 m pair is what makes the added detail
visible rather than an artefact of independent scaling.
