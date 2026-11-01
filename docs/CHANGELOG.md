# Changelog

## v2.1.0 - response to MSF delivery feedback (06 Oct 2026)

| MSF point | v2.1 |
|---|---|
| Train in °C, not Kelvin | LST converted to °C when a granule is read; target, metrics, rasters and tables all °C |
| Do not train on anomalies | One Random Forest per overpass on absolute °C (ARSET design); spatial-block hold-out |
| Use available Sentinel-2, no scene thresholds | Scene cloud ladder removed; per-pixel SCL masking; ±15→30→60 d per-pixel filling. v2.0 also averaged cloudy acquisitions in without any pixel mask |
| Stripes still present | QC bits 2-3 now read; flagged lines + halo removed; scan-line destriping; heavily striped scenes rejected |
| Cloud-shaped artefacts | Column gap filler removed (it streaked cloud holes); cloud buffer; cold-outlier and 15 °C floor (Manila) |
| 2 °C / 4 °C values in July | Same as above; values outside the cleaned range never reach training or maps |
| Exclude unclean images such as 07/07/2024 | Scene grading A/B/C with logged reasons; 2024-07-07 rejected |
| Edges / waterfront cut, neighbourhoods missing | Shoreline pixels mapped but not trained on; maps cut at 10 m WorldCover shoreline; composites blank where < 5 clean overpasses; `OBS_COUNT` layer |
| Artefacts in MEAN layers | Two-way (pixel + overpass) composite removes footprint seams; Day composites 09-16 h only |
| Pixel uncertainty | `MEAN_LST_SE` per pixel; `OBS_COUNT` |
| "25+ days" / "205 days/yr" above threshold | v2.0 multiplied a fraction of overpasses by 365. Now `HOTFRAC_*` = fraction of clean overpasses |
| Land cover unavailable (Vienna) | WorldCover tile hard-coded to N12E120; now derived from the bounding box |
| Synthesis shows the last city's results | Tables were named by mode only; all now scoped by area and mode, synthesis uses one study-area variable |
| Same plot twice | Figures were both drawn and returned; `_save` now shows once and closes; land-cover tables merged |
| Same colour scale | Contact sheet uses one shared °C scale |
| Time of day: how to interpret | Boxplot of clean overpass median LST by hour; composites restricted to 09-16 h |
| Config needs kernel restart | Settings passed as `Config(...)` keywords in the notebook; `%autoreload` in every notebook |
| Quick start lacks prerequisites; no schema | README prerequisites table; data-flow diagram (`img/dataflow.png`) |
| Transferability notebook: needed? | Retired - one model per overpass makes it moot |
| Collection 1 or 2? | Collection 2 (`ECO_L2T_LSTE` v002, files `ECOv002_*`) - confirmed |

Breaking: v2.0 harvest caches and models are refused with a message; re-run `01_pipeline`
with `rebuild=True`. Hot-day outputs renamed `HOTDAYS_*` → `HOTFRAC_*`. Product tables
renamed `*_<mode>.csv` → `*_<area>_<mode>.csv`.
