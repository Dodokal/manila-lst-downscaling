"""
lstpipe - ECOSTRESS land surface temperature downscaling, 70 m to 10 m.
=======================================================================

A pipeline that produces one 10 m LST map per satellite overpass, each anchored to its own
thermal observation, plus the composite layers derived from them.

Typical use from a notebook::

    from lstpipe import Config, run_pipeline
    cfg = Config.manila(mode="Day")
    result = run_pipeline(cfg)

To analyse a completed run without repeating any of it::

    from lstpipe import Config, session
    cfg = Config.manila(mode="Day")
    h, fit = session.load(cfg)          # from disk only - no network

or step by step, when you want the intermediates::

    from lstpipe import Config, acquire, covariates, harvest, model, products
    cfg    = Config.manila(mode="Day")
    scenes = acquire.screen_and_download(cfg)
    grids  = covariates.build_grids(cfg)
    data   = harvest.collect(cfg, scenes, grids)
    fit    = model.train(cfg, data)
    maps   = products.per_scene_maps(cfg, fit, data, grids)

Modules
-------
config        study area, paths, constants - everything tunable in one place
acquire       granule search, streaming quality screen, resilient download
preprocess    QC masking, water/cloud removal, detector gap filling
covariates    reference grids, DEM, land cover, per-scene Sentinel-2 pairing
harvest       assembles the pooled training set across overpasses
model         training on the anomaly target, accuracy, persistence
products      per-scene maps, planner composites, hot-day layers
diagnostics   validation analyses (residuals, land cover, time of day, quality)
session       restore a completed run from disk, without network access
transfer      transferability testing against another city
viz           figures

Design notes
------------
Two decisions determine the results and are documented at their point of use:

* **Per-scene Sentinel-2 pairing** (`covariates.s2_for_date`). Each thermal scene is paired
  with optical imagery from within a short window of its own acquisition. Reusing one shared
  composite makes the predictors identical across overpasses, and the model then cannot
  distinguish scenes at all.

* **Anomaly target** (`model.train`). The model predicts each pixel's departure from its
  scene mean, not absolute temperature. Scene level depends on weather, season and hour,
  none of which are visible to Sentinel-2; it is supplied instead by ECOSTRESS itself
  through the residual correction.
"""

__version__ = "2.0.1"

from .config import Config
from .pipeline import run_pipeline
from . import session

__all__ = ["Config", "run_pipeline", "session",
           "acquire", "preprocess", "covariates", "harvest",
           "model", "products", "diagnostics", "transfer", "viz"]
