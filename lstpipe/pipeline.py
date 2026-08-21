"""
The pipeline as a single call.
==============================

`run_pipeline` executes acquisition through to composites and returns everything the later
notebooks need. Use it when you want the products; use the individual modules when you want
to inspect or vary an intermediate step.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import acquire, covariates as cov, harvest as hv, model as mdl, products as prd


@dataclass
class Result:
    """Everything one pipeline run produces."""
    cfg: object
    scenes: list
    grids: object
    harvest: object
    fit: object
    inventory: pd.DataFrame
    composites: dict

    def summary(self) -> str:
        return (f"{self.cfg.name} · {self.cfg.mode}\n"
                f"  {self.harvest.n_scenes} scenes · {self.harvest.X.shape[0]:,} samples\n"
                f"  {self.fit.summary()}\n"
                f"  {len(self.inventory)} maps -> {self.cfg.dirs['scenes']}")


def run_pipeline(cfg, since: str | None = None, save_model: bool = True,
                 make_composites: bool = True, cache: bool = True,
                 rebuild: bool = False, verbose: bool = True) -> Result:
    """Acquisition, harvest, training, per-scene maps and composites.

    Parameters
    ----------
    cfg : Config
    since : str, optional
        Ignore overpasses before this date.
    cache : bool
        Write the screening manifest and harvest cache, and reuse them if present. These
        are what :func:`lstpipe.session.load` restores, so the later notebooks need no
        downloads. Leave enabled unless deliberately forcing a clean run.
    rebuild : bool
        Ignore existing caches. Required after changing the date range, quality
        threshold, pairing window or feature list, none of which a cache can detect.
    """
    def step(n, title):
        if verbose:
            print(f"\n{'=' * 74}\n{n}. {title}\n{'=' * 74}")

    step(1, "ACQUISITION")
    scenes = acquire.screen_and_download(cfg, cache=cache, rebuild=rebuild)
    if not scenes:
        raise RuntimeError("no scenes passed screening - relax the threshold or widen dates")

    step(2, "REFERENCE GRIDS AND STATIC LAYERS")
    grids = cov.build_grids(cfg)

    step(3, "PER-SCENE HARVEST")
    h = hv.collect(cfg, scenes, grids, since=since, verbose=verbose,
                   cache=cache, rebuild=rebuild)

    step(4, "MODEL TRAINING")
    fit = mdl.train(cfg, h, verbose=verbose)
    if save_model:
        mdl.save(cfg, fit)

    step(5, "PER-OVERPASS MAPS")
    inv = prd.per_scene_maps(cfg, fit, h, grids, verbose=verbose)

    comp = {}
    if make_composites:
        step(6, "PLANNER COMPOSITES")
        comp = prd.composites(cfg, verbose=verbose)

    result = Result(cfg=cfg, scenes=scenes, grids=grids, harvest=h, fit=fit,
                    inventory=inv, composites=comp)
    if verbose:
        print(f"\n{'=' * 74}\nCOMPLETE\n{'=' * 74}")
        print(result.summary())
    return result
