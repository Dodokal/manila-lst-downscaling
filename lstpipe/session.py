"""
Restoring a completed run without touching the network.
=======================================================

A pipeline run downloads granules, screens them, fetches optical imagery and trains a
model. None of that needs repeating to analyse the result, yet the obvious way to write a
second notebook - call the same functions again - does exactly that.

:func:`load` restores a finished run from disk alone: the pooled training arrays from the
harvest cache and the fitted model from the model store. No search, no screening, no
imagery, no retraining. It is what the validation and transferability notebooks should use.

    from lstpipe import Config, session
    cfg = Config.manila(mode="Day")
    h, fit = session.load(cfg)

If nothing has been cached yet, it says so and points at the pipeline notebook rather than
silently starting a multi-hour job.
"""

from __future__ import annotations

import numpy as np

from . import harvest as hv
from . import model as mdl


def migrate(cfg, verbose: bool = True) -> int:
    """Rename caches written before filenames were scoped by study area.

    Earlier versions named these by acquisition mode alone, which let runs over different
    areas overwrite one another. This moves any legacy file to its scoped name so an
    existing run does not have to be repeated. Files that would overwrite an existing
    scoped cache are left alone.
    """
    legacy = [(cfg.dirs["products"] / f"scenes_{cfg.mode}.json", cfg.cache("manifest")),
              (cfg.dirs["products"] / f"harvest_{cfg.mode}.npz", cfg.cache("harvest")),
              (cfg.dirs["models"] / f"{cfg.name}_lst_rf_{cfg.mode}.joblib", cfg.cache("model")),
              (cfg.dirs["models"] / f"{cfg.name}_lst_rf_{cfg.mode}_meta.json",
               cfg.cache("model_meta"))]
    moved = 0
    for old, new in legacy:
        if old.exists() and not new.exists():
            old.rename(new)
            moved += 1
            if verbose:
                print(f"  {old.name} -> {new.name}")
    if verbose:
        print(f"migrated {moved} cache file(s)" if moved else "nothing to migrate")
    return moved


def available(cfg) -> dict:
    """What has been cached for this configuration."""
    return {"harvest": cfg.cache("harvest").exists(),
            "model": cfg.cache("model").exists(),
            "manifest": cfg.cache("manifest").exists(),
            "maps": len(list(cfg.dirs["scenes"].glob(f"*_{cfg.mode}_lst_*m_C.tif")))}


def load(cfg, verbose: bool = True):
    """Restore a completed run: the pooled samples and the fitted model.

    Returns
    -------
    (harvest, fit)
        ``fit`` carries the same accuracy figures as the original training run, recomputed
        from the restored arrays rather than re-fitting, so numbers match exactly.

    Raises
    ------
    FileNotFoundError
        If no harvest cache exists for this configuration.
    """
    have = available(cfg)
    if not have["harvest"]:
        # Report what *is* present, so the cause is obvious: a pipeline that was never
        # run, one run for the other mode, or outputs written under a different root.
        found = [k for k in ("manifest", "model") if have[k]]
        lines = [f"no cached run for {cfg.name} {cfg.mode}.",
                 f"  expected: {hv.cache_path(cfg)}",
                 f"  output root: {cfg.base_dir}"]
        if found:
            lines.append(f"  found instead: {', '.join(found)}"
                         + (f", {have['maps']} maps" if have["maps"] else ""))
        others = sorted(q.name for q in cfg.dirs["products"].glob("harvest_*.npz"))
        if others:
            lines.append(f"  caches present for other runs: {', '.join(others)}")
        lines += ["",
                  "  Run 01_pipeline.ipynb for this mode first. It writes the harvest",
                  "  cache and model that this function restores. If it has already been",
                  "  run, confirm both notebooks resolve the same output root above."]
        raise FileNotFoundError("\n".join(lines))

    h = hv.Harvest.load(hv.cache_path(cfg), expect_key=cfg.key)
    if verbose:
        print(f"restored {cfg.name} {cfg.mode}: {h.n_scenes} scenes · "
              f"{h.X.shape[0]:,} samples")

    if have["model"]:
        rf, meta = mdl.load(cfg)
        fit = _rebuild_fit(cfg, h, rf)
        if verbose:
            print(f"  model trained {meta['trained']}")
    else:
        if verbose:
            print("  no saved model - fitting from the cached samples "
                  "(seconds, no network)")
        fit = mdl.train(cfg, h, cross_validate=False, verbose=False)

    if verbose:
        print(f"  {fit.summary()}")
        if have["maps"]:
            print(f"  {have['maps']} maps on disk")
    return h, fit


def _rebuild_fit(cfg, h, rf) -> mdl.Fit:
    """Reconstruct a Fit around an already-trained model.

    The split is regenerated with the same seed and stratification as training, so the
    held-out set is identical and the reported metrics match the original run exactly.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    scene_mean = {k: h.y[h.scene_key == k].mean() for k in set(h.scene_key)}
    sm = np.array([scene_mean[k] for k in h.scene_key])
    y_anom = h.y - sm

    strata = mdl.temperature_strata(y_anom, cfg.n_strata)
    idx = np.arange(len(y_anom))
    i_tr, i_te = train_test_split(idx, test_size=cfg.test_size,
                                  random_state=cfg.seed_split, stratify=strata)

    pa, ta = rf.predict(h.X[i_te]), y_anom[i_te]
    pattern = mdl.metrics(ta, pa)
    absolute = mdl.metrics(h.y[i_te], pa + sm[i_te])

    imp = (pd.DataFrame({"feature": list(cfg.features)[:rf.n_features_in_],
                         "importance": rf.feature_importances_})
           .sort_values("importance", ascending=False))

    per_scene = pd.DataFrame()
    p = cfg.path("products", f"accuracy_by_scene_{cfg.mode}.csv")
    if p.exists():
        per_scene = pd.read_csv(p)

    return mdl.Fit(rf=rf, y_anom=y_anom, scene_mean=scene_mean,
                   train_idx=i_tr, test_idx=i_te, pattern=pattern, absolute=absolute,
                   per_scene=per_scene, importance=imp)
