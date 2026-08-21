"""
Runs over different study areas must not read each other's caches.

This reproduces a real failure: a transferability test against Addis Ababa loaded Manila's
screening manifest, because the filename was keyed on acquisition mode alone. It then
harvested Manila granules against an Addis grid, matched nothing, and reported "no usable
scenes" - far from the actual cause.

Run: python tests/test_isolation.py
"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from lstpipe import Config
from lstpipe import acquire, harvest as hv, model as mdl, session

with tempfile.TemporaryDirectory() as d:
    manila = Config.manila(mode="Day", base_dir=d)
    addis = Config.addis(mode="Day", base_dir=d)
    night = Config.manila(mode="Night", base_dir=d)

    # every cache filename must be distinct across areas and modes
    for what in ("manifest", "harvest", "model", "model_meta"):
        paths = {manila.cache(what), addis.cache(what), night.cache(what)}
        assert len(paths) == 3, f"{what} collides: {paths}"
    assert manila.base_dir == addis.base_dir     # they do share an output root
    print("cache filenames distinct for manila/addis/night in a shared root: PASS")
    print(f"  {manila.cache('manifest').name} | {addis.cache('manifest').name}")

    # --- the original failure: Addis reading Manila's manifest ---
    from lstpipe.acquire import Scene
    from datetime import datetime, timezone as tz
    def _scene(i):
        f = Path(d) / f"g{i}.tif"; f.write_text("x")
        return Scene(granule=None, lst_file=str(f), water_file=str(f), qc_file=str(f),
                     cloud_file=str(f), acquired=datetime(2025, 1, i + 1, 8, tzinfo=tz.utc),
                     local_hour=8)
    acquire.save_manifest(manila, [_scene(i) for i in range(5)])
    assert acquire.load_manifest(manila) is not None, "manila must load its own manifest"
    assert acquire.load_manifest(addis) is None, "addis must NOT read manila's manifest"
    print("addis cannot load manila's manifest: PASS")

    # even if a manifest were placed at the addis path, the area stamp catches it
    import shutil, json
    shutil.copy(manila.cache("manifest"), addis.cache("manifest"))
    assert acquire.load_manifest(addis) is None, "area stamp must reject a foreign manifest"
    print("area stamp rejects a manifest copied to the wrong filename: PASS")

    # --- harvest caches must be equally isolated ---
    class _S:
        def __init__(s, h): s.local_hour = h
    class _SD:
        def __init__(s, h): s.scene = _S(h); s.s2_info = {"gap_days": 2.0, "n_items": 3}; s.lst = object()
    rng = np.random.default_rng(0)
    h = hv.Harvest(X=rng.random((500, 10)), y=300 + rng.random(500),
                   scene_key=np.array(["2025-01-15_0830"] * 500, dtype=object),
                   lulc=None, scenes={"2025-01-15_0830": _SD(8)}, gaps=pd.DataFrame())
    h.save(hv.cache_path(manila), key=manila.key)

    assert hv.cache_path(manila).exists() and not hv.cache_path(addis).exists()
    hv.Harvest.load(hv.cache_path(manila), expect_key=manila.key)      # correct owner
    try:
        hv.Harvest.load(hv.cache_path(manila), expect_key=addis.key)
        raise AssertionError("should have refused a foreign harvest cache")
    except ValueError as e:
        assert "belongs to" in str(e)
    print("harvest cache refuses to load under another run's key: PASS")

    # session must report the manila cache and not confuse it for addis
    assert session.available(manila)["harvest"] is True
    assert session.available(addis)["harvest"] is False
    try:
        session.load(addis, verbose=False)
        raise AssertionError("should have raised for addis")
    except FileNotFoundError as e:
        assert "addis" in str(e)
    print("session.available and session.load are per-run: PASS")

print("\nALL ISOLATION TESTS PASS")
