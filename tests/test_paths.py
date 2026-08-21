"""
Output paths must not depend on which directory a notebook runs from, and the pipeline
must write the caches that `session.load` restores.

Run: python tests/test_paths.py
"""
import os, sys, tempfile, inspect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lstpipe import Config
from lstpipe.pipeline import run_pipeline

with tempfile.TemporaryDirectory() as d:
    root = Path(d) / "project"
    (root / "lstpipe").mkdir(parents=True)
    (root / "notebooks").mkdir()
    here = os.getcwd()
    try:
        os.chdir(root / "notebooks")
        a = Config.manila(mode="Day")
        os.chdir(root)
        b = Config.manila(mode="Day")
    finally:
        os.chdir(here)
    assert a.base_dir == b.base_dir == str(root.resolve()), (a.base_dir, b.base_dir)
    assert "notebooks" not in a.base_dir
print(f"base_dir identical from notebooks/ and project root: PASS")

# an explicit base_dir always wins
with tempfile.TemporaryDirectory() as d:
    c = Config.manila(mode="Day", base_dir=d)
    assert c.base_dir == str(Path(d).resolve())
print("explicit base_dir respected: PASS")

# Day and Night share a root but keep separate scene folders and caches
with tempfile.TemporaryDirectory() as d:
    day = Config.manila(mode="Day", base_dir=d)
    night = Config.manila(mode="Night", base_dir=d)
    assert day.base_dir == night.base_dir
    assert day.dirs["scenes"] != night.dirs["scenes"]
    from lstpipe import harvest as hv
    assert hv.cache_path(day) != hv.cache_path(night)
    assert hv.cache_path(day).parent == hv.cache_path(night).parent
print("Day and Night share a root, separate caches and scene folders: PASS")

# run_pipeline must propagate caching, or session.load can never succeed
src = inspect.getsource(run_pipeline)
assert "screen_and_download(cfg, cache=cache" in src, "manifest cache not propagated"
assert "cache=cache, rebuild=rebuild" in src, "harvest cache not propagated"
assert "cache: bool = True" in src, "caching must default on"
print("run_pipeline writes the manifest and harvest caches by default: PASS")

# the failure message must be diagnostic
from lstpipe import session
with tempfile.TemporaryDirectory() as d:
    cfg = Config.manila(mode="Day", base_dir=d)
    try:
        session.load(cfg)
        raise AssertionError("should have raised")
    except FileNotFoundError as e:
        msg = str(e)
        assert "expected:" in msg and "output root:" in msg and "01_pipeline" in msg
print("missing-cache error reports the expected path and output root: PASS")

print("\nALL PATH TESTS PASS")
