"""
Transferability testing - retired in v2.1.
==========================================

v2.0 trained one pooled model in Manila and asked how well it predicted Valenzuela and
Addis Ababa. In v2.1 every overpass gets its own model trained on that overpass's own
pixels, so there is no Manila model to transfer: a new city is simply run through
``01_pipeline`` with ``Config.for_city(...)``. MSF's review asked whether this notebook was
still needed; it is not. The v2.0 code is kept beside this file as
``_transfer_v20_legacy.py.txt`` for the record.
"""


def evaluate(*_, **__):
    raise RuntimeError("transferability testing was retired in v2.1 (one model per "
                       "overpass). Run 01_pipeline with Config.for_city(...) instead.")
