"""
Figures.
========

Plotting is kept separate from analysis so that every diagnostic returns a table that can
be compared across runs, and the figure is an optional view of it. Each function saves to
the quicklooks directory and returns the figure.
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .products import read_raster

THERMAL = "RdYlBu_r"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 12, "axes.titlesize": 13})
DAY_C, NIGHT_C, ACCENT, BLUE = "#D6604D", "#065A82", "#1C7293", "#4393C3"


def _save(cfg, fig, name, dpi=140):
    """Save, show once, and close.

    v2.0 returned the figure as well as letting the inline backend draw it, so Jupyter
    displayed every diagnostic figure twice.
    """
    path = cfg.path("quicklooks", f"{cfg.name}_{name}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return str(path)


def scene_comparison(cfg, path_coarse, path_fine, title=None):
    """Coarse and fine maps side by side on a shared colour scale.

    The shared scale is deliberate: identical stretch is what makes added detail visible
    rather than an artefact of independent scaling.
    """
    a, _, _ = read_raster(path_coarse)
    b, _, _ = read_raster(path_fine)
    vals = np.concatenate([a[np.isfinite(a)].ravel(), b[np.isfinite(b)].ravel()])
    vmin, vmax = np.percentile(vals, [2, 98])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].imshow(a, cmap=THERMAL, vmin=vmin, vmax=vmax)
    ax[0].set_title(f"ECOSTRESS {cfg.res_coarse} m")
    im = ax[1].imshow(b, cmap=THERMAL, vmin=vmin, vmax=vmax)
    ax[1].set_title(f"downscaled {cfg.res_fine} m")
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="LST (°C)")
    if title:
        fig.suptitle(title, fontweight="bold")
    _save(cfg, fig, f"{title or 'comparison'}.png".replace(" ", "_"))
    return None


def planner_products(cfg, comp: dict):
    """Mean, hot-spot frequency and 95th percentile."""
    arrays = comp["arrays"]
    ncol = min(3, len(arrays))
    nrow = int(np.ceil(len(arrays) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 5 * nrow))
    for a_ in np.atleast_1d(ax).ravel()[len(arrays):]:
        a_.axis("off")
    for a_, (name, arr) in zip(np.atleast_1d(ax).ravel(), arrays.items()):
        if "FREQ" in name:
            im = a_.imshow(arr, cmap="inferno", vmin=0, vmax=1)
            label = "fraction of overpasses"
        elif "COUNT" in name:
            im = a_.imshow(np.where(arr > 0, arr, np.nan), cmap="viridis")
            label = "clean overpasses"
        elif "_SE_" in name:
            v = arr[np.isfinite(arr)]
            im = a_.imshow(arr, cmap="magma_r", vmin=0, vmax=np.percentile(v, 98) if v.size else 1)
            label = "standard error (°C)"
        else:
            v = arr[np.isfinite(arr)]
            im = a_.imshow(arr, cmap=THERMAL, vmin=np.percentile(v, 2), vmax=np.percentile(v, 98))
            label = "LST (°C)"
        a_.set_title(name.replace("_", " "), fontweight="bold")
        a_.set_xticks([]); a_.set_yticks([])
        plt.colorbar(im, ax=a_, shrink=0.8, label=label)
    fig.suptitle(f"{cfg.name.title()} - {cfg.mode} planner products from {comp['n']} overpasses",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(cfg, fig, f"planner_products_{cfg.mode}.png")
    return None


def contact_sheet(cfg, max_scenes: int = 12, only_good: bool = True,
                  shared_scale: bool = True):
    """A grid of per-scene maps on one shared colour scale.

    With ``shared_scale`` (default) every panel uses the same °C range, so equal colours
    mean equal temperatures across panels - v2.0 stretched each panel separately, which made
    a dark blue mean 8 °C in one panel and 20 °C in the next.
    """
    inv_p = cfg.path("products", f"inventory_{cfg.key}.csv")
    if inv_p.exists():
        inv = pd.read_csv(inv_p)
        if only_good and "in_composites" in inv:
            inv = inv[inv.in_composites.astype(bool)]
        tifs = [t for t in inv.file_10m if os.path.exists(t)][:max_scenes]
    else:
        tifs = sorted(glob.glob(str(cfg.path("scenes",
                      f"{cfg.name}_*_{cfg.mode}_lst_{cfg.res_fine}m_C.tif"))))[:max_scenes]
    if not tifs:
        print("no maps to plot")
        return None

    arrays = [read_raster(t)[0] for t in tifs]
    if shared_scale:
        allv = np.concatenate([a[np.isfinite(a)][::7] for a in arrays])
        lo, hi = np.percentile(allv, [2, 98])
    cols = 3
    rows = int(np.ceil(len(tifs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for a_, t, v in zip(axes, tifs, arrays):
        vv = v[np.isfinite(v)]
        if vv.size:
            vmin, vmax = (lo, hi) if shared_scale else np.percentile(vv, [2, 98])
            im = a_.imshow(v, cmap=THERMAL, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=a_, shrink=0.7, label="LST (°C)")
        a_.set_title(os.path.basename(t).replace(f"{cfg.name}_", "")
                     .replace(f"_lst_{cfg.res_fine}m_C.tif", ""), fontsize=10)
        a_.set_xticks([]); a_.set_yticks([])
    for a_ in axes[len(tifs):]:
        a_.axis("off")
    fig.suptitle(f"{cfg.name.title()} - {cfg.mode} scenes"
                 + (" (shared colour scale)" if shared_scale else ""), fontweight="bold")
    fig.tight_layout()
    _save(cfg, fig, f"contact_sheet_{cfg.mode}.png", dpi=130)
    return None


def time_of_day(cfg, table: pd.DataFrame):
    """Boxplot of clean-overpass median LST by local hour (answers MSF: bias by hour?)."""
    if table is None or not len(table):
        print("nothing to plot")
        return None
    hours = sorted(table.local_hour.dropna().unique())
    data = [table.loc[table.local_hour == hr, "median_C"].values for hr in hours]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, positions=hours, widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", lw=1.6))
    for b, hr in zip(bp["boxes"], hours):
        inside = cfg.product_hours is None or cfg.product_hours[0] <= hr <= cfg.product_hours[1]
        b.set_facecolor(DAY_C if inside else "#BBBBBB"); b.set_alpha(0.8)
    for hr, d in zip(hours, data):
        ax.scatter(np.full(len(d), hr), d, s=18, color="k", zorder=3)
    ax.set_xlabel("local hour of overpass")
    ax.set_ylabel("scene median LST (°C)")
    ax.set_title(f"{cfg.name.title()} {cfg.mode}: temperature depends strongly on overpass hour"
                 + ("\n(coloured = used in composites)" if cfg.product_hours else ""))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(cfg, fig, f"time_of_day_{cfg.mode}.png")
    return None


def residual_panel(cfg, res: np.ndarray, by_scene: pd.DataFrame | None = None):
    """Residual histogram on a log scale, Q-Q plot, and per-scene bias."""
    from scipy import stats
    mu, sd = float(res.mean()), float(res.std())
    ncol = 3 if by_scene is not None and len(by_scene) else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5.6 * ncol, 4.4))

    ax[0].hist(res, bins=140, color=BLUE)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].axvline(mu, color="red", ls="--", lw=1.2, label=f"bias {mu:+.2f} °C")
    for k in (2, 3):
        ax[0].axvline(mu + k * sd, color="orange", ls=":", lw=1)
        ax[0].axvline(mu - k * sd, color="orange", ls=":", lw=1)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("residual (°C)"); ax[0].set_ylabel("pixels (log)")
    ax[0].set_title(f"residuals · skew {stats.skew(res):+.2f}", fontweight="bold")
    ax[0].legend(fontsize=8)

    step = max(1, len(res) // 20000)
    osm, osr = stats.probplot(res[::step], dist="norm", fit=False)
    ax[1].scatter(osm, osr, s=2, alpha=0.3, color=NIGHT_C)
    lim = max(abs(osm).max(), abs(osr).max())
    ax[1].plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax[1].set_xlabel("theoretical quantiles"); ax[1].set_ylabel("observed")
    ax[1].set_title("Q–Q plot (tail behaviour)", fontweight="bold")

    if ncol == 3:
        colors = [DAY_C if b > 0 else BLUE for b in by_scene.bias_C]
        ax[2].barh(range(len(by_scene)), by_scene.bias_C, color=colors)
        ax[2].axvline(0, color="k", lw=0.8)
        ax[2].set_yticks(range(len(by_scene)))
        ax[2].set_yticklabels(by_scene.date, fontsize=6)
        ax[2].set_xlabel("bias (°C)")
        ax[2].set_title("per-scene bias", fontweight="bold")
    fig.tight_layout()
    _save(cfg, fig, f"residuals_{cfg.mode}.png")
    return None


def landcover_boxplots(cfg, min_scenes: int = 3, stat: str = "mean_C"):
    """Temperature by land cover across overpasses, both modes when available."""
    from matplotlib.patches import Patch
    frames = {}
    for m in ("Day", "Night"):
        c = cfg if cfg.mode == m else cfg.with_mode(m)
        p = c.path("products", f"zonal_stats_lulc_{c.key}.csv")
        if p.exists():
            d = pd.read_csv(p)
            if len(d):
                frames[m] = d
    if not frames:
        print("no zonal statistics found")
        return None

    common = None
    for d in frames.values():
        keep = d.groupby("lulc").date.nunique()
        s = set(keep[keep >= min_scenes].index)
        common = s if common is None else (common & s)
    ref = "Day" if "Day" in frames else list(frames)[0]
    order = (frames[ref][frames[ref].lulc.isin(common)]
             .groupby("lulc")[stat].median().sort_values(ascending=False).index.tolist())

    modes = list(frames)
    colors = {"Day": DAY_C, "Night": NIGHT_C}
    ncol = 2 if len(modes) == 2 else 1
    fig, ax = plt.subplots(1, ncol, figsize=(8.5 * ncol, 5.4))
    ax = np.atleast_1d(ax)

    width = 0.36 if len(modes) == 2 else 0.6
    for j, m in enumerate(modes):
        d = frames[m]
        data = [d.loc[d.lulc == c, stat].dropna().values for c in order]
        pos = np.arange(len(order)) + (j - (len(modes) - 1) / 2) * (width + 0.04)
        bp = ax[0].boxplot(data, positions=pos, widths=width, patch_artist=True,
                           medianprops=dict(color="black", lw=1.6))
        for b in bp["boxes"]:
            b.set_facecolor(colors[m]); b.set_alpha(0.75)
            b.set_edgecolor("black"); b.set_linewidth(0.7)
        for i, c in enumerate(order):
            v = d.loc[d.lulc == c, stat].dropna().values
            if v.size:
                jitter = np.random.default_rng(i).normal(0, width / 7, v.size)
                ax[0].scatter(np.full(v.size, pos[i]) + jitter, v, s=7,
                              color="k", alpha=0.3, zorder=3, linewidths=0)
    ax[0].set_xticks(np.arange(len(order)))
    ax[0].set_xticklabels(order, rotation=25, ha="right")
    ax[0].set_ylabel("mean LST per overpass (°C)")
    ax[0].set_title(f"Surface temperature by land cover - {cfg.name.title()}",
                    fontweight="bold")
    ax[0].grid(axis="y", alpha=0.25)
    ax[0].legend(handles=[Patch(facecolor=colors[m], alpha=0.75, edgecolor="k", label=m)
                          for m in modes], loc="upper right")

    if len(modes) == 2:
        gap = (frames["Day"].groupby("lulc")[stat].median() -
               frames["Night"].groupby("lulc")[stat].median()).reindex(order)
        bars = ax[1].barh(np.arange(len(order))[::-1], gap.values,
                          color=["#B2182B" if g < gap.median() else BLUE for g in gap.values],
                          edgecolor="k", linewidth=0.6)
        ax[1].set_yticks(np.arange(len(order))[::-1])
        ax[1].set_yticklabels(order)
        ax[1].set_xlabel("day − night median LST (°C)")
        ax[1].set_title("diurnal range by land cover\n(red = retains heat overnight)",
                        fontweight="bold")
        ax[1].grid(axis="x", alpha=0.25)
        for b, g in zip(bars, gap.values):
            ax[1].text(g + 0.06, b.get_y() + b.get_height() / 2, f"{g:.1f}",
                       va="center", fontsize=8)
    fig.tight_layout()
    _save(cfg, fig, "landcover_boxplots.png", dpi=150)
    return None


def day_night_panel(cfg, layers: dict):
    """Day, night, diurnal range and compound risk."""
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    for a_, (key, cmap, label) in zip(ax, [
            ("day", THERMAL, "°C"), ("night", THERMAL, "°C"),
            ("dtr", "magma", "K"), ("compound", "inferno", "flag")]):
        arr = layers[key]
        v = arr[np.isfinite(arr)]
        if key == "compound":
            im = a_.imshow(arr, cmap=cmap, vmin=0, vmax=1)
        else:
            im = a_.imshow(arr, cmap=cmap, vmin=np.percentile(v, 2), vmax=np.percentile(v, 98))
        a_.set_xticks([]); a_.set_yticks([])
        plt.colorbar(im, ax=a_, shrink=0.8, label=label)
    for a_, t in zip(ax, ["mean day LST", "mean night LST", "diurnal range",
                          "compound risk: hot day and hot night"]):
        a_.set_title(t, fontweight="bold")
    fig.suptitle(f"{cfg.name.title()} - day/night heat structure at {cfg.res_fine} m",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(cfg, fig, "daynight_structure.png")
    return None


def transfer_panel(cfg, result):
    """Transfer skill against the home baseline, plus feature overlap."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = ["home\n(source)", "as-is", "no elevation", "retrained\nlocally"]
    values = [result.home[2], result.as_is[2], result.without_dem[2], result.local[2]]
    ax[0].bar(labels, values, color=["#9AA7B4", NIGHT_C, ACCENT, DAY_C],
              edgecolor="k", linewidth=0.5)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].axhline(result.home[2], color="grey", ls="--", lw=1, label="home skill")
    ax[0].set_ylabel("pattern R²")
    ax[0].set_title(f"{result.city.title()} transferability", fontweight="bold")
    ax[0].legend(fontsize=8)

    ov = result.overlap
    ax[1].barh(ov.feature, ov.inside_pct, color=ACCENT, edgecolor="k", linewidth=0.5)
    ax[1].axvline(90, color="red", ls="--", lw=1)
    ax[1].set_xlim(0, 105)
    ax[1].set_xlabel("% of target pixels inside the trained range")
    ax[1].set_title("feature-space overlap", fontweight="bold")
    fig.tight_layout()
    _save(cfg, fig, f"transfer_{result.city}_{cfg.mode}.png")
    return None
