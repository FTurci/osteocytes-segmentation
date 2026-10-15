"""Shared loading, cleaning, features, geometry and tests for the phenotype notebooks.

Data come from ../data/aggregate/ (refreshed by scripts/import_series.py); manual tags from annotate.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import ConvexHull, Delaunay

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "aggregate"
ANNOTATIONS = HERE / "annotations.csv"
BLOB_REVIEW = HERE / "blob_review.csv"

GROUP_COLOURS = {"elongated": "#2a78d6", "round": "#eb6834"}
INK = dict(primary="#0b0b0b", secondary="#52514e", muted="#898781", context="#c3c2b7", grid="#e1e0d9")


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9,
        "axes.edgecolor": INK["context"], "axes.labelcolor": INK["secondary"], "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": INK["grid"],
        "grid.linewidth": 0.6, "xtick.color": INK["muted"], "ytick.color": INK["muted"], "legend.frameon": False,
    })
    pd.set_option("display.width", 200)


def series_tag(series):
    return series.replace(" ", "_").replace(".", "_")


# ---------------------------------------------------------------- data and tags

def load():
    """Cells and dendrites of every series, with the manual tags:
    tag        phenotype-mode tag (elongated, round, blob/outlier, unsure) or ""
    review     blob-review tag (junk, real but dirty, real (clean), unsure) or ""
    phenotype  elongated / round where the phenotype tag says so, else ""
    junk       True where the cell was judged junk (review junk, or blob/outlier never reviewed)
    """
    cells = pd.read_csv(DATA / "osteocytes.csv")
    dendrites = pd.read_csv(DATA / "dendrites.csv")
    ann = pd.read_csv(ANNOTATIONS)[["series", "label", "tag"]] if ANNOTATIONS.exists() else None
    rev = (pd.read_csv(BLOB_REVIEW)[["series", "label", "tag"]].rename(columns={"tag": "review"})
           if BLOB_REVIEW.exists() else None)
    if ann is not None:
        cells = cells.merge(ann, on=["series", "label"], how="left")
    if rev is not None:
        cells = cells.merge(rev, on=["series", "label"], how="left")
    for c in ("tag", "review"):
        cells[c] = cells[c].fillna("") if c in cells else ""
    cells["phenotype"] = cells["tag"].where(cells["tag"].isin(["elongated", "round"]), "")
    cells["junk"] = (cells["review"] == "junk") | ((cells["tag"] == "blob/outlier") & (cells["review"] == ""))
    cells["tagged"] = cells["tag"] != ""
    return cells, dendrites


def clean(cells, exclude_unsure=False):
    """Remove the cells tagged junk (and unsure ones if asked); untagged cells are kept and counted."""
    unsure = (cells["review"] == "unsure") | ((cells["tag"] == "unsure") & (cells["review"] == ""))
    drop = cells["junk"] | (unsure & exclude_unsure)
    summary = pd.DataFrame({"cells": cells.groupby("series").size(),
                            "junk": cells[cells["junk"]].groupby("series").size(),
                            "untagged": cells[~cells["tagged"]].groupby("series").size(),
                            "kept": cells[~drop].groupby("series").size()}).fillna(0).astype(int)
    return cells[~drop].copy(), summary


# ---------------------------------------------------------------- features

FEATURE_LABELS = {
    "axis_ratio_2_1": "Cell axis ratio 2:1 (high = round)",
    "body_elongation": "Body axis ratio 2:1 (high = round)",
    "body_flatness": "Body axis ratio 3:2 (low = pancake)",
    "body_volume_um3": "Body volume (µm³)",
    "body_axis_1_um": "Body length (µm)",
    "volume_um3": "Cell volume (µm³)",
    "process_fraction": "Share of volume in processes",
    "n_dendrites": "Dendrites per cell",
    "mean_dendrite_reach_um": "Mean dendrite reach (µm)",
    "max_dendrite_reach_um": "Longest dendrite reach (µm)",
    "total_dendrite_path_um": "Total dendrite path (µm)",
    "dendrite_starriness": "Dendrite starriness (0 one axis, 1 star)",
    "dendrite_planarity": "Dendrite planarity (flat star)",
    "dendrite_randomness": "Dendrite randomness (3D star)",
    "dendrite_axis_alignment": "Dendrites along body axis, mean |cos|",
    "Solidity": "Solidity",
    "nucleus_volume_um3": "Nucleus volume (µm³)",
    "radial_position": "Relative radial position (0 axis, 1 hull)",
    "distance_to_axis_um": "Distance from hull axis (µm)",
}


def add_features(cells, dendrites, min_dendrites_for_direction=3, weight_by_reach=False):
    """Body-shape ratios, process fraction and dendrite-direction features (orientation tensor of the base→tip
    directions; linearity t1-t2, planarity 2(t2-t3), randomness 3 t3, starriness = 1 - linearity, alignment = mean
    |cos| with the body axis). Direction features are NaN below min_dendrites_for_direction dendrites."""
    c = cells.copy()
    c["body_elongation"] = c["body_axis_2_um"] / c["body_axis_1_um"]
    c["body_flatness"] = c["body_axis_3_um"] / c["body_axis_2_um"]
    c["process_fraction"] = 1 - c["body_volume_um3"] / c["volume_um3"]
    tips = ["tip_z_um", "tip_y_um", "tip_x_um"]
    bases = ["base_z_um", "base_y_um", "base_x_um"]
    has_base = all(b in dendrites for b in bases)
    index = {k: i for i, k in enumerate(zip(c["series"], c["label"]))}
    centre = c[[f"body_centroid_{a}_um" for a in "zyx"]].to_numpy()
    axis = c[[f"body_axis_{a}" for a in "zyx"]].to_numpy()
    out = {k: np.full(len(c), np.nan) for k in ["linearity", "planarity", "randomness", "starriness", "axis_alignment"]}
    for key, d in dendrites.groupby(["series", "label"]):
        i = index.get(key)
        if i is None or len(d) < min_dendrites_for_direction:
            continue
        u = d[tips].to_numpy() - (d[bases].to_numpy() if has_base else centre[i])
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        w = d["reach_um"].to_numpy() if weight_by_reach else np.ones(len(d))
        t3, t2, t1 = np.linalg.eigvalsh((w[:, None, None] * u[:, :, None] * u[:, None, :]).sum(0) / w.sum())
        out["linearity"][i], out["planarity"][i], out["randomness"][i] = t1 - t2, 2 * (t2 - t3), 3 * t3
        out["starriness"][i] = 1 - (t1 - t2)
        out["axis_alignment"][i] = np.average(np.abs(u @ axis[i]), weights=w)
    for k, v in out.items():
        c[f"dendrite_{k}"] = v
    return c


# ---------------------------------------------------------------- position relative to the hull axis

def hull_axis_positions(points, query, rng, n_samples=20000):
    """Main axis of the filled 2D hull of points, then each query point's relative radial position
    (distance from the axis / distance from the axis to the hull boundary on its side) and distance (µm)."""
    hull = ConvexHull(points)
    samples = uniform_in_hull(points, hull, rng, n_samples)
    centre = samples.mean(0)
    axis = np.linalg.eigh(np.cov((samples - centre).T))[1][:, -1]
    r, d = radial_positions(query, centre, axis, hull)
    return r, d, (centre, axis, hull, points)


def uniform_in_hull(points, hull, rng, n_samples=20000):
    """Points drawn uniformly inside the convex hull (rejection sampling from the bounding box)."""
    tri = Delaunay(points[hull.vertices])
    samples = rng.uniform(points.min(0), points.max(0), size=(n_samples, points.shape[1]))
    return samples[tri.find_simplex(samples) >= 0]


def radial_positions(query, centre, axis, hull):
    """Relative radial position and distance from the axis (see hull_axis_positions)."""
    foot = centre + np.outer((query - centre) @ axis, axis)
    radial = query - foot
    d = np.linalg.norm(radial, axis=1)
    u = radial / np.where(d > 0, d, 1)[:, None]
    normals, offsets = hull.equations[:, :-1], hull.equations[:, -1]
    nu = u @ normals.T
    t = -(foot @ normals.T + offsets) / np.where(nu > 1e-12, nu, np.nan)
    return d / np.nanmin(np.where(t > 0, t, np.nan), axis=1), d


def add_positions(cells, all_cells, dendrites, rng):
    """Radial position of each cell in the image plane, relative to the hull of every cell of its stack
    (centroids and dendrite tips, so the hull encloses the cells' full extent)."""
    c = cells.copy()
    c["radial_position"], c["distance_to_axis_um"] = np.nan, np.nan
    geometry = {}
    for s, g in c.groupby("series"):
        members = all_cells[all_cells["series"] == s]
        tips = dendrites[(dendrites["series"] == s) & dendrites["label"].isin(members["label"])]
        pts = np.vstack([members[["centroid_y_um", "centroid_x_um"]].to_numpy(), tips[["tip_y_um", "tip_x_um"]].to_numpy()])
        if len(members) < 5:
            continue
        r, d, geometry[s] = hull_axis_positions(pts, g[["centroid_y_um", "centroid_x_um"]].to_numpy(), rng)
        c.loc[g.index, "radial_position"], c.loc[g.index, "distance_to_axis_um"] = r, d
    return c, geometry


# ---------------------------------------------------------------- tests

def within_series_permutation(values, is_b, series, rng, n_perm=10000):
    """Difference in means (b - a) and its two-sided p-value, group labels shuffled within each series."""
    values, is_b, series = np.asarray(values, float), np.asarray(is_b, bool), np.asarray(series)
    blocks = [np.flatnonzero(series == s) for s in np.unique(series)]
    diff = lambda b: values[b].mean() - values[~b].mean()
    observed, shuffled, null = diff(is_b), is_b.copy(), np.empty(n_perm)
    for i in range(n_perm):
        for blk in blocks:
            shuffled[blk] = rng.permutation(is_b[blk])
        null[i] = diff(shuffled)
    return observed, (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)


def cliffs_delta(a, b):
    """P(b > a) - P(b < a): -1 to 1, 0 = no difference (rank-based effect size)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = stats.mannwhitneyu(b, a).statistic
    return 2 * u / (len(a) * len(b)) - 1
