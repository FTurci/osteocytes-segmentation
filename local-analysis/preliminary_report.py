"""Figures and result tables of the website's preliminary-analysis page (../preliminary.qmd).

    ~/Projects/osteo-track/lif-files/.venv/bin/python local-analysis/preliminary_report.py

Reads ../data/aggregate/ and the manual tags (annotations.csv, blob_review.csv); writes ../preliminary/:
figures (PNG) and the markdown fragments the page includes (_*.md). Re-run after re-tagging or re-importing series,
then re-read the page's prose, which states the conclusions in words.
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

import connectivity
from osteo_common import (GROUP_COLOURS, INK, HERE, add_features, add_positions, clean, cliffs_delta, load,
                          radial_positions, series_tag, style, uniform_in_hull, within_series_permutation)

OUT = HERE.parent / "preliminary"
SEED = 20261015
N_PERM = 10000
N_BOOT = 2000
MATCH_TOLERANCE = 0.25  # relative body-volume difference for a size-matched pair
LINK_GAP_UM = 1.5       # two cells are linked when the gap between their masks is at most this
MIN_GROUP_3D = 5        # groups with at least this many cells get a 3D model on the page
DECIMATE = 0.75         # share of mesh triangles removed in the group models (the per-cell GLBs are dense)
# 3 shades per tag, cycled over a group's cells so that neighbours with the same tag stay distinguishable
SHADES = {"elongated": ["#2a78d6", "#7fb2ec", "#16467f"], "round": ["#eb6834", "#f5a77f", "#a33c12"],
          "untyped": ["#898781", "#c3c2b7", "#52514e"]}
GROUPS = ["elongated", "round"]

# feature: (label, unit, log scale)
FEATURES = {
    "body_axis_1_um": ("Body length $a_1$", "µm", False),
    "body_axis_2_um": ("Body width $a_2$", "µm", False),
    "body_axis_3_um": ("Body thickness $a_3$", "µm", False),
    "body_volume_um3": ("Body volume $V_b$", "µm³", True),
    "volume_um3": ("Cell volume $V$", "µm³", True),
    "axis_ratio_2_1": ("Cell axis ratio $a_2/a_1$", "", False),
    "body_elongation": ("Body axis ratio $a_2/a_1$", "", False),
    "body_flatness": ("Body axis ratio $a_3/a_2$", "", False),
    "Solidity": ("Solidity $S$", "", False),
    "n_dendrites": ("Dendrites per cell $N_d$", "", False),
    "mean_dendrite_reach_um": ("Mean dendrite reach", "µm", False),
    "max_dendrite_reach_um": ("Longest dendrite reach", "µm", False),
    "total_dendrite_path_um": ("Total dendrite path", "µm", True),
    "process_fraction": ("Process fraction $f_p$", "", False),
    "dendrite_starriness": ("Dendrite starriness $s$", "", False),
    "nucleus_volume_um3": ("Nucleus volume", "µm³", False),
}
BOXPLOTS = ["body_axis_1_um", "body_axis_2_um", "body_axis_3_um", "body_volume_um3", "axis_ratio_2_1", "n_dendrites"]
FEATURE_SETS = {
    "body volume": ["body_volume_um3"],
    "dendrite count": ["n_dendrites"],
    "body volume + dendrite count": ["body_volume_um3", "n_dendrites"],
    "cell axis ratio": ["axis_ratio_2_1"],
    "body thickness": ["body_axis_3_um"],
    "body volume + dendrite count + cell axis ratio": ["body_volume_um3", "n_dendrites", "axis_ratio_2_1"],
}
LOG_FEATURES = {"body_volume_um3", "volume_um3", "n_dendrites"}


def transform(df, cols):
    return np.column_stack([np.log1p(df[c]) if c in LOG_FEATURES else df[c] for c in cols])


def loso_probabilities(g, cols):
    """Out-of-stack probability of 'round' from a class-balanced logistic regression, one stack held out at a time."""
    p = np.empty(len(g))
    for s in g["series"].unique():
        test = (g["series"] == s).to_numpy()
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
        model.fit(transform(g[~test], cols), g.loc[~test, "is_round"])
        p[test] = model.predict_proba(transform(g[test], cols))[:, 1]
    return p


def bootstrap_auc(y, p, rng):
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    aucs = [roc_auc_score(y[i], p[i]) for i in idx if 0 < y[i].sum() < len(i)]
    return np.percentile(aucs, [2.5, 97.5])


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def num(v):
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(":--" if i == 0 else "--:" for i in range(len(cols))) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def jitter_box(ax, g, f, rng):
    label, unit, log = FEATURES[f]
    data = [g.loc[g["phenotype"] == k, f].dropna() for k in GROUPS]
    ax.boxplot(data, positions=[0, 1], widths=0.55, showfliers=False, medianprops=dict(color=INK["primary"], lw=1.6),
               boxprops=dict(color=INK["secondary"]), whiskerprops=dict(color=INK["secondary"]),
               capprops=dict(color=INK["secondary"]))
    for i, (k, v) in enumerate(zip(GROUPS, data)):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=12, color=GROUP_COLOURS[k], alpha=0.65, lw=0, zorder=3)
    ax.set_xticks([0, 1], [f"{k}\n(n = {len(v)})" for k, v in zip(GROUPS, data)])
    ax.set_title(label, loc="left", pad=15)
    ax.set_ylabel(unit)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[0] in "25" else ""))
    ax.grid(axis="x", visible=False)


def group_model(series, labels, kept, contacts, path):
    """GLB of a group: each cell's mesh from the site (data/<tag>/cells/), decimated and shaded by tag, plus a small
    black sphere at every contact and a grey one at every near-contact."""
    import pyvista as pv
    import trimesh
    from trimesh.exchange.gltf import export_glb
    sys.path.insert(0, str(HERE.parent / "scripts"))
    from import_series import Z_TO_Y_UP

    info = kept.set_index(["series", "label"])
    scene, counts = trimesh.Scene(), {}
    for l in labels:
        m = trimesh.load(HERE.parent / "data" / series_tag(series) / "cells" / f"osteocyte_{l}.glb", force="mesh")
        poly = pv.PolyData.from_regular_faces(m.vertices, m.faces).decimate(DECIMATE)
        tag = info.loc[(series, l), "phenotype"] or "untyped"
        colour = SHADES[tag][counts.get(tag, 0) % 3]
        counts[tag] = counts.get(tag, 0) + 1
        mesh = trimesh.Trimesh(poly.points, poly.regular_faces, process=False)
        rgba = [int(colour[i:i + 2], 16) for i in (1, 3, 5)] + [255]
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.6))
        scene.add_geometry(mesh, geom_name=f"osteocyte_{l}_{tag}", node_name=f"osteocyte_{l}_{tag}")
    for touching, rgba in [(True, [20, 20, 20, 255]), (False, [160, 160, 160, 255])]:
        c = contacts[(contacts["gap_um"] == 0) == touching]
        for i, (x, y, z) in enumerate(c[["at_x_um", "at_y_um", "at_z_um"]].to_numpy()):
            ball = trimesh.creation.icosphere(subdivisions=1, radius=1.2)
            ball.apply_translation([x, y, z])
            ball.apply_transform(Z_TO_Y_UP)
            ball.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.8))
            name = f"{'contact' if touching else 'near'}_{i}"
            scene.add_geometry(ball, geom_name=name, node_name=name)
    path.write_bytes(export_glb(scene, include_normals=True))


def location_section(kept, dendrites, rng):
    """Radial position of round and elongated cells relative to each stack's hull axis, against the positions
    uniformly placed cells would have in the same hull."""
    from scipy import stats

    located, geometry = add_positions(kept, kept, dendrites, rng)
    g = located[(located["phenotype"] != "") & located["radial_position"].notna()].copy()
    g["is_round"] = g["phenotype"] == "round"
    # reference: radial positions of points uniform in each stack's hull; u = F0(r) is uniform under that null
    reference = {s: np.sort(radial_positions(uniform_in_hull(pts, hull, rng, 50000), centre, axis, hull)[0])
                 for s, (centre, axis, hull, pts) in geometry.items()}
    g["u"] = [np.searchsorted(reference[s], r) / len(reference[s]) for s, r in zip(g["series"], g["radial_position"])]

    diff, p = within_series_permutation(g["radial_position"], g["is_round"], g["series"], rng, N_PERM)
    e, r = g.loc[~g["is_round"], "radial_position"], g.loc[g["is_round"], "radial_position"]
    delta = cliffs_delta(e, r)
    rows = []
    for k in GROUPS:
        d = g[g["phenotype"] == k]
        rows.append({"group": k, "cells": len(d), "median radial position": f"{d['radial_position'].median():.2f}",
                     "median, uniform placement": f"{np.median([np.median(reference[s]) for s in d['series']]):.2f}",
                     "share in the inner half of the reference": f"{(d['u'] < 0.5).mean():.0%}",
                     "KS p vs uniform placement": fmt_p(stats.kstest(d["u"], "uniform").pvalue)})
    table = pd.DataFrame(rows)
    (OUT / "_location.md").write_text(
        md_table(table) + "\n: Radial position by tag. *Uniform placement*: cells placed uniformly in the same hulls; "
        "*inner half*: below that reference's median, 50% expected. {#tbl-location}\n\n"
        f"Round minus elongated, mean radial position: {diff:+.3f} (within-stack permutation p = {fmt_p(p)}, "
        f"Cliff's δ = {delta:+.2f}).\n")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    grid = np.linspace(0, 1, 201)
    for k in GROUPS:
        d = g[g["phenotype"] == k]
        v = np.sort(d["radial_position"])
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=GROUP_COLOURS[k], lw=2,
                label=f"{k} (n = {len(v)})")
        w = d["series"].value_counts(normalize=True)
        ref = sum(w[s] * np.searchsorted(reference[s], grid) / len(reference[s]) for s in w.index)
        ax.plot(grid, ref, ":", color=GROUP_COLOURS[k], lw=1.4)
        ax2.hist(d["u"], bins=np.linspace(0, 1, 6), histtype="step", lw=2, color=GROUP_COLOURS[k], density=True)
    ax.plot([], [], ":", color=INK["muted"], label="uniform placement, same stacks")
    ax.set(xlabel="relative radial position (0 = axis, 1 = hull)", ylabel="cumulative share of cells", xlim=(0, 1))
    ax.set_title("Radial position", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    ax2.axhline(1, color=INK["muted"], ls=":", lw=1.4)
    ax2.set(xlabel="position relative to uniform placement, $F_0(r)$", ylabel="density (1 = as uniform)", xlim=(0, 1))
    ax2.set_title("Over- and under-occupancy", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "location.png")
    plt.close(fig)

    n = len(geometry)
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(12, 6.2), squeeze=False)
    for ax, (s, (centre, axis, hull, pts)) in zip(axes.flat, sorted(geometry.items())):
        ax.fill(pts[hull.vertices, 1], pts[hull.vertices, 0], color=INK["grid"], zorder=0)
        span = np.linalg.norm(pts.max(0) - pts.min(0)) / 2
        ends = centre + np.outer([-span, span], axis)
        ax.plot(ends[:, 1], ends[:, 0], color=INK["primary"], lw=1.2, ls="--")
        c = located[located["series"] == s]
        colour = c["phenotype"].map(GROUP_COLOURS).fillna(INK["context"]).where(c["phenotype"] != "", INK["context"])
        ax.scatter(c["centroid_x_um"], c["centroid_y_um"], c=colour, s=18, edgecolor="white", lw=0.5, zorder=3)
        ax.set_title(s, loc="left", fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "location_maps.png")
    plt.close(fig)
    print(f"location: round - elongated {diff:+.3f}, p = {p:.3f}")
    print(table.to_string(index=False))


def connectivity_section(kept, rng):
    """Contacts between cells: table, figure, and a 3D model per large group, for the page's connectivity section."""
    pairs = connectivity.restrict(connectivity.all_pairs(kept["series"].unique()), kept)
    members, G = connectivity.groups(kept, pairs, LINK_GAP_UM)
    members.to_csv(OUT / "connected_groups.csv", index=False)
    links = pairs[pairs["gap_um"] <= LINK_GAP_UM + 1e-9]

    sizes = members.groupby(["series", "group"]).size()
    per_series = pd.DataFrame({
        "cells": members.groupby("series").size(),
        "isolated": (sizes == 1).groupby("series").sum(),
        "in a group": sizes[sizes > 1].groupby("series").sum(),
        "groups": (sizes > 1).groupby("series").sum(),
        "largest group": sizes.groupby("series").max(),
    }).fillna(0).astype(int)
    per_series.loc["total"] = per_series.sum()
    per_series.loc["total", "largest group"] = sizes.max()
    tot = per_series.loc["total"]
    via = links["via"].value_counts()
    touching = links[links["gap_um"] == 0]
    (OUT / "_connectivity.md").write_text(
        md_table(per_series.reset_index().rename(columns={"series": "stack"}))
        + f"\n: Cells linked to at least one other (gap ≤ {LINK_GAP_UM:g} µm), per stack. {{#tbl-connectivity}}\n\n"
        f"With this tolerance **{tot['in a group']} of {tot['cells']}** kept cells ({tot['in a group'] / tot['cells']:.0%}) "
        f"are linked to at least one other, in {tot['groups']} groups; {tot['isolated']} are isolated. "
        f"Of the {len(links)} links, {len(touching)} are direct contacts (median contact area "
        f"{touching['contact_area_um2'].median():.0f} µm²) and {len(links) - len(touching)} are near-contacts. "
        f"**{via.get('process–process', 0)}** meet process to process, {via.get('body–process', 0)} body to process "
        f"and {via.get('body–body', 0)} body to body.\n")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), gridspec_kw=dict(width_ratios=[1.1, 1.1, 0.8]))
    ax = axes[0]
    ax.hist(pairs["gap_um"], bins=np.arange(0, connectivity.MAX_GAP_UM + 0.25, 0.25), color=INK["secondary"])
    ax.axvline(LINK_GAP_UM, color=GROUP_COLOURS["round"], lw=1.5)
    ax.set_xlabel("gap between the masks (µm)")
    ax.set_ylabel("pairs of neighbouring cells")
    ax.set_title("Gaps between neighbouring cells", loc="left")
    ax = axes[1]
    thresholds = np.arange(0, connectivity.MAX_GAP_UM + 0.01, 0.25)
    linked = []
    for t in thresholds:
        comp = [len(c) for c in nx.connected_components(connectivity.link_graph(kept, pairs, t))]
        linked.append(sum(n for n in comp if n > 1) / len(kept))
    ax.plot(thresholds, linked, color=INK["primary"], lw=1.8)
    ax.axvline(LINK_GAP_UM, color=GROUP_COLOURS["round"], lw=1.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("gap tolerance (µm)")
    ax.set_ylabel("share of cells linked to another")
    ax.set_title("Depends little on the tolerance", loc="left")
    ax = axes[2]
    deg = members.merge(kept[["series", "label", "phenotype"]], on=["series", "label"])
    for i, k in enumerate(GROUPS):
        v = deg.loc[deg["phenotype"] == k, "degree"]
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v + rng.uniform(-0.12, 0.12, len(v)), s=12, alpha=0.6, lw=0,
                   color=GROUP_COLOURS[k])
        ax.plot([i - 0.22, i + 0.22], [v.median()] * 2, color=INK["primary"], lw=2)
        ax.text(i, -0.9, f"n={len(v)}", ha="center", fontsize=7, color=INK["muted"])
    ax.set_xticks([0, 1], GROUPS)
    ax.set_yticks(range(int(deg["degree"].max()) + 1))
    ax.set_ylim(-1.3, None)
    ax.set_ylabel("linked neighbours per cell")
    ax.set_title("Neighbours by tag", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "connectivity.png")
    plt.close(fig)

    models = OUT / "groups"
    models.mkdir(exist_ok=True)
    for old in models.glob("*.glb"):
        old.unlink()
    big = (members[members["group_size"] >= MIN_GROUP_3D].groupby(["series", "group"])["label"].apply(list)
           .reset_index().assign(n=lambda d: d["label"].map(len)).sort_values(["n", "series"], ascending=[False, True]))
    options = []
    for _, r in big.iterrows():
        c = links[(links["series"] == r["series"]) & links["a"].isin(r["label"]) & links["b"].isin(r["label"])]
        name = f"{series_tag(r['series'])}_group_{r['group']}.glb"
        group_model(r["series"], r["label"], kept, c, models / name)
        tags = deg.set_index(["series", "label"]).loc[[(r["series"], l) for l in r["label"]], "phenotype"]
        options.append((name, f"{r['series']}, group {r['group']}: {r['n']} cells "
                              f"({(tags == 'elongated').sum()} elongated, {(tags == 'round').sum()} round)"))
    select = "\n".join(f'<option value="preliminary/groups/{n}">{t}</option>' for n, t in options)
    (OUT / "_groups.md").write_text(f"""```{{=html}}
<div class="group-viewer">
<select id="group-select" onchange="document.getElementById('group-model').src = this.value">
{select}
</select>
<model-viewer id="group-model" src="preliminary/groups/{options[0][0]}" camera-controls touch-action="pan-y"
  interaction-prompt="none" shadow-intensity="0" exposure="1.1" class="scene-viewer" camera-orbit="0deg 35deg auto">
</model-viewer>
</div>
```
""")
    size_mb = sum(f.stat().st_size for f in models.glob("*.glb")) / 1e6
    print(f"connectivity: {tot['in a group']}/{tot['cells']} linked, {len(options)} group models ({size_mb:.1f} MB)")


def main():
    style()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    cells, dendrites = load()
    kept, _ = clean(cells)
    kept = add_features(kept, dendrites)
    g = kept[kept["phenotype"] != ""].reset_index(drop=True)
    g["is_round"] = (g["phenotype"] == "round").astype(int)
    y = g["is_round"].to_numpy()

    # ------------------------------------------------------------ counts
    per_series = (g.groupby(["series", "phenotype"]).size().unstack(fill_value=0).reindex(columns=GROUPS, fill_value=0)
                  .join(kept.groupby("series").size().rename("kept")).join(cells.groupby("series").size().rename("segmented")))
    per_series["not typed"] = per_series["kept"] - per_series[GROUPS].sum(axis=1)
    per_series = per_series[["segmented", "kept", "elongated", "round", "not typed"]]
    per_series.loc["total"] = per_series.sum()
    n = dict(segmented=len(cells), kept=len(kept), junk=int(cells["junk"].sum()),
             elongated=int((g["phenotype"] == "elongated").sum()), round=int(y.sum()))
    n["untyped"] = n["kept"] - n["elongated"] - n["round"]
    (OUT / "_counts.md").write_text(
        f"Of the **{n['segmented']}** segmented cells, **{n['junk']}** were judged junk (segmentation artefacts, "
        f"fragments, soft tissue) and removed. Of the remaining **{n['kept']}**, **{n['elongated']}** were tagged "
        f"elongated and **{n['round']}** round; **{n['untyped']}** real but irregular cells could not be given a "
        f"type and are left out of the comparison.\n\n"
        + md_table(per_series.reset_index().rename(columns={"series": "stack"})))

    # ------------------------------------------------------------ group comparison
    rows = []
    for f, (label, unit, _) in FEATURES.items():
        d = g.dropna(subset=[f])
        e, r = d.loc[d["phenotype"] == "elongated", f], d.loc[d["phenotype"] == "round", f]
        _, p = within_series_permutation(d[f], d["is_round"], d["series"], rng, N_PERM)
        rows.append(dict(feature=label + (f" ({unit})" if unit else ""), key=f, elongated=e.median(), round=r.median(),
                         ratio=r.median() / e.median(), delta=cliffs_delta(e, r), p=p))
    tests = pd.DataFrame(rows)
    tests["p_holm"] = multipletests(tests["p"], method="holm")[1]
    tests = tests.sort_values("p").reset_index(drop=True)
    tests.to_csv(OUT / "group_tests.csv", index=False)
    show = pd.DataFrame({
        "feature": tests["feature"],
        "elongated, median": tests["elongated"].map(num),
        "round, median": tests["round"].map(num),
        "round / elongated": tests["ratio"].map(lambda v: f"{v:.2f}"),
        "Cliff's δ": tests["delta"].map(lambda v: f"{v:+.2f}"),
        "p (within stack)": tests["p"].map(fmt_p),
        "p (Holm)": [("**" + fmt_p(p) + "**") if p < 0.05 else fmt_p(p) for p in tests["p_holm"]],
    })
    (OUT / "_tests.md").write_text(md_table(show))

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.4))
    for ax, f in zip(axes.flat, BOXPLOTS):
        jitter_box(ax, g, f, rng)
        p = tests.loc[tests["key"] == f, "p_holm"].item()
        dlt = tests.loc[tests["key"] == f, "delta"].item()
        ax.text(0, 1.02, f"δ = {dlt:+.2f}, p$_{{Holm}}$ {'<' if p < 0.001 else '='} {max(p, 0.001):.3f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=INK["secondary"])
    fig.tight_layout()
    fig.savefig(OUT / "boxplots.png")
    plt.close(fig)

    # ------------------------------------------------------------ do size and dendrites alone give two groups?
    bic_rows = []
    for name, cols in [("body volume + dendrite count", ["body_volume_um3", "n_dendrites"]),
                       ("body volume + dendrite count + cell axis ratio", ["body_volume_um3", "n_dendrites", "axis_ratio_2_1"])]:
        X = StandardScaler().fit_transform(transform(g, cols))
        fits = {k: GaussianMixture(k, covariance_type="full", n_init=20, random_state=SEED).fit(X) for k in (1, 2, 3)}
        bic = {k: m.bic(X) for k, m in fits.items()}
        ari = adjusted_rand_score(y, fits[2].predict(X))
        bic_rows.append({"features": name, **{f"BIC, {k} cluster{'s' if k > 1 else ''}": f"{b:.1f}" for k, b in bic.items()},
                         "preferred": f"{min(bic, key=bic.get)}", "ARI of 2 clusters vs tags": f"{ari:.2f}"})
    gmm = pd.DataFrame(bic_rows)
    gmm.to_csv(OUT / "mixture_bic.csv", index=False)

    auc_rows, rocs = [], {}
    for name, cols in FEATURE_SETS.items():
        p = loso_probabilities(g, cols)
        auc = roc_auc_score(y, p)
        lo, hi = bootstrap_auc(y, p, rng)
        rocs[name] = roc_curve(y, p)[:2]
        auc_rows.append({"features": name, "AUC (leave one stack out)": f"{auc:.2f}", "95% CI": f"{lo:.2f} – {hi:.2f}"})
    aucs = pd.DataFrame(auc_rows)
    aucs.to_csv(OUT / "separation_auc.csv", index=False)
    (OUT / "_separation.md").write_text(md_table(gmm) + "\n: Gaussian mixtures on standardised log body volume, "
                                        "log(1 + dendrite count) and axis ratio; lower BIC is better; ARI 1 = clusters "
                                        "identical to the tags, 0 = chance. {#tbl-mixture}\n\n"
                                        + md_table(aucs) + "\n: Agreement of a logistic regression with the tags, "
                                        "each stack predicted by a model trained on the others; AUC 0.5 = chance, "
                                        "1 = perfect. 95% CI by bootstrap over cells. {#tbl-auc}\n")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for k in GROUPS:
        d = g[g["phenotype"] == k]
        ax.scatter(d["body_volume_um3"], d["n_dendrites"] + rng.uniform(-0.2, 0.2, len(d)), s=16, alpha=0.7, lw=0,
                   color=GROUP_COLOURS[k], label=f"{k} (n = {len(d)})")
    ax.set_xscale("log")
    ax.set_xlabel("Body volume $V_b$ (µm³)")
    ax.set_ylabel("Dendrites per cell $N_d$ (jittered)")
    ax.set_title("Size and dendrite count by tag", loc="left")
    for name, colour, ls in [("body volume + dendrite count", INK["muted"], "--"),
                             ("cell axis ratio", INK["context"], ":"),
                             ("body volume + dendrite count + cell axis ratio", INK["primary"], "-")]:
        fpr, tpr = rocs[name]
        auc = aucs.loc[aucs["features"] == name, "AUC (leave one stack out)"].item()
        ax2.plot(fpr, tpr, ls, color=colour, lw=1.6, label=f"{name} ({auc})")
    ax2.plot([0, 1], [0, 1], color=INK["grid"], lw=1)
    ax2.set_xlabel("elongated cells called round (false-positive rate)")
    ax2.set_ylabel("round cells called round (true-positive rate)")
    ax2.set_title("Recovering the tags, leave one stack out", loc="left")
    ax2.set_aspect("equal")

    ax.legend(loc="upper left")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "size_dendrites.png")
    plt.close(fig)

    # dendrites at equal size: size classes, and each round cell against same-stack elongated cells of similar size
    g["size_class"] = pd.qcut(g["body_volume_um3"], 4)
    classes = g["size_class"].cat.categories
    rows = []
    for c in classes:
        d = g[g["size_class"] == c]
        row = {"body volume (µm³)": f"{c.left:.0f} – {c.right:.0f}"}
        for k in GROUPS:
            v = d.loc[d["phenotype"] == k, "n_dendrites"]
            row[f"{k}: cells"] = len(v)
            row[f"{k}: median dendrites"] = f"{v.median():g}" if len(v) else "–"
        rows.append(row)
    by_size = pd.DataFrame(rows)

    pairs = []
    for _, r in g[g["phenotype"] == "round"].iterrows():
        e = g[(g["phenotype"] == "elongated") & (g["series"] == r["series"])
              & (np.abs(np.log(g["body_volume_um3"] / r["body_volume_um3"])) < np.log(1 + MATCH_TOLERANCE))]
        if len(e):
            pairs.append(dict(round=r["n_dendrites"], elongated=e["n_dendrites"].mean(), n_matches=len(e)))
    pairs = pd.DataFrame(pairs)
    diff = pairs["round"] - pairs["elongated"]
    more, fewer = int((diff > 0).sum()), int((diff < 0).sum())
    (OUT / "_dendrites.md").write_text(
        md_table(by_size) + "\n: Dendrites per cell in four size classes (quartiles of body volume, all typed cells). "
        "{#tbl-size-classes}\n\n"
        f"**Matched comparison.** Each round cell is compared with the elongated cells of the same stack whose body volume "
        f"is within ±{MATCH_TOLERANCE:.0%} of its own. {len(pairs)} of {int(y.sum())} round cells have such matches "
        f"(median {pairs['n_matches'].median():g} each). The round cell has more dendrites than the average of its "
        f"matches in **{more}**, fewer in **{fewer}** and as many in {len(pairs) - more - fewer}; median difference "
        f"{diff.median():+.1f} dendrites.\n")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    for i, c in enumerate(classes):
        for j, k in enumerate(GROUPS):
            v = g.loc[(g["size_class"] == c) & (g["phenotype"] == k), "n_dendrites"]
            x = i + (j - 0.5) * 0.36
            ax.scatter(x + rng.uniform(-0.1, 0.1, len(v)), v + rng.uniform(-0.15, 0.15, len(v)), s=14, alpha=0.7,
                       lw=0, color=GROUP_COLOURS[k], label=k if i == 0 else None)
            if len(v):
                ax.plot([x - 0.14, x + 0.14], [v.median()] * 2, color=INK["primary"], lw=2)
            ax.text(x, -1.6, f"n={len(v)}", ha="center", fontsize=7, color=INK["muted"])
    ax.set_xticks(range(len(classes)), [f"{c.left:.0f}–{c.right:.0f}" for c in classes])
    ax.set_xlabel("Body volume class (µm³)")
    ax.set_ylabel("Dendrites per cell (jittered)")
    ax.set_ylim(-2.4, None)
    ax.set_title("Dendrites by size class (bars: medians)", loc="left")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)

    top = max(pairs["round"].max(), pairs["elongated"].max()) + 1
    ax2.plot([0, top], [0, top], color=INK["context"], lw=1)
    ax2.scatter(pairs["elongated"], pairs["round"] + rng.uniform(-0.15, 0.15, len(pairs)), s=18, alpha=0.75, lw=0,
                color=GROUP_COLOURS["round"])
    ax2.text(0.04, 0.96, f"above the line: {more}\nbelow: {fewer}", transform=ax2.transAxes, va="top", fontsize=8,
             color=INK["secondary"])
    ax2.set_xlim(0, top)
    ax2.set_ylim(0, top)
    ax2.set_aspect("equal")
    ax2.set_xlabel(f"mean dendrites of same-stack elongated cells\nwith body volume within ±{MATCH_TOLERANCE:.0%}")
    ax2.set_ylabel("dendrites of the round cell")
    ax2.set_title("Each round cell vs its size-matched cells", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "dendrites_by_size.png")
    plt.close(fig)

    location_section(kept, dendrites, rng)
    connectivity_section(kept, rng)

    sig = tests[tests["p_holm"] < 0.05]
    print(f"{len(g)} typed cells; significant after Holm: {', '.join(sig['key'])}")
    print(gmm.to_string(index=False))
    print(aucs.to_string(index=False))
    print(by_size.to_string(index=False))
    print(f"matched round cells: {len(pairs)}, more {more}, fewer {fewer}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
