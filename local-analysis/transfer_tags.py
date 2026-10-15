"""Carry manual tags over to the new labels of re-segmented series, by position.

    python transfer_tags.py <export_root> "sample 1.1" "sample 1.3" ...            # show the mapping only
    python transfer_tags.py <export_root> "sample 1.1" ... --apply                 # rewrite the tag files

Run it after exporting the re-segmented series and BEFORE importing them into the site: the old cell positions
are read from ../data/<series>/ (the version currently on the site), the new ones from <export_root>.
Each old cell goes to the new cell containing its body centre, else to the new cell with most voxels within MAX_UM
of it. When several old cells now form one new cell, their tag carries over only if they agree; otherwise the new
cell is listed for tagging.
annotations.csv, blob_review.csv and blob_overrides.csv are updated for these series (originals kept as
*.before_<date>.csv); tags of old cells with no new cell are dropped, and new cells without a tag are listed
for tagging (python annotate.py tags only untagged cells).
"""
import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SITE_DATA = HERE.parent / "data"
TAG_FILES = ["annotations.csv", "blob_review.csv", "blob_overrides.csv"]  # annotations first: its conflicts apply to all
LEDGER = HERE / "tag_transfers.csv"  # series and export already transferred: a second --apply would remap new labels
MAX_UM = 6.0


def series_tag(series):
    return series.replace(" ", "_").replace(".", "_")


def mapping(export_root, series):
    tag = series_tag(series)
    old = pd.read_csv(SITE_DATA / tag / "osteocyte_props.csv")
    new = pd.read_csv(export_root / f"tables_series_{tag}" / "osteocyte_props.csv")
    volume = np.load(export_root / f"tables_series_{tag}" / "labels.npz")
    labels, vs = volume["final_labels"], float(volume["voxel_size_um"])
    centre = lambda df: df[[f"body_centroid_{a}_um" for a in "zyx"]].to_numpy()
    reach = int(np.ceil(MAX_UM / vs))
    offsets = np.argwhere(np.ones((2 * reach + 1,) * 3)) - reach
    offsets = offsets[np.linalg.norm(offsets, axis=1) * vs <= MAX_UM]
    rows = []
    for (_, r), c in zip(old.iterrows(), centre(old)):
        z, y, x = np.clip(np.round(c / vs).astype(int), 0, np.array(labels.shape) - 1)
        hit, how = int(labels[z, y, x]), "contains centre"
        if not hit:
            pts = np.array([z, y, x]) + offsets
            pts = pts[np.all((pts >= 0) & (pts < labels.shape), axis=1)]
            near = labels[tuple(pts.T)]
            near = near[near > 0]
            hit, how = (int(np.bincount(near).argmax()), f"within {MAX_UM:g} µm") if len(near) else (0, "none")
        rows.append(dict(series=series, old_label=int(r["label"]), new_label=hit or None, match=how))
    m = pd.DataFrame(rows)
    # a cell that vanished must not hand its tag to a neighbour that another old cell already occupies by its centre
    by_centre = set(m.loc[m["match"] == "contains centre", "new_label"])
    neighbour = (m["match"] != "contains centre") & m["new_label"].isin(by_centre)
    m.loc[neighbour, ["new_label", "match"]] = [None, "gone (neighbour taken)"]
    taken = m.dropna(subset=["new_label"])
    clashes = taken[taken.duplicated("new_label", keep=False)]
    return m, clashes, sorted(set(new["label"]) - set(taken["new_label"].astype(int)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_root", type=Path)
    ap.add_argument("series", nargs="+")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    ledger = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=["series", "export_written", "applied"])
    written = {s: json.loads((args.export_root / f"tables_series_{series_tag(s)}" / "summary.json").read_text())["written"]
               for s in args.series}
    done = [s for s in args.series if ((ledger["series"] == s) & (ledger["export_written"] == written[s])).any()]
    if done:
        print(f"already transferred for these exports, skipped: {done}")
    args.series = [s for s in args.series if s not in done]
    if not args.series:
        return

    maps = {}
    for s in args.series:
        m, clashes, untagged = mapping(args.export_root, s)
        maps[s] = m
        print(f"\n{s}: {m['new_label'].notna().sum()} of {len(m)} old cells matched "
              f"({(m['match'] == 'contains centre').sum()} by centre), {m['new_label'].isna().sum()} without a new cell")
        if len(clashes):
            print(f"  {len(clashes)} old cells share a new cell (merged): "
                  + ", ".join(f"{int(a)}->{int(b)}" for a, b in clashes[["old_label", "new_label"]].values))
        print(f"  new cells without an old cell (to tag): {untagged}")

    conflicted = {s: set() for s in maps}  # new cells whose merged old cells disagree in annotations.csv
    for name in TAG_FILES:
        path = HERE / name
        if not path.exists():
            continue
        tags = pd.read_csv(path)
        keep = tags[~tags["series"].isin(args.series)]
        moved, dropped = [], []
        for s, m in maps.items():
            t = tags[tags["series"] == s].merge(m[["old_label", "new_label"]], left_on="label", right_on="old_label", how="left")
            dropped.append(t[t["new_label"].isna()])
            t = t.dropna(subset=["new_label"]).copy()
            t["label"] = t["new_label"].astype(int)
            # merged cells: the tag carries over only if all old cells agree
            col = "decision" if "decision" in t else "tag"
            agree = t.groupby("label")[col].transform("nunique") == 1
            if name == TAG_FILES[0]:
                conflicted[s] = set(t.loc[~agree, "label"])
            bad = ~agree | t["label"].isin(conflicted[s])
            if bad.any():
                print(f"  {name}, {s}: merged cells with different tags, left for re-tagging: {sorted(set(t.loc[bad, 'label']))}")
            moved.append(t[~bad].drop_duplicates("label").drop(columns=["old_label", "new_label"]))
        moved, dropped = pd.concat(moved), pd.concat(dropped)
        print(f"\n{name}: {len(moved)} tags carried over, {len(dropped)} dropped (old cell gone)"
              + (f": {dropped[['series', 'label']].values.tolist()}" if len(dropped) else ""))
        if args.apply:
            backup = path.with_name(f"{path.stem}.before_{date.today().isoformat()}.csv")
            if not backup.exists():
                shutil.copy2(path, backup)
            pd.concat([keep, moved[tags.columns]]).sort_values(["series", "label"]).to_csv(path, index=False)
            print(f"  written; original kept as {backup.name}")
    if args.apply:
        rows = pd.DataFrame([dict(series=s, export_written=written[s], applied=date.today().isoformat()) for s in maps])
        pd.concat([ledger, rows]).to_csv(LEDGER, index=False)
    else:
        print("\nnothing written (add --apply)")


if __name__ == "__main__":
    main()
