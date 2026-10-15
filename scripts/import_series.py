"""Copy one series' pipeline export into data/<tag>/ and convert its meshes to GLB for the web viewer.

    python scripts/import_series.py <export_root> "<series name>" [--no-obj]

<export_root> is the folder in which segment_pipeline_interactive.ipynb wrote
tables_series_<tag>/, osteocytes_obj_series_<tag>/, renderings_series_<tag>/ and validation_series_<tag>/.
data/<tag>/ is replaced as a whole, so cells dropped by a new export do not linger.

Writes scene.glb (osteocytes + translucent nuclei; the page toggles the nuclei), one GLB per cell,
the rendered GIF/PNG and the dendrite-validation GIF/PNG per cell, the tables, and the original scene OBJ/MTL files zipped as scene_obj.zip,
then regenerates the site pages (scripts/build_pages.py).
"""
import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from trimesh.exchange.gltf import export_glb

import build_pages

ROOT = Path(__file__).resolve().parent.parent
SCENE_OBJS = ["scene_labels", "scene_labels_nuclei"]
TABLES = ["osteocyte_props.csv", "dendrites.csv", "summary.log", "summary.json"]
# glTF is y-up; the scene OBJs are (x, y, z) in µm with z the thin stack axis, so z becomes up
Z_TO_Y_UP = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])


def series_tag(series):
    return series.replace(" ", "_").replace(".", "_")


def read_scene_obj(path):
    """Objects of a scene OBJ + MTL written by the pipeline: name -> (vertices, faces, rgba)."""
    colours, name = {}, None
    for line in path.with_suffix(".mtl").read_text().splitlines():
        key, *vals = line.split() or [""]
        if key == "newmtl":
            name = vals[0]
            colours[name] = [1.0, 1.0, 1.0, 1.0]
        elif key == "Kd":
            colours[name][:3] = map(float, vals)
        elif key == "d":
            colours[name][3] = float(vals[0])

    objects, verts, faces, current = {}, [], {}, None
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append(line[2:])
        elif line.startswith("f "):
            faces[current].append(line[2:])
        elif line.startswith("o "):
            current = line[2:].strip()
            faces[current] = []
    verts = np.loadtxt(verts, dtype=np.float32).reshape(-1, 3)
    for name, f in faces.items():
        f = np.loadtxt(f, dtype=np.int64).reshape(-1, 3) - 1
        used = np.unique(f)
        objects[name] = (verts[used], np.searchsorted(used, f), colours[name])
    return objects


def to_mesh(name, verts, faces, rgba):
    rgba = (np.asarray(rgba) * 255).round().astype(np.uint8)
    material = trimesh.visual.material.PBRMaterial(
        name=name, baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.6,
        alphaMode="BLEND" if rgba[3] < 255 else "OPAQUE")
    mesh = trimesh.Trimesh(verts, faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    mesh.apply_transform(Z_TO_Y_UP)
    return mesh


def write_glb(path, meshes):
    scene = trimesh.Scene()
    for name, mesh in meshes.items():
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    path.write_bytes(export_glb(scene, include_normals=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_root", type=Path)
    ap.add_argument("series")
    ap.add_argument("--no-obj", action="store_true", help="do not copy the scene OBJ/MTL downloads")
    args = ap.parse_args()

    tag = series_tag(args.series)
    tables = args.export_root / f"tables_series_{tag}"
    objs = args.export_root / f"osteocytes_obj_series_{tag}"
    renders = args.export_root / f"renderings_series_{tag}"
    validation = args.export_root / f"validation_series_{tag}"
    meta = json.loads((tables / "summary.json").read_text())
    assert meta["series"] == args.series, f"{tables} holds {meta['series']!r}, not {args.series!r}"

    out = ROOT / "data" / tag
    if out.exists():
        shutil.rmtree(out)
    (out / "cells").mkdir(parents=True)

    for name in TABLES:
        shutil.copy2(tables / name, out / name)

    # scene_labels_nuclei holds every osteocyte object of scene_labels plus the nuclei
    objects = read_scene_obj(objs / "scene_labels_nuclei.obj")
    meshes = {name: to_mesh(name, *obj) for name, obj in objects.items()}
    write_glb(out / "scene.glb", meshes)
    for name, mesh in meshes.items():
        if name.startswith("osteocyte_"):
            write_glb(out / "cells" / f"{name}.glb", {name: mesh})
    if not args.no_obj:
        with zipfile.ZipFile(out / "scene_obj.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for scene in SCENE_OBJS:
                for ext in (".obj", ".mtl"):
                    z.write(objs / f"{scene}{ext}", f"{tag}_{scene}{ext}")

    # only the cells of this export: the rendering folders keep files of labels from earlier exports
    with open(tables / "osteocyte_props.csv", newline="") as f:
        labels = {int(float(r["label"])) for r in csv.DictReader(f)}
    n_images = 0
    for folder, suffixes in [(renders, ["_rotation.gif", "_mip.png"]), (validation, ["_dendrites.gif", "_dendrites.png"])]:
        for label in labels:
            for suffix in suffixes:
                image = folder / f"osteocyte_{label}{suffix}"
                if image.exists():
                    shutil.copy2(image, out / "cells" / image.name)
                    n_images += 1
    if not validation.is_dir():
        print(f"no {validation.name}/ (exported with SAVE_VALIDATION off, or before it existed): "
              "the cell dialogs will show the mesh only")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{args.series} -> {out.relative_to(ROOT)}: {len(list((out / 'cells').glob('*.glb')))} cell meshes, "
          f"{n_images} images, {size:.1f} MB")
    build_pages.main()


if __name__ == "__main__":
    main()
