# Keeping the site up to date

Live site: <https://FTurci.github.io/osteocytes-segmentation/>
(published from the `gh-pages` branch; sources on `main`)

Paths below are the ones on this machine:

- pipeline notebook: `~/Projects/osteo-track/lif-files/New_data_sept2026/segment_pipeline_interactive.ipynb`
- Python with trimesh: `~/Projects/osteo-track/lif-files/.venv/bin/python`
- this repo: `~/Projects/osteocytes-segmentation`

## Add or reprocess a series

1. **Segment and export** in the notebook.
   - If the notebook file changed on disk, reload it in the editor without saving and restart the kernel
     (Run All); the kernel runs the editor's copy, not the file.
   - Pick the series, tune the parameters, keep `SAVE_RENDERINGS` (gallery GIFs) and
     `SAVE_VALIDATION` (dendrite-check GIFs in the cell dialogs) ticked, click **Export**.
   - Export writes `tables_series_<tag>/` (incl. `summary.json`), `osteocytes_obj_series_<tag>/`
     and `renderings_series_<tag>/` next to the notebook.

2. **Import into the site** (replaces `data/<tag>/`, regenerates `series/`, `_series_index.md`
   and `data/aggregate/` for the analysis page):

   ```bash
   ~/Projects/osteo-track/lif-files/.venv/bin/python scripts/import_series.py \
       ~/Projects/osteo-track/lif-files/New_data_sept2026 "sample 4.1"
   ```

   The series name is exactly as in the LIF file (`"sample 1.3"`, `"smple 1.2"`, ...).
   Error `summary.json not found` = the export came from a stale kernel: reload, restart, export again.

3. **Check locally**: `quarto preview` (re-renders on change). Do not run `quarto render`
   while a preview is running; the two clash on `_site/`.

4. **Commit and publish**:

   ```bash
   git add data series _series_index.md
   git commit -m "Reprocess sample 4.1: <what changed>"
   git push
   quarto publish gh-pages
   ```

   `quarto publish gh-pages` renders the site and pushes `_site/` to the `gh-pages` branch;
   GitHub Pages updates within a minute or two. Hard-reload (Cmd+Shift+R) if the browser shows old CSVs.

## Reprocess in batch (no napari)

After a change to the pipeline code, rerun every published series with the parameters it was published with
(read from `data/<tag>/summary.json`), export, and import into the site in one go:

```bash
cd ~/Projects/osteo-track/lif-files/New_data_sept2026
../.venv/bin/python segment_batch.py --from-site ~/Projects/osteocytes-segmentation \
    --import-to ~/Projects/osteocytes-segmentation
```

- `--dry-run` first: prints each series' parameters (non-default ones, and options that are new since the export
  and take the notebook defaults) without running.
- One series: `--params data/<tag>/summary.json`, or `--series "sample 1.3"` for the notebook defaults.
- Override for the whole run: `--set KEY=VALUE` (repeatable), e.g. `--set SAVE_RENDERINGS=false`.
- `--jobs 4` segments 4 series at once in separate processes (each logs to `batch_<series>.log`); imports
  still run one after another at the end. 4–5 suits this MacBook (5 performance cores, 18 GB; ~1–2 GB per job).
- Roughly 25–60 s per series. Then `quarto preview`, commit, publish as in step 4 above.

The script executes the notebook's own cells, so it always runs the current pipeline code; there is nothing to
keep in sync.

## Other changes

- **Edited `analysis.qmd`, `index.qmd`, `assets/` or `_quarto.yml`**: commit, push, `quarto publish gh-pages`.
- **Edited `scripts/build_pages.py`**: run `python3 scripts/build_pages.py` (standard library only)
  to regenerate the pages, then commit and publish.
- **Remove a series**: delete `data/<tag>/`, run `python3 scripts/build_pages.py`, commit, publish.

## Statistical analysis (local)

`local-analysis/statistics.ipynb` reads `data/aggregate/` (refreshed by every import), so after importing
series just re-run it top to bottom (Restart & Run All). Tables go to `local-analysis/results/`, figures to
`local-analysis/figures/`. It runs in the pipeline venv (`~/Projects/osteo-track/lif-files/.venv`) or any
environment with `local-analysis/requirements.txt`. The series -> sample grouping is `SAMPLE_OF` in its
data cell. It is not part of the website.

## Preliminary-analysis page

`preliminary.qmd` (on the site) includes figures and tables from `preliminary/`, written by
`local-analysis/preliminary_report.py` from `data/aggregate/` and the manual tags. After re-tagging or re-importing:

```bash
~/Projects/osteo-track/lif-files/.venv/bin/python local-analysis/preliminary_report.py
```

then re-read the page's conclusions (prose, not generated), commit `preliminary/` and publish.
The connectivity section reads the label volumes (`labels.npz`) of the exports next to the pipeline notebook and caches
the cell pairs in `local-analysis/results/cell_contacts.csv`: delete that file after re-segmenting a series.

## Manual annotation

`local-analysis/annotate.py` (standard library; opens in the browser, tags saved as you go):

- `python3 annotate.py --annotator FT`: every cell as elongated / round / blob-outlier / unsure → `annotations.csv`
- `python3 annotate.py --mode blobs --annotator FT`: only the cells tagged blob/outlier or unsure, as
  junk / real but dirty / real (clean) / unsure → `blob_review.csv`

- add `--amend` to either command to review and correct existing tags (series order, not blind, current tag
  shown); `--only round` (or any tag) limits it to the cells currently carrying that tag

Keys are shown on the buttons; V switches to the rotating dendrite check. When every cell is tagged, a
Finish button stops the app (or Ctrl+C in its terminal). After re-segmenting series, carry tags over with
`transfer_tags.py` before importing them (see its docstring). Labels change if a series is
re-segmented with different parameters, which invalidates tags for that series.

## Voxel data

Each export writes `cells_series_<tag>/osteocyte_<label>.npz` next to the notebook (raw ch1 and ch0 crops, mask,
nucleus, neighbouring labels, soft tissue, offset, voxel size) and `tables_series_<tag>/labels.npz` (whole label
volume). These stay local; they are not imported into the website.

## Things to remember

- Parameters used for each series are in `data/<tag>/summary.json`; the notebook resets to its defaults
  on restart.
- Every re-import adds new binaries (GLB, GIF, zip) to git history; if the repo grows too large,
  move `data/**/*.{glb,gif,zip}` to Git LFS.
- GitHub Pages limits: 1 GB site, 100 MB per file (largest file now ~27 MB).
- Series001 currently fails in the pipeline ("No points given") and is not on the site.
