"""Tag osteocytes by eye from their rotating renderings.

    python annotate.py [--mode phenotype|blobs] [--annotator NAME] [--port 8765] [--order random|series]
    python annotate.py --amend [--mode phenotype|blobs] [--only TAG]     # review and correct existing tags
    python annotate.py --amend --only untyped     # kept cells still without a round/elongated tag

Modes:
  phenotype  every cell: elongated / round / blob-outlier / unsure            -> annotations.csv
  blobs      only the cells tagged blob/outlier or unsure in annotations.csv:
             junk / real but dirty / real (clean) / unsure                     -> blob_review.csv

Opens a page in the browser showing one cell at a time (rotating ch1 projection and orthogonal MIPs). Keys:
the number or letter shown on each button, V to switch the main view to the rotating dendrite check,
arrows to move, N to edit the note. Cells come in a fixed random order and without series, label or automatic
group, so the tagging is blind; the series and label are shown once a cell is tagged. Every tag is written at once
(series, label, tag, note, annotator, time); restarting resumes at the first untagged cell. When every cell is
tagged, a Finish button stops the app (Ctrl+C in the terminal works too).
With --amend the review is not blind: cells come in series order with their series, label and current tag shown,
tagging a cell moves to the next one, and --only TAG limits it to cells currently tagged TAG. Standard library only.
"""
import argparse
import csv
import json
import random
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIELDS = ["series", "label", "tag", "note", "annotator", "time"]
SEED = 20261014

# per mode: output file, tags as (tag, keys, colour), and which cells to show
MODES = {
    "phenotype": dict(
        out=HERE / "annotations.csv", title="Osteocyte annotation",
        tags=[("elongated", "1e", "#2a78d6"), ("round", "2r", "#eb6834"),
              ("blob/outlier", "3b", "#1baf7a"), ("unsure", "0u", "#6b6a66")]),
    "blobs": dict(
        out=HERE / "blob_review.csv", title="Blob review",
        source=HERE / "annotations.csv", source_tags={"blob/outlier", "unsure"},
        tags=[("junk", "1j", "#6b6a66"), ("real but dirty", "2d", "#eb6834"),
              ("real (clean)", "3c", "#2a78d6"), ("unsure", "0u", "#3d3c3a")]),
}


def series_tag(series):
    return series.replace(" ", "_").replace(".", "_")


def read_tags(path):
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {(r["series"], int(r["label"])): r for r in csv.DictReader(f)}


def load_cells(mode, order, only=None, annotations=None):
    with open(DATA / "aggregate" / "osteocytes.csv", newline="") as f:
        cells = [(r["series"], int(float(r["label"]))) for r in csv.DictReader(f)]
    if "source" in mode:
        source = read_tags(mode["source"])
        cells = [c for c in cells if source.get(c, {}).get("tag") in mode["source_tags"]]
    if only == "untyped":
        # kept cells without a phenotype: tagged blob/outlier or unsure, but not reviewed as junk
        review = read_tags(HERE / "blob_review.csv")
        cells = [c for c in cells if annotations.get(c, {}).get("tag") in ("blob/outlier", "unsure")
                 and review.get(c, {}).get("tag") != "junk"]
    elif only:
        cells = [c for c in cells if annotations.get(c, {}).get("tag") == only]
    if order == "random":
        random.Random(SEED).shuffle(cells)
    return cells


LOCK = threading.Lock()


def save(path, annotations):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for key in sorted(annotations):
            w.writerow(annotations[key])
    tmp.replace(path)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#15171a; --panel:#1f2226; --ink:#eee; --muted:#9a9893; --blue:#2a78d6; }
  * { box-sizing: border-box; }
  html, body { height:100%; }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px system-ui, -apple-system, sans-serif;
         display:flex; flex-direction:column; overflow:hidden; }
  header { display:flex; align-items:center; gap:16px; padding:10px 16px; background:var(--panel); }
  header .progress { flex:1; height:6px; background:#333; border-radius:3px; overflow:hidden; }
  header .progress div { height:100%; background:var(--blue); }
  main { flex:1; min-height:0; display:grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap:16px; padding:16px; }
  @media (max-width: 900px) { body { overflow:auto; } main { grid-template-columns: 1fr; } .view { height:60vh; } }
  .view { background:#000; border-radius:6px; position:relative; min-height:0; overflow:hidden; }
  .view img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; image-rendering:pixelated; }
  .view .caption { position:absolute; top:8px; left:12px; color:var(--muted); font-size:13px; }
  .side { display:flex; flex-direction:column; gap:12px; min-height:0; }
  .side .mipbox { flex:1; min-height:0; background:#000; border-radius:6px; position:relative; }
  .side .mipbox img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
  .done { display:none; align-items:center; gap:16px; padding:10px 16px; background:#1b3a2a; }
  .done.show { display:flex; } .done button { background:#1baf7a; }
  .buttons { display:grid; grid-template-columns:repeat(2, 1fr); gap:8px; }
  button { font:inherit; padding:12px; border-radius:6px; border:2px solid transparent; color:#fff; cursor:pointer; background:#333; }
  button.active { border-color:#fff; box-shadow:0 0 0 2px #fff inset; }
  .nav { display:flex; gap:8px; } .nav button { flex:1; }
  input { font:inherit; width:100%; padding:8px; border-radius:6px; border:1px solid #444; background:#111; color:var(--ink); }
  .meta { color:var(--muted); font-size:13px; min-height:1.2em; }
  kbd { background:rgba(0,0,0,.35); border-radius:3px; padding:0 4px; font-size:12px; }
</style></head><body>
<header><strong id="count"></strong><div class="progress"><div id="bar"></div></div><span class="meta" id="saved"></span></header>
<div class="done" id="done"><span id="donetext"></span><button id="finish">Finish and stop the app</button></div>
<main>
  <div class="view"><span class="caption" id="caption"></span><img id="gif" alt="rotating rendering"></div>
  <div class="side">
    <div class="buttons" id="buttons"></div>
    <input id="note" placeholder="note (optional, N to edit, Enter to keep)">
    <div class="nav"><button id="prev">← previous</button><button id="view">dendrite check <kbd>V</kbd></button><button id="next">next untagged →</button></div>
    <div class="meta" id="who"></div>
    <div class="mipbox"><img id="mip" alt="orthogonal maximum projections"></div>
  </div>
</main>
<script>
const TAGS = __TAGS__;
const AMEND = __AMEND__;
let cells = [], i = 0, dendriteView = false;
const $ = id => document.getElementById(id);
const keymap = {};
for (const [t, keys, colour] of TAGS) {
  const b = document.createElement("button");
  b.dataset.tag = t; b.style.background = colour;
  b.innerHTML = `${t[0].toUpperCase() + t.slice(1)} <kbd>${keys[0]}</kbd>`;
  b.onclick = () => tag(t);
  $("buttons").appendChild(b);
  for (const k of keys) keymap[k] = t;
}
async function load() {
  cells = await (await fetch("/api/cells")).json();
  if (!cells.length) { $("count").textContent = "no cells to review"; return; }
  i = AMEND ? 0 : Math.max(0, cells.findIndex(c => !c.tag));
  show();
}
function show() {
  const c = cells[i];
  $("gif").src = dendriteView ? c.dendrites : c.gif; $("mip").src = c.mip;
  $("caption").textContent = dendriteView ? "dendrite check: grey body, blue dendrites, green flat parts, orange short" : "";
  $("view").innerHTML = (dendriteView ? "ch1 rotation" : "dendrite check") + " <kbd>V</kbd>";
  $("note").value = c.note || "";
  document.querySelectorAll("[data-tag]").forEach(b => b.classList.toggle("active", b.dataset.tag === c.tag));
  const done = cells.filter(c => c.tag).length;
  $("count").textContent = AMEND ? `amend · cell ${i + 1} of ${cells.length} · current tag: ${c.tag || "none"}`
                                 : `cell ${i + 1} of ${cells.length} · ${done} tagged`;
  $("bar").style.width = (100 * done / cells.length) + "%";
  $("who").textContent = (c.tag || AMEND) ? `${c.series} #${c.label}` : "";
  const all = done === cells.length && (!AMEND || i === cells.length - 1);
  $("done").classList.toggle("show", all);
  $("donetext").textContent = !all ? "" : AMEND ? `Last cell reached; all changes are saved.`
                                    : `All ${cells.length} cells tagged and saved. You can still review them with the arrows.`;
  [cells[i + 1], cells[i + 2]].forEach(n => { if (n) { new Image().src = n.gif; } });  // preload
}
async function tag(t) {
  const c = cells[i];
  c.tag = t; c.note = $("note").value;
  const r = await fetch("/api/annotate", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ series: c.series, label: c.label, tag: t, note: c.note }) });
  $("saved").textContent = r.ok ? `saved ${c.series} #${c.label}: ${t}` : "SAVE FAILED";
  if (!r.ok) return;
  const next = AMEND ? -1 : cells.findIndex((x, k) => k > i && !x.tag);
  i = next >= 0 ? next : Math.min(i + 1, cells.length - 1);
  show();
}
$("finish").onclick = async () => {
  await fetch("/api/quit", { method: "POST" });
  document.body.innerHTML = `<div style="margin:auto; text-align:center; font-size:18px; line-height:1.6">` +
    `All ${cells.length} cells tagged and saved.<br>The app has stopped; you can close this tab.</div>`;
};
$("prev").onclick = () => { i = Math.max(0, i - 1); show(); };
$("view").onclick = () => { dendriteView = !dendriteView; show(); };
$("next").onclick = () => { const n = cells.findIndex((x, k) => k > i && !x.tag); i = n >= 0 ? n : Math.min(i + 1, cells.length - 1); show(); };
$("note").addEventListener("keydown", e => { if (e.key === "Enter" || e.key === "Escape") $("note").blur(); });
document.addEventListener("keydown", e => {
  if (document.activeElement === $("note")) return;
  const k = e.key.toLowerCase();
  if (keymap[k]) tag(keymap[k]);
  else if (k === "v") $("view").click();
  else if (e.key === "ArrowLeft") $("prev").click();
  else if (e.key === "ArrowRight") { i = Math.min(i + 1, cells.length - 1); show(); }
  else if (k === "n") { e.preventDefault(); $("note").focus(); }
});
if (AMEND) $("next").textContent = "next →";
load();
</script></body></html>
"""


SERVER = {}


def make_handler(mode, cells, annotations, annotator, amend=False):
    page = (PAGE.replace("__TITLE__", mode["title"] + (" (amend)" if amend else ""))
            .replace("__TAGS__", json.dumps(mode["tags"])).replace("__AMEND__", json.dumps(amend)).encode())
    tags = {t for t, _, _ in mode["tags"]}
    known = set(cells)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self.send(200, page, "text/html; charset=utf-8")
            if self.path == "/api/cells":
                rows = []
                for s, l in cells:
                    a = annotations.get((s, l), {})
                    base = f"/data/{series_tag(s)}/cells/osteocyte_{l}"
                    rows.append(dict(series=s, label=l, tag=a.get("tag", ""), note=a.get("note", ""),
                                     gif=base + "_rotation.gif", mip=base + "_mip.png", dendrites=base + "_dendrites.gif"))
                return self.send(200, json.dumps(rows).encode(), "application/json")
            if self.path.startswith("/data/"):
                f = (DATA / unquote(self.path[len("/data/"):])).resolve()
                if DATA.resolve() in f.parents and f.is_file() and f.suffix in (".gif", ".png"):
                    return self.send(200, f.read_bytes(), "image/gif" if f.suffix == ".gif" else "image/png")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path == "/api/quit":
                self.send(200, b"bye", "text/plain")
                threading.Thread(target=SERVER["server"].shutdown, daemon=True).start()
                return
            if self.path != "/api/annotate":
                return self.send(404, b"not found", "text/plain")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            key = (body["series"], int(body["label"]))
            if key not in known or body["tag"] not in tags:
                return self.send(400, b"unknown cell or tag", "text/plain")
            with LOCK:
                annotations[key] = dict(series=key[0], label=key[1], tag=body["tag"], note=body.get("note", ""),
                                        annotator=annotator, time=datetime.now().isoformat(timespec="seconds"))
                save(mode["out"], annotations)
            self.send(200, b"ok", "text/plain")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=list(MODES), default="phenotype")
    ap.add_argument("--annotator", default="")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--order", choices=["random", "series"], default="random")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--amend", action="store_true", help="review and correct existing tags, in series order, not blind")
    ap.add_argument("--only", help="with --amend: only cells currently carrying this tag, e.g. round or junk; "
                                   "'untyped' = kept cells (not junk) still without a round/elongated tag")
    args = ap.parse_args()

    mode = MODES[args.mode]
    annotations = read_tags(mode["out"])
    order = "series" if args.amend else args.order
    cells = load_cells(mode, order, args.only if args.amend else None, annotations)
    if args.only and not cells:
        raise SystemExit(f"no cells tagged {args.only!r} in {mode['out'].name}; tags: {sorted({t for t, _, _ in mode['tags']})}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(mode, cells, annotations, args.annotator, amend=args.amend))
    SERVER["server"] = server
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{args.mode}: {len(cells)} cells, {sum(k in annotations for k in cells)} already tagged; annotating at {url} "
          f"(writes {mode['out'].name}); Ctrl+C to stop")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    tagged = sum(k in annotations for k in cells)
    print(f"\nstopped: {tagged} of {len(cells)} cells tagged, saved in {mode['out']}")


if __name__ == "__main__":
    main()
