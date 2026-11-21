"""Step 05: national municipal polygons (MGN 2018) and contiguity graph for the BYM2 model.

Source: DANE Geoportal, MGN 2018 integrated with CNPV 2018, layer "Municipios"
https://geoportal.dane.gov.co/mparcgis/rest/services/MGN2018/Serv_CapaMunicipiosInt_2018/MapServer/0
Queried one department at a time (the all-country query is too large), with server-side generalisation
(maxAllowableOffset, degrees); if a department fails, smaller offsets are tried.

Contiguity: polygons that intersect after a 100 m buffer in a metric projection (EPSG:9377, MAGNA-SIRGAS Origen
Nacional), so that municipalities whose generalised boundaries no longer touch exactly are still neighbours.
Units that the harmonisation rule merged are dissolved first.
Islands and disconnected components (San Andrés and Providencia, at least): each component other than the
largest is linked to the nearest municipality of the rest of the graph (centroid distance), until the graph
is connected. This keeps a single ICAR component and one BYM2 scaling factor.

Outputs: data/raw/mgn2018_municipios_colombia.geojson (as served), data/processed/
municipios_mgn2018.gpkg, adjacency.csv (1-based, order of covariates.csv), adjacency_summary.json

Usage (from the repository root):  python src/05_adjacency.py
"""
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from libpysal.weights import W, fuzzy_contiguity

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROV = ROOT / "data" / "raw" / "PROVENANCE.json"
LAYER = "https://geoportal.dane.gov.co/mparcgis/rest/services/MGN2018/Serv_CapaMunicipiosInt_2018/MapServer/0"
GEO = RAW / "mgn2018_municipios_colombia.geojson"
OFFSETS = [0.0005, 0.0002, 0.001, 0.0001]
spec = importlib.util.spec_from_file_location("panel", ROOT / "src" / "03_build_panel.py")
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


def download(depts):
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    meta = s.get(LAYER, params={"f": "json"}, timeout=120).json()
    fields = [f["name"] for f in meta["fields"]]
    dept_field = next(f for f in fields if f.upper() in ("DPTO_CCDGO", "DPTO"))
    code_field = next(f for f in fields if f.upper() in ("MPIO_CDPMP", "MPIO_CCNCT"))
    feats, log = [], {}
    for d in depts:
        for off in OFFSETS:
            try:
                r = s.get(LAYER + "/query", params={"where": f"{dept_field}='{d}'", "outFields": f"{dept_field},{code_field}",
                                                    "outSR": 4326, "maxAllowableOffset": off, "f": "geojson"}, timeout=300)
                r.raise_for_status()
                js = r.json()
                fs = [f for f in js.get("features", []) if f.get("geometry")]
                if fs and len(fs) == len(js["features"]):
                    feats += fs
                    log[d] = {"offset": off, "n": len(fs)}
                    break
            except (requests.RequestException, ValueError) as e:
                log.setdefault(d + "_errors", []).append(f"{off}: {type(e).__name__}")
        else:
            raise SystemExit(f"department {d} could not be downloaded: {log}")
        print(d, log[d], flush=True)
    GEO.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    prov = json.loads(PROV.read_text(encoding="utf-8"))
    prov[GEO.name] = {"url": LAYER + "/query (one request per department, outSR=4326, f=geojson)",
                      "downloaded_utc": datetime.now(timezone.utc).isoformat(), "bytes": GEO.stat().st_size,
                      "sha256": hashlib.sha256(GEO.read_bytes()).hexdigest(), "code_field": code_field,
                      "per_department": log}
    PROV.write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    order = pd.read_csv(PROC / "covariates.csv", dtype={"muni": str})["muni"].tolist()
    if not GEO.exists():
        download(sorted({m[:2] for m in order} | {m[:2] for m in panel.HARMONISE}))
    g = gpd.read_file(GEO)
    code_field = next(c for c in g.columns if c.upper() in ("MPIO_CDPMP", "MPIO_CCNCT"))
    g["muni"] = panel.harmonise(g[code_field].astype(str).str.zfill(5))
    g = g[["muni", "geometry"]].to_crs(9377)
    g["geometry"] = g.buffer(0)
    g = g.dissolve(by="muni").reset_index()
    diff = set(order) ^ set(g["muni"])
    assert not diff, sorted(diff)
    g = g.set_index("muni").loc[order].reset_index()
    # intersection-based contiguity: polygons that touch or overlap after a 100 m buffer are neighbours
    # (vertex-based queen fails with generalised boundaries, which no longer share vertices)
    w = fuzzy_contiguity(g, buffering=True, buffer=100, predicate="intersects")
    nb = {i: set(v) for i, v in w.neighbors.items()}

    # connect components (islands first)
    cent = np.column_stack([g.centroid.x, g.centroid.y])
    links = []
    while True:
        wc = W({i: sorted(v) for i, v in nb.items()}, silence_warnings=True)
        labels = wc.component_labels
        if wc.n_components == 1:
            break
        sizes = np.bincount(labels)
        small = np.argmin(sizes)
        inside = np.where(labels == small)[0]
        outside = np.where(labels != small)[0]
        dmat = np.linalg.norm(cent[inside][:, None] - cent[outside][None], axis=2)
        a, b = np.unravel_index(dmat.argmin(), dmat.shape)
        i, j = int(inside[a]), int(outside[b])
        nb[i].add(j)
        nb[j].add(i)
        links.append({"from": order[i], "to": order[j], "km": round(float(dmat[a, b]) / 1000, 1)})
    edges = sorted({(i + 1, j + 1) for i, v in nb.items() for j in v if i < j})
    pd.DataFrame(edges, columns=["node1", "node2"]).to_csv(PROC / "adjacency.csv", index=False)
    g.to_crs(4326).to_file(PROC / "municipios_mgn2018.gpkg", driver="GPKG")
    card = [len(v) for v in nb.values()]
    summary = {"n_areas": len(g), "n_edges": len(edges), "islands_before_linking": w.islands and
               [order[i] for i in w.islands], "components_before_linking": int(w.n_components),
               "links_added": links, "neighbors_min_mean_max": [min(card), round(float(np.mean(card)), 2), max(card)],
               "buffer_m": 100, "crs_contiguity": "EPSG:9377"}
    (PROC / "adjacency_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
