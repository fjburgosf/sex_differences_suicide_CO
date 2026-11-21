"""Step 05b: alternative neighbourhood graph for an additional spatial sensitivity analysis.

Each municipality is joined to its k = 6 nearest municipalities by distance between centroids (EPSG:9377), and the
graph is made symmetric (an edge exists if either municipality is among the six nearest of the other). This graph
does not depend on the 100 m buffer of the main contiguity graph, nor on the links created for islands.

Output: data/processed/adjacency_knn6.csv (1-based, order of covariates.csv), adjacency_knn6_summary.json

Usage (from the repository root):  python src/05b_adjacency_knn.py
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
K = 6


def main():
    order = pd.read_csv(PROC / "covariates.csv", dtype={"muni": str})["muni"].tolist()
    g = gpd.read_file(PROC / "municipios_mgn2018.gpkg")
    col = next(c for c in g.columns if c.lower() in ("muni", "mpio_cdpmp", "muni_h"))
    g[col] = g[col].astype(str).str.zfill(5)
    g = g.set_index(col).loc[order].to_crs(9377)
    P = np.c_[g.geometry.centroid.x, g.geometry.centroid.y]
    _, idx = cKDTree(P).query(P, k=K + 1)
    edges = {(min(i, j), max(i, j)) for i in range(len(P)) for j in idx[i, 1:]}
    I, J = zip(*sorted(edges))
    A = coo_matrix((np.ones(len(I)), (I, J)), shape=(len(P), len(P)))
    ncomp, _ = connected_components(A, directed=False)
    assert ncomp == 1, ncomp
    pd.DataFrame({"node1": np.array(I) + 1, "node2": np.array(J) + 1}).to_csv(PROC / "adjacency_knn6.csv", index=False)
    main_edges = len(pd.read_csv(PROC / "adjacency.csv"))
    s = {"k": K, "n_areas": len(P), "n_edges": len(I), "n_edges_main_graph": main_edges, "components": int(ncomp)}
    (PROC / "adjacency_knn6_summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(s)


if __name__ == "__main__":
    main()
