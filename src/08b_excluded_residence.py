"""Step 08b: distribution of the suicides excluded for unknown municipality of residence .

Reads the DANE vital statistics with the same case definition as step 03 (X60-X84, Y87.0) and describes the records
without a valid municipality of residence by year, sex and department of occurrence, as counts and as shares of all
suicides of the same group.

Output: results/summary/excluded_residence.json

Usage (from the repository root):  python src/08b_excluded_residence.py
"""
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("panel", ROOT / "src" / "03_build_panel.py")
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


def main():
    frames = []
    for y in panel.YEARS:
        f = sorted((panel.EEVV_ROOT / f"dane_eevv_{y}").glob("*/*.csv"))[0]
        d = pd.read_csv(f, usecols=lambda c: c.strip('"') in panel.EEVV_COLS, dtype=str, encoding="latin-1")
        d.columns = [c.strip('"') for c in d.columns]
        cb = d["C_BAS1"].str.strip().str.upper()
        d = d[cb.str.match(r"^(X[67]\d|X8[0-4])") | (cb == "Y870")].copy()
        dp, mp = d["CODPTORE"].fillna(""), d["CODMUNRE"].fillna("")
        raw = dp.str.strip().str.zfill(2) + mp.str.strip().str.zfill(3)
        raw[(dp.str.strip() == "") | (mp.str.strip() == "")] = ""
        frames.append(pd.DataFrame({"year": y, "sex": d["SEXO"].map({"1": "M", "2": "F"}).fillna("unknown"),
                                    "dept_occurrence": d["COD_DPTO"].str.strip().str.zfill(2), "muni_raw": raw.values}))
    c = pd.concat(frames, ignore_index=True)
    munis = set(pd.read_csv(ROOT / "data" / "processed" / "covariates.csv", dtype={"muni": str})["muni"])
    c["excluded"] = ~panel.harmonise(c["muni_raw"]).isin(munis)

    def tab(by):
        g = c.groupby(by).excluded.agg(["sum", "size"])
        g["share"] = (g["sum"] / g["size"]).round(4)
        return {str(k): {"excluded": int(r["sum"]), "total": int(r["size"]), "share": float(r["share"])} for k, r in g.iterrows()}
    out = {"total": {"excluded": int(c.excluded.sum()), "records": int(len(c)), "share": round(float(c.excluded.mean()), 4)},
           "by_year": tab("year"), "by_sex": tab("sex"), "by_dept_occurrence": tab("dept_occurrence")}
    d = out["by_dept_occurrence"]
    out["top_departments_by_share"] = sorted(((k, v) for k, v in d.items() if v["total"] >= 100),
                                             key=lambda kv: -kv[1]["share"])[:5]
    (ROOT / "results" / "summary" / "excluded_residence.json").write_text(json.dumps(out, indent=1),
                                                                                 encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("total", "by_year", "by_sex", "top_departments_by_share")}, indent=1))


if __name__ == "__main__":
    main()
