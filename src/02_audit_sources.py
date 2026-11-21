"""Step 02: raw audit of the outcome sources (no modelling, no covariate-outcome association).

- INMLCF presumed suicides 2015-2024 (municipality of occurrence): counts by year and sex,
  completeness of the variables of interest.
- DANE vital statistics, non-fetal deaths 2015-2024, read in place: suicides = ICD-10 underlying
  cause (C_BAS1) X60-X84 or Y87.0; counts by year and sex, and residence code unknown (municipal
  code ending in 999 or missing).

Output: results/audit/audit_sources.json

Usage (from the repository root):  python src/02_audit_sources.py [EEVV_ROOT]
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "audit"
EEVV_ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "raw" / "eevv"))
EEVV_COLS = ["C_BAS1", "SEXO", "CODPTORE", "CODMUNRE", "COD_DPTO", "COD_MUNIC"]


def unknown_share(s):
    return round(100 * s.str.strip().str.lower().isin(["sin información", "sin informacion", "no sabe / no informa",
                                                       "no había sido implementada", "por determinar"]).mean(), 1)


def inmlcf():
    d = pd.read_csv(RAW / "inmlcf_suicidios_2015_2024.csv", dtype=str)
    tab = pd.crosstab(d["Año del hecho"], d["Sexo de la victima"])
    return {"n": len(d), "by_year_sex": tab.to_dict(orient="index"),
            "n_municipalities_with_cases": int(d["Código Dane Municipio"].nunique()),
            "pct_unknown": {c: unknown_share(d[c]) for c in
                            ["Estado Civil", "Escolaridad", "Pertenencia Étnica", "Zona del Hecho", "Razón del Suicidio",
                             "Identidad de Género", "Orientación Sexual"]},
            "born_venezuela": int((d["País de Nacimiento"] == "Venezuela").sum())}


def eevv():
    rows = {}
    for y in range(2015, 2025):
        f = sorted((EEVV_ROOT / f"dane_eevv_{y}").glob("*/*.csv"))[0]
        d = pd.read_csv(f, usecols=lambda c: c.strip('"') in EEVV_COLS, dtype=str, encoding="latin-1")
        d.columns = [c.strip('"') for c in d.columns]
        cb = d["C_BAS1"].str.strip().str.upper()
        s = d[cb.str.match(r"^(X[67]\d|X8[0-4])") | (cb == "Y870")]
        res_unknown = s["CODPTORE"].isna() | s["CODMUNRE"].fillna("999").str.zfill(3).str.endswith("999")
        rows[y] = {"suicides": int(len(s)), "male": int((s["SEXO"] == "1").sum()), "female": int((s["SEXO"] == "2").sum()),
                   "residence_unknown": int(res_unknown.sum()), "file": str(f)}
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"inmlcf": inmlcf(), "eevv": eevv()}
    rep["eevv_total"] = sum(v["suicides"] for v in rep["eevv"].values())
    (OUT / "audit_sources.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep["inmlcf"].items() if k != "by_year_sex"}, ensure_ascii=False, indent=1))
    print("EEVV total:", rep["eevv_total"])
    for y, v in rep["eevv"].items():
        print(y, v["suicides"], v["male"], v["female"], v["residence_unknown"])


if __name__ == "__main__":
    main()
