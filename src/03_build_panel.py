"""Step 03: national panel municipality x year x sex x 5-year age band, 2015-2024.

Cases (three numerators, each kept in its own file)
- eevv_residence (MAIN): DANE vital statistics, non-fetal deaths, underlying cause (C_BAS1) X60-X84 or Y87.0,
  assigned to the municipality of RESIDENCE (CODPTORE + CODMUNRE).
- eevv_extended (sensitivity 2): the same plus events of undetermined intent, Y10-Y34.
- inmlcf_occurrence (sensitivity 1): INMLCF presumed suicides, municipality of OCCURRENCE.
Unknown or foreign residence (municipal code ending in 999, missing, or not a municipality of the
projections after harmonisation) is excluded from the municipal panel and reported. Unknown age or
indeterminate sex is reassigned fractionally with the known age-sex distribution of the same
municipality-year, then of the national year, then of the period; municipality-year totals
are not changed.

Population: DANE municipal projections by sex and single-year age, area "Total"; 2015-2017 from the
2005-2017 VP series and 2018-2024 from the 2018-2042 VP series. The rural share of 2018 (centros poblados
y rural disperso / total) is saved for the covariates.

Harmonisation: municipalities created after the 2005-2017 series are merged into the
municipality they were segregated from, for the whole period, so that every unit has the same territory
in all years (see HARMONISE).

Expected counts: national age-sex specific rates of the SAME year, computed from the cases with known
municipality, applied to each municipality (sum of E equals sum of y within each year and sex).

Outputs (data/processed/): pop_cells.parquet, cases_<source>.parquet, panel_<source>.csv
(muni, year, sex, y, E, pop), rural_share_2018.csv, build_checks.json

Usage (from the repository root):  python src/03_build_panel.py [EEVV_ROOT]
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
PROV = ROOT / "data" / "raw" / "PROVENANCE.json"
EEVV_ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "raw" / "eevv"))
YEARS = list(range(2015, 2025))
BANDS = list(range(0, 85, 5))
EEVV_COLS = ["COD_DPTO", "COD_MUNIC", "CODPTORE", "CODMUNRE", "C_BAS1", "SEXO", "GRU_ED1"]

# Municipalities absent from the 2005-2017 series -> municipality they were segregated from.
# Filled after the code check of step 1; every code in the 2018-2042 series must be in the 2005-2017
# series or here.
HARMONISE = {"27493": "27615",  # Nuevo Belén de Bajirá (Chocó, population from 2024) -> Riosucio, Chocó
             "94663": "94343"}  # Mapiripana (Guainía, population 0 from 2020) -> Barrancominas, Guainía


def band(age):
    return int(min(age // 5 * 5, 80))


def eevv_age(code):
    """GRU_ED1 (DANE): 00-08 under 5 years, 09 = 5-9, ..., 24 = 80-84, 25-28 = 85+, 29 unknown."""
    try:
        c = int(code)
    except (TypeError, ValueError):
        return np.nan
    if c <= 8:
        return 0
    if c >= 29:
        return np.nan
    return min(5 * (c - 8), 80)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def harmonise(codes):
    return codes.map(lambda c: HARMONISE.get(c, c))


# ----------------------------------------------------------- population
def sex_age_cols(columns, pattern):
    cols = {}
    for c in columns:
        m = re.match(pattern, str(c))
        if m:
            cols[c] = ("M" if m.group(1) == "Hombres" else "F", band(int(m.group(2))))
    return cols


def load_population():
    cache = OUT / "pop_cells.parquet"
    rural = OUT / "rural_share_2018.csv"
    if cache.exists() and rural.exists():
        return pd.read_parquet(cache)
    old = pd.read_excel(RAW / "DCD-area-sexo-edad-proypoblacion-Mun-2005-2017_VP.xlsx", sheet_name="NuevaMpal",
                        header=11, dtype={"DP": str, "MPIO": str})
    old = old[(old["ÁREA GEOGRÁFICA"].str.strip().str.lower() == "total") & old["AÑO"].isin([2015, 2016, 2017])]
    old = old.rename(columns={"MPIO": "muni", "AÑO": "year"})
    oc = sex_age_cols(old.columns, r"^(Hombres|Mujeres)_(\d+)")

    new = pd.read_excel(RAW / "PPED-AreaSexoEdadMun-2018-2042_VP.xlsx", sheet_name="PobMunicipalxÁreaSexoEdad",
                        header=8, dtype=str)
    new = new.rename(columns=dict(zip(new.columns[:6], ["DP", "DPNOM", "muni", "name", "year", "area"])))
    new["year"] = pd.to_numeric(new["year"], errors="coerce")
    nc = sex_age_cols(new.columns, r"^(Hombres|Mujeres) (\d+) año")
    new2018 = new[new["year"] == 2018].copy()
    tot = new2018[list(nc)].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    new2018["pop_area"] = tot
    new2018["muni"] = harmonise(new2018["muni"].str.zfill(5))
    a = new2018.pivot_table(index="muni", columns=new2018["area"].str.strip().str.lower(), values="pop_area",
                            aggfunc="sum")
    rural_col = next(c for c in a.columns if c.startswith("centros"))
    pd.DataFrame({"muni": a.index, "rural_share_2018": (a[rural_col] / a["total"]).to_numpy()}) \
        .to_csv(rural, index=False)
    new = new[(new["area"].str.strip().str.lower() == "total") & new["year"].isin(YEARS)]

    rows = []
    for frame, cols in ((old, oc), (new, nc)):
        long = frame.melt(id_vars=["muni", "year"], value_vars=list(cols), var_name="col", value_name="pop")
        long["sex"] = long["col"].map(lambda c: cols[c][0])
        long["age"] = long["col"].map(lambda c: cols[c][1])
        long["pop"] = pd.to_numeric(long["pop"], errors="coerce")
        rows.append(long.groupby(["muni", "year", "sex", "age"], as_index=False)["pop"].sum())
    pop = pd.concat(rows, ignore_index=True)
    pop["muni"] = harmonise(pop["muni"].astype(str).str.zfill(5))
    pop["year"] = pop["year"].astype(int)
    pop = pop.groupby(["muni", "year", "sex", "age"], as_index=False)["pop"].sum()
    pop.to_parquet(cache, index=False)
    return pop


# ---------------------------------------------------------------- cases
def load_eevv(extended):
    frames, prov = [], {}
    for y in YEARS:
        f = sorted((EEVV_ROOT / f"dane_eevv_{y}").glob("*/*.csv"))[0]
        d = pd.read_csv(f, usecols=lambda c: c.strip('"') in EEVV_COLS, dtype=str, encoding="latin-1")
        d.columns = [c.strip('"') for c in d.columns]
        cb = d["C_BAS1"].str.strip().str.upper()
        keep = cb.str.match(r"^(X[67]\d|X8[0-4])") | (cb == "Y870")
        if extended:
            keep |= cb.str.match(r"^(Y[12]\d|Y3[0-4])")
        d = d[keep].copy()
        d["year"] = y
        d["age"] = d["GRU_ED1"].map(eevv_age)
        d["sex"] = d["SEXO"].map({"1": "M", "2": "F"})
        dp, mp = d["CODPTORE"].fillna(""), d["CODMUNRE"].fillna("")
        d["muni_raw"] = dp.str.strip().str.zfill(2) + mp.str.strip().str.zfill(3)
        d.loc[(dp.str.strip() == "") | (mp.str.strip() == ""), "muni_raw"] = ""
        frames.append(d[["year", "sex", "age", "muni_raw"]])
        prov[f"EEVV_{y}"] = {"path_read_in_place": str(f), "sha256": sha256(f), "bytes": f.stat().st_size}
    return pd.concat(frames, ignore_index=True), prov


def load_inmlcf():
    d = pd.read_csv(RAW / "inmlcf_suicidios_2015_2024.csv", dtype=str, keep_default_na=False)
    d["year"] = d["Año del hecho"].astype(int)
    lower = d["Grupo de Edad Quinquenal "].str.extract(r"\((\d+)")[0].astype(float)
    d["age"] = lower.map(lambda a: band(a) if pd.notna(a) else np.nan)
    d["sex"] = d["Sexo de la victima"].map({"Hombre": "M", "Mujer": "F"})
    d["muni_raw"] = d["Código Dane Municipio"].str.strip().str.zfill(5)
    return d[["year", "sex", "age", "muni_raw"]]


def reassign(cases):
    """Fractional counts by muni, year, sex, age; unknown sex/age spread proportionally."""
    known = cases.dropna(subset=["sex", "age"])
    base = known.groupby(["muni", "year", "sex", "age"]).size().rename("n").reset_index()
    by_my = {k: g.groupby(["sex", "age"])["n"].sum() for k, g in base.groupby(["muni", "year"])}
    by_y = {k: g.groupby(["sex", "age"])["n"].sum() for k, g in base.groupby("year")}
    period = base.groupby(["sex", "age"])["n"].sum()
    unknown = cases[cases[["sex", "age"]].isna().any(axis=1)]
    extra = []
    for r in unknown.itertuples(index=False):
        for w in (by_my.get((r.muni, r.year)), by_y.get(r.year), period):
            if w is None:
                continue
            if pd.notna(r.sex):
                w = w[w.index.get_level_values("sex") == r.sex]
            if pd.notna(r.age):
                w = w[w.index.get_level_values("age") == r.age]
            if w.sum() > 0:
                break
        w = w / w.sum()
        extra += [{"muni": r.muni, "year": r.year, "sex": s, "age": a, "n": f} for (s, a), f in w.items()]
    out = pd.concat([base, pd.DataFrame(extra)], ignore_index=True)
    return out.groupby(["muni", "year", "sex", "age"], as_index=False)["n"].sum(), len(unknown)


def build(source, cases, pop, munis):
    cases = cases.copy()
    cases["muni"] = harmonise(cases["muni_raw"])
    valid = cases["muni"].isin(munis)
    excluded = cases[~valid]
    cases = cases[valid]
    frac, n_unknown = reassign(cases)
    cell = pop.merge(frac, on=["muni", "year", "sex", "age"], how="left")
    cell["n"] = cell["n"].fillna(0.0)
    rates = cell.groupby(["year", "sex", "age"], as_index=False)[["n", "pop"]].sum()
    rates["r"] = rates["n"] / rates["pop"]
    cell = cell.merge(rates[["year", "sex", "age", "r"]], on=["year", "sex", "age"])
    cell["E"] = cell["r"] * cell["pop"]
    cell = cell.drop(columns="r").rename(columns={"n": "cases"})
    cell.to_parquet(OUT / f"cases_{source}.parquet", index=False)
    panel = cell.groupby(["muni", "year", "sex"], as_index=False)[["cases", "E", "pop"]].sum()
    # integer counts: known-sex records plus the reassigned share of indeterminate sex, rounded
    panel["y"] = panel["cases"].round().astype(int)
    panel[["muni", "year", "sex", "y", "E", "pop", "cases"]].to_csv(OUT / f"panel_{source}.csv", index=False)
    raw_codes = excluded["muni_raw"]
    return {"records": int(len(cases) + len(excluded)), "excluded_residence_unknown_or_invalid": int(len(excluded)),
            "excluded_by_year": excluded.groupby("year").size().to_dict(),
            "excluded_code_examples": raw_codes.value_counts().head(8).to_dict(),
            "in_panel": int(len(cases)), "unknown_age_or_sex_reassigned": int(n_unknown),
            "indeterminate_sex": int(cases["sex"].isna().sum()),
            "in_panel_by_sex_known": cases["sex"].value_counts().to_dict(),
            "y_sum_rounded": int(panel["y"].sum()), "cases_sum_fractional": round(float(panel["cases"].sum()), 4),
            "E_sum": round(float(panel["E"].sum()), 4)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    pop = load_population()
    munis = set(pop["muni"])
    assert pop.groupby("muni").size().nunique() == 1, "unequal cells per municipality"
    checks = {"n_municipalities": len(munis), "harmonise": HARMONISE,
              "missing_pop_cells": int(pop["pop"].isna().sum()),
              "pop_total_by_year": pop.groupby("year")["pop"].sum().round(0).astype(int).to_dict()}
    eevv, prov_e = load_eevv(extended=False)
    checks["eevv_residence"] = build("eevv_residence", eevv, pop, munis)
    ext, _ = load_eevv(extended=True)
    checks["eevv_extended"] = build("eevv_extended", ext, pop, munis)
    checks["inmlcf_occurrence"] = build("inmlcf_occurrence", load_inmlcf(), pop, munis)
    prov = json.loads(PROV.read_text(encoding="utf-8"))
    prov.update(prov_e)
    PROV.write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "build_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1, default=int),
                                           encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=1, default=int))


if __name__ == "__main__":
    main()
