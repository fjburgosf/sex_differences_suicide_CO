"""Step 04: municipal covariates (CNPV 2018) and collinearity check before any outcome model.

- Model A: municipal multidimensional poverty index (IPM, % of people in poverty).
- Model B: five dimensions chosen by the theoretical framework:
  low educational attainment, informal work, no health insurance, barriers to health services
  (all % of people deprived, IPM 2018) and rural share 2018 (population of centros poblados y rural
  disperso / total, DANE projections).
- Sensitivity 6: unsatisfied basic needs (NBI, % of people), instead of the IPM.
All covariates are standardised (mean 0, SD 1) over the municipalities of the panel, unweighted.
Municipalities merged by the harmonisation rule take the 2018 population-weighted mean.

Rule: if any VIF of model B exceeds 5, the health dimension (no health insurance or barriers to
health services) with the higher VIF is dropped, and the decision is logged. The check uses only the
covariates; no outcome is read here.

Outputs: data/processed/covariates.csv (raw and z-scores), results/covariates/collinearity.json

Usage (from the repository root):  python src/04_covariates.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "covariates"
spec = importlib.util.spec_from_file_location("panel", ROOT / "src" / "03_build_panel.py")
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)

DIMS = {"Bajo logro educativo": "low_education", "Trabajo informal": "informal_work",
        "Sin aseguramiento en salud": "no_health_insurance",
        "Barreras de acceso a servicios de salud": "health_access_barriers"}
MODEL_B = ["low_education", "informal_work", "no_health_insurance", "health_access_barriers", "rural_share"]
HEALTH = ["no_health_insurance", "health_access_barriers"]
NBI_FILL = {"94663": "94343"}


def vif(df):
    X = df.to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    out = {}
    for k, c in enumerate(df.columns):
        y, Z = X[:, k], np.column_stack([np.ones(len(X)), np.delete(X, k, axis=1)])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r2 = 1 - ((y - Z @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = float(1 / (1 - r2))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    f = RAW / "dane_ipm_censal_municipal_2018.xlsx"
    ipm = pd.read_excel(f, sheet_name="2_IPM Mpio y Dpt", dtype={"ID": str}).rename(
        columns={"ID": "muni", "IPM Municipal": "ipm"})[["muni", "ipm"]]
    dep = pd.read_excel(f, sheet_name="6_Privaciones IPM Dpt-Mpio", header=1, dtype={"ID": str})
    dep = dep.rename(columns={"ID": "muni", **DIMS})[["muni"] + list(DIMS.values())]
    nbi = pd.read_excel(RAW / "CNPV-2018-NBI-DIVIPOLA-2021.xlsx", sheet_name="Municipios", header=None, dtype=str)
    nbi = nbi[nbi[2].fillna("").str.fullmatch(r"\d{3}") & nbi[0].fillna("").str.fullmatch(r"\d{2}")]
    nbi = pd.DataFrame({"muni": nbi[0] + nbi[2], "nbi": pd.to_numeric(nbi[4])})
    # The NBI file uses DIVIPOLA 2021, where Mapiripana (94663) is reported within Barrancominas (94343);
    # Mapiripana takes the Barrancominas value (NBI sensitivity only)
    for missing_code, source_code in NBI_FILL.items():
        if missing_code not in set(nbi.muni):
            nbi = pd.concat([nbi, pd.DataFrame({"muni": [missing_code],
                                                "nbi": nbi.loc[nbi.muni == source_code, "nbi"].to_numpy()})])
    rural = pd.read_csv(PROC / "rural_share_2018.csv", dtype={"muni": str})

    cov = ipm.merge(dep, on="muni", how="outer").merge(nbi, on="muni", how="outer")
    cov["muni"] = cov["muni"].str.zfill(5)
    for c in ["ipm", "nbi"] + list(DIMS.values()):
        cov[c] = pd.to_numeric(cov[c], errors="coerce")
    pop18 = pd.read_parquet(PROC / "pop_cells.parquet")
    pop18 = pop18[pop18.year == 2018]
    # population of 2018 by ORIGINAL code is needed for the weighted merge; harmonised codes are used here,
    # so the weight of a merged-away municipality comes from the 2018-2042 series read again
    cov["muni_h"] = panel.harmonise(cov["muni"])
    merged = sorted(set(cov.loc[cov.muni != cov.muni_h, "muni"]))
    if merged:
        new = pd.read_excel(RAW / "PPED-AreaSexoEdadMun-2018-2042_VP.xlsx", sheet_name="PobMunicipalxÁreaSexoEdad",
                            header=8, dtype=str)
        new = new.rename(columns=dict(zip(new.columns[:6], ["DP", "DPNOM", "muni", "name", "year", "area"])))
        new = new[(new.year == "2018") & (new.area.str.strip().str.lower() == "total")]
        tot_col = next(c for c in new.columns if str(c).strip().lower() == "total")
        w = new.set_index(new.muni.str.zfill(5))[tot_col].astype(float)
        cov["w"] = cov["muni"].map(w)
    else:
        cov["w"] = 1.0
    cols = ["ipm", "nbi"] + list(DIMS.values())
    agg = cov.groupby("muni_h").apply(lambda g: pd.Series({c: np.average(g[c], weights=g["w"]) for c in cols}),
                                      include_groups=False).reset_index().rename(columns={"muni_h": "muni"})
    agg = agg.merge(rural, on="muni", how="left")
    munis = sorted(pop18["muni"].unique())
    missing = sorted(set(munis) - set(agg.dropna().muni))
    agg = agg[agg.muni.isin(munis)].sort_values("muni").reset_index(drop=True)
    assert len(agg) == len(munis) and not missing, missing
    for c in cols + ["rural_share"]:
        src = "rural_share_2018" if c == "rural_share" else c
        if c == "rural_share":
            agg["rural_share"] = agg[src]
        agg["z_" + c] = (agg[c] - agg[c].mean()) / agg[c].std(ddof=0)

    corr = agg[["ipm", "nbi"] + MODEL_B].corr().round(3)
    v = vif(agg[MODEL_B])
    dropped = None
    if max(v.values()) > 5:
        dropped = max(HEALTH, key=lambda c: v[c])
    model_b = [c for c in MODEL_B if c != dropped]
    rep = {"n_municipalities": len(agg), "merged_codes": merged, "describe": agg[cols + ["rural_share"]].describe()
           .round(2).to_dict(), "correlation": corr.to_dict(), "vif_model_b": {k: round(x, 3) for k, x in v.items()},
           "vif_rule": "drop the health dimension with the higher VIF if any VIF > 5",
           "dropped": dropped, "model_b_final": model_b,
           "vif_after": {k: round(x, 3) for k, x in vif(agg[model_b]).items()}}
    agg.drop(columns=["rural_share_2018"]).to_csv(PROC / "covariates.csv", index=False)
    (OUT / "collinearity.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_municipalities", "merged_codes", "vif_model_b", "dropped",
                                          "model_b_final", "vif_after")}, ensure_ascii=False, indent=1))
    print(corr.to_string())


if __name__ == "__main__":
    main()
