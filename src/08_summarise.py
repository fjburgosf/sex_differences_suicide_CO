"""Step 08: descriptive statistics by sex and a summary of all model runs.

- Descriptives (main source, EEVV by residence): suicides, crude rate per 100 000 and age-standardised rate
  (direct method, standard = national population of 2018, both sexes) by sex and year; municipalities with no
  suicide in the period, by sex.
- Model runs: for every run in results/model/, relative risk per SD (median, 95% CrI) by sex and the
  women/men ratio of relative risks, with diagnostics.

Outputs: results/summary/descriptives.json, model_runs.csv

Usage (from the repository root):  python src/08_summarise.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
OUT = RES / "summary"
LABEL = {"ipm": "Multidimensional poverty (IPM)", "nbi": "Unsatisfied basic needs (NBI)",
         "low_education": "Low educational attainment", "informal_work": "Informal work",
         "no_health_insurance": "No health insurance", "health_access_barriers": "Barriers to health services",
         "rural_share": "Rural population share"}


def descriptives():
    cell = pd.read_parquet(PROC / "cases_eevv_residence.parquet")
    std = cell[cell.year == 2018].groupby("age")["pop"].sum()
    std = std / std.sum()
    g = cell.groupby(["sex", "year", "age"], as_index=False)[["cases", "pop"]].sum()
    g["r"] = g["cases"] / g["pop"]
    g["w"] = g["age"].map(std)
    asr = (g.assign(x=g.r * g.w).groupby(["sex", "year"])["x"].sum() * 1e5).round(2)
    tot = cell.groupby(["sex", "year"])[["cases", "pop"]].sum()
    crude = (1e5 * tot["cases"] / tot["pop"]).round(2)
    pan = pd.read_csv(PROC / "panel_eevv_residence.csv", dtype={"muni": str})
    zero = pan.groupby(["sex", "muni"])["y"].sum().eq(0).groupby("sex").sum()
    per = cell.groupby(["sex", "age"])[["cases", "pop"]].sum()
    per_asr = ((per["cases"] / per["pop"]).groupby("sex").apply(lambda r: (r.droplevel(0) * std).sum()) * 1e5).round(2)
    return {"cases_by_sex_year": tot["cases"].round(0).astype(int).unstack().to_dict(orient="index"),
            "crude_rate_by_sex_year": crude.unstack().to_dict(orient="index"),
            "asr_by_sex_year": asr.unstack().to_dict(orient="index"),
            "asr_period_by_sex": per_asr.to_dict(),
            "male_female_ratio_asr_period": round(per_asr["M"] / per_asr["F"], 3),
            "cases_total_by_sex": cell.groupby("sex")["cases"].sum().round(0).astype(int).to_dict(),
            "municipalities_with_no_suicide_2015_2024": zero.astype(int).to_dict(),
            "n_municipalities": int(pan["muni"].nunique()),
            "standard": "national population 2018, both sexes, 5-year bands (0-4 ... 80+)"}


def fmt(d, a="rr_median", lo="rr_lo", hi="rr_hi"):
    return f"{d[a]:.2f} ({d[lo]:.2f}–{d[hi]:.2f})"


def model_runs():
    rows = []
    for f in sorted((RES / "model").glob("*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        for cov, r in s["rr_per_sd"].items():
            rows.append({"run": s["tag"], "source": s["source"], "years": f"{s['years'][0]}-{s['years'][-1]}",
                         "likelihood": "negbin" if s["negbin"] else "poisson", "spatial": s["spatial"],
                         "covariate": LABEL.get(cov, cov), "women_rr": fmt(r["F"]), "men_rr": fmt(r["M"]),
                         "ratio_w_m": fmt(r["ratio_F_over_M"], "median", "lo", "hi"),
                         "p_ratio_gt1": r["ratio_F_over_M"]["p_gt1"], "women_p_gt1": r["F"]["p_rr_gt1"],
                         "men_p_gt1": r["M"]["p_rr_gt1"], "max_rhat": round(s["diagnostics"]["max_rhat"], 4),
                         "min_ess": round(s["diagnostics"]["min_ess_bulk"]), "divergences": s["diagnostics"]["divergences"]})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = descriptives()
    (OUT / "descriptives.json").write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
    t = model_runs()
    if t.empty:
        print(json.dumps(d, indent=1, default=float))
        print("no model runs yet")
        return
    t.to_csv(OUT / "model_runs.csv", index=False)
    print(json.dumps({k: d[k] for k in ("asr_period_by_sex", "male_female_ratio_asr_period", "cases_total_by_sex",
                                        "municipalities_with_no_suicide_2015_2024")}, indent=1, default=float))
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
