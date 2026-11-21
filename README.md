# Sex differences in municipal socioeconomic deprivation and suicide mortality in Colombia, 2015–2024: analysis code

This repository contains the analysis code used to estimate, by recorded sex, the association between municipal
socioeconomic deprivation and suicide mortality in the 1,121 Colombian municipalities between 2015 and 2024.
Suicide deaths (ICD-10 X60–X84 and Y87.0) by municipality of residence, year, sex and age group are modelled with
Bayesian Poisson models with BYM2 spatial municipal effects. Expected deaths come from national age-specific rates of
the same sex and year. The covariates are the census-based municipal multidimensional poverty index (IPM) of 2018,
four of its household deprivation indicators, the rural population share and the index of unsatisfied basic needs
(NBI). The women/men ratio of relative risks is computed draw by draw.

The repository includes only the analysis scripts. It does not include the source data.

## Data

All sources are public. The microdata are not redistributed here.

| Source | How to obtain it | Location in this repository |
|---|---|---|
| Presumed suicides 2015–2024, National Institute of Legal Medicine and Forensic Sciences | Downloaded by step 01 from datos.gov.co (dataset `f75u-mirk`) | `data/raw/inmlcf_suicidios_2015_2024.csv` |
| Municipal population projections by area, sex and age, 2005–2017 and 2018–2042 (DANE) | Downloaded by step 01 from the DANE website | `data/raw/DCD-area-sexo-edad-proypoblacion-Mun-2005-2017_VP.xlsx`, `data/raw/PPED-AreaSexoEdadMun-2018-2042_VP.xlsx` |
| Census-based municipal multidimensional poverty measure, CNPV 2018 (DANE) | Download the municipal annex from the DANE multidimensional poverty page (census information) and save it with this name; it must contain the sheets `2_IPM Mpio y Dpt` and `6_Privaciones IPM Dpt-Mpio` | `data/raw/dane_ipm_censal_municipal_2018.xlsx` |
| Unsatisfied basic needs by municipality, CNPV 2018 (DANE, DIVIPOLA 2021) | Download from the DANE unsatisfied basic needs page; it must contain the sheet `Municipios` | `data/raw/CNPV-2018-NBI-DIVIPOLA-2021.xlsx` |
| Vital statistics, non-fetal deaths 2015–2024 (DANE) | Download the CSV microdata of each year from the DANE microdata catalogue (ANDA), <https://microdatos.dane.gov.co/index.php/catalog> | `data/raw/eevv/dane_eevv_YYYY/<folder>/<file>.csv`, one folder per year |
| Municipal boundaries, MGN 2018 (DANE Geoportal) | Downloaded by step 05 from the DANE Geoportal service | `data/raw/mgn2018_municipios_colombia.geojson` |

The vital statistics files must contain the variables `COD_DPTO`, `COD_MUNIC`, `CODPTORE`, `CODMUNRE`, `C_BAS1`,
`SEXO` and `GRU_ED1`.

## Installation

Python 3.12 was used.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Running the analysis

Run the scripts from the repository root, in this order:

```bash
python src/01_get_sources.py
python src/02_audit_sources.py
python src/03_build_panel.py
python src/04_covariates.py
python src/05_adjacency.py
python src/05b_adjacency_knn.py
python src/07_run_models.py --lanes 1
python src/08_summarise.py
python src/08b_excluded_residence.py
```

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 01 | `01_get_sources.py` | Downloads the open sources, checks the files provided by the user and records their SHA-256 | `data/raw/`, `data/raw/PROVENANCE.json` |
| 02 | `02_audit_sources.py` | Counts suicides by year and sex in both outcome sources and the completeness of the variables | `results/audit/audit_sources.json` |
| 03 | `03_build_panel.py` | Builds the panel of municipality, year, sex and five-year age group, with populations and expected deaths, for the three outcome definitions | `data/processed/panel_*.csv`, `cases_*.parquet`, `build_checks.json` |
| 04 | `04_covariates.py` | Builds the standardised municipal covariates and checks their collinearity before any outcome model | `data/processed/covariates.csv`, `results/covariates/collinearity.json` |
| 05 | `05_adjacency.py` | Downloads the municipal boundaries and builds the contiguity graph (100 m buffer; islands linked to the nearest municipality) | `data/processed/adjacency.csv`, `municipios_mgn2018.gpkg` |
| 05b | `05b_adjacency_knn.py` | Builds an alternative graph of the six nearest municipalities | `data/processed/adjacency_knn6.csv` |
| 06 | `06_model.py` | Fits one Poisson (or negative binomial) BYM2 model by sex with NumPyro; called by step 07 | `results/model/<run>/summary.json`, `posterior.npz`, `rr_municipal.csv` |
| 07 | `07_run_models.py` | Runs the main models and all sensitivity analyses; completed runs are skipped | `results/model/<run>/`, `results/logs/<run>.log` |
| 08 | `08_summarise.py` | Age-standardised rates by sex and year, and a summary of all model runs | `results/summary/descriptives.json`, `model_runs.csv` |
| 08b | `08b_excluded_residence.py` | Describes the deaths without municipality of residence by year, sex and department | `results/summary/excluded_residence.json` |

## Model runs

`07_run_models.py` runs these analyses (`--only <run> ...` runs a subset):

| Run | Description |
|---|---|
| `A_main`, `B_main` | Model A (IPM) and model B (low educational attainment, informal work, no health insurance, barriers to health services and rural share), deaths by municipality of residence |
| `A_inmlcf`, `B_inmlcf` | Forensic records, assigned by municipality of occurrence |
| `A_extended`, `B_extended` | Adding deaths of undetermined intent (Y10–Y34) |
| `A_negbin`, `B_negbin` | Negative binomial likelihood |
| `A_nonspatial`, `B_nonspatial` | Without spatial component |
| `A_y2018`, `B_y2018` | Period 2018–2024 |
| `A_nbi` | Unsatisfied basic needs instead of the IPM |
| `A_rho_uniform` | Beta(1, 1) prior for the mixing parameter ρ |
| `A_knn6` | Graph of the six nearest municipalities |
| `A_ipm_rural` | IPM and rural share in the same model |
| `A_negbin_hn` | Negative binomial likelihood with a half-normal prior on overdispersion |
| `A_main_ppc`, `B_main_ppc` | Same as the main models (posterior predictive checks by stratum of expected deaths are written by every run) |

Each run uses four chains of 4000 warm-up and 12 000 sampling iterations with a fixed seed (20261118), and reports
R̂, effective sample sizes, divergent transitions and posterior predictive checks in `summary.json`. A main model
took about 20 minutes on a desktop computer; the negative binomial runs take longer. `--low-priority` runs the models
at below-normal CPU priority on Windows.

## Deprivation indicators

In the census-based IPM, deprivation is identified at the household level: a household is deprived in an indicator
when at least one member meets its condition, and all its members are counted as deprived. The municipal indicators
are therefore percentages of persons living in deprived households, not individual coverage measures.
