"""Step 07: run the main models and the pre-specified sensitivities, two at a time.

Each run calls 06_model.py with 4 chains; runs whose summary.json already exists are skipped.
Logs: results/logs/<tag>.log

Usage (from the repository root):  python src/07_run_models.py [--lanes 2] [--low-priority]
Completed runs are skipped, so the queue can be stopped and started again; a run that is stopped midway
starts from the beginning (the sampler keeps no checkpoint). --low-priority runs the models at below-normal
CPU priority (Windows), so the computer stays usable.
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src" / "06_model.py"
LOGS = ROOT / "results" / "logs"

SENS = {"inmlcf": ["--source", "inmlcf_occurrence"],      # 1. INMLCF, municipality of occurrence
        "extended": ["--source", "eevv_extended"],        # 2. suicide plus undetermined intent (Y10-Y34)
        "negbin": ["--negbin"],                           # 3. negative binomial likelihood
        "nonspatial": ["--nonspatial"],                   # 4. without spatial component
        "y2018": ["--years", "2018-2024"]}                # 5. 2018-2024
RUNS = {"A_main": ["--covs", "ipm"], "B_main": ["--covs", "B"]}
for k, extra in SENS.items():
    RUNS[f"A_{k}"] = ["--covs", "ipm"] + extra
    RUNS[f"B_{k}"] = ["--covs", "B"] + extra
RUNS["A_nbi"] = ["--covs", "nbi"]                         # 6. NBI instead of IPM
# additional sensitivity analyses
EXTRA_RUNS = {"A_main_ppc": ["--covs", "ipm"],                               # same as A_main, stratified predictive checks
          "B_main_ppc": ["--covs", "B"],
          "A_rho_uniform": ["--covs", "ipm", "--rho-prior", "uniform"],   # Beta(1, 1) prior for rho
          "A_knn6": ["--covs", "ipm", "--adjacency", "knn6"],             # six-nearest-neighbour graph
          "A_ipm_rural": ["--covs", "ipm,rural_share"],                   # IPM and rural share jointly
          "A_negbin_hn": ["--covs", "ipm", "--negbin", "--nb-prior", "halfnormal", "--warmup", "6000"]}
RUNS.update(EXTRA_RUNS)


LOW = False
EXTRA = []


def run(tag):
    if (ROOT / "results" / "model" / tag / "summary.json").exists():
        return tag, "skipped"
    with open(LOGS / f"{tag}.log", "w", encoding="utf-8") as log:
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) if LOW else 0
        r = subprocess.run([sys.executable, str(MODEL), "--tag", tag] + RUNS[tag] + EXTRA, cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, creationflags=flags)
    status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
    print(tag, status, flush=True)
    return tag, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--extra", default="", help='extra arguments for 06_model.py, e.g. "--warmup 2000 --draws 4000"')
    a = ap.parse_args()
    global LOW, EXTRA
    LOW = a.low_priority
    EXTRA = a.extra.split()
    LOGS.mkdir(parents=True, exist_ok=True)
    tags = a.only or list(RUNS)
    with ThreadPoolExecutor(a.lanes) as ex:
        results = list(ex.map(run, tags))
    print("done:", results, flush=True)


if __name__ == "__main__":
    main()
