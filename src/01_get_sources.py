"""Step 01: gather the raw sources (unaltered) and record their provenance.

Downloaded by this script:
- presumed suicides 2015-2024 of the National Institute of Legal Medicine and Forensic Sciences (datos.gov.co,
  dataset f75u-mirk), national CSV export;
- DANE municipal population projections by area, sex and single-year age, 2005-2017 and 2018-2042.

Downloaded by the user (see README) and placed in data/raw/:
- dane_ipm_censal_municipal_2018.xlsx   DANE census-based municipal multidimensional poverty measure, CNPV 2018
                                        (sheets "2_IPM Mpio y Dpt" and "6_Privaciones IPM Dpt-Mpio");
- CNPV-2018-NBI-DIVIPOLA-2021.xlsx      DANE unsatisfied basic needs by municipality, CNPV 2018 (sheet "Municipios");
- eevv/dane_eevv_YYYY/<folder>/<file>.csv   DANE vital statistics microdata, non-fetal deaths, one folder per year
                                        2015-2024 (read in place by steps 02-03 and hashed there).

Files in data/raw/ are never edited. Origin, SHA-256, size and UTC time go to data/raw/PROVENANCE.json; re-running
skips existing files and only re-hashes them.

Usage (from the repository root):  python src/01_get_sources.py
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROV = RAW / "PROVENANCE.json"
PROJ = "https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Municipal/"
DOWNLOADS = {
    "inmlcf_suicidios_2015_2024.csv": "https://www.datos.gov.co/api/views/f75u-mirk/rows.csv?accessType=DOWNLOAD",
    "DCD-area-sexo-edad-proypoblacion-Mun-2005-2017_VP.xlsx": PROJ + "DCD-area-sexo-edad-proypoblacion-Mun-2005-2017_VP.xlsx",
    "PPED-AreaSexoEdadMun-2018-2042_VP.xlsx": PROJ + "PPED-AreaSexoEdadMun-2018-2042_VP.xlsx",
}
MANUAL = ["dane_ipm_censal_municipal_2018.xlsx", "CNPV-2018-NBI-DIVIPOLA-2021.xlsx"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    prov = json.loads(PROV.read_text(encoding="utf-8")) if PROV.exists() else {}
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    for name, url in DOWNLOADS.items():
        dst = RAW / name
        if not dst.exists():
            with s.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with dst.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            prov[name] = {"url": url, "downloaded_utc": datetime.now(timezone.utc).isoformat()}
        prov.setdefault(name, {"url": url}).update({"bytes": dst.stat().st_size, "sha256": sha256(dst)})
        print(f"{name:58s} {prov[name]['sha256'][:16]}")
    missing = [n for n in MANUAL if not (RAW / n).exists()]
    for name in MANUAL:
        if (RAW / name).exists():
            prov[name] = {"provided_by_user": True, "bytes": (RAW / name).stat().st_size, "sha256": sha256(RAW / name)}
            print(f"{name:58s} {prov[name]['sha256'][:16]}")
    years = [y for y in range(2015, 2025) if not list((RAW / "eevv" / f"dane_eevv_{y}").glob("*/*.csv"))]
    PROV.write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    if missing or years:
        raise SystemExit(f"Missing files (see README): {missing} {['eevv/dane_eevv_%d' % y for y in years]}")


if __name__ == "__main__":
    main()
