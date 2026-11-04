"""Fetch an independent R25 for every galaxy from the PGC catalogue (Paturel et al. 2003, VizieR VII/237).

R25 = 0.5 * D25, with D25 = 0.1 * 10**logD25 arcmin. For each galaxy name the nearest PGC entry
with a valid logD25 within 1 arcmin of the resolved position is used.
"""
import os, glob, re
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import astropy.units as u
from astroquery.vizier import Vizier

HERE = os.path.dirname(os.path.abspath(__file__))
FITS_ROOT = os.environ.get("FITS_ROOT", os.path.join(HERE, "fits"))
SD = os.path.join(FITS_ROOT, "SDSS_FITS")
FB = os.path.join(FITS_ROOT, "Fallback_FITS")
OUT = os.path.join(HERE, "data", "r25_pgc.csv")

def query(gal):
    name = gal.replace("_", " ")
    name = re.sub(r"^MESSIER 0*", "M ", name)
    v = Vizier(columns=["_r", "PGC", "ANames", "logD25"], row_limit=20)
    try:
        r = v.query_object(name, catalog="VII/237", radius=1 * u.arcmin)
    except Exception as e:
        return dict(galaxy=gal, query=name, pgc=None, logD25=None, R25_arcsec=None, sep_arcmin=None, note=f"error: {e}")
    if not r:
        return dict(galaxy=gal, query=name, pgc=None, logD25=None, R25_arcsec=None, sep_arcmin=None, note="no match")
    t = r[0].to_pandas().dropna(subset=["logD25"]).sort_values("_r")
    # Prefer the entry whose alternate names include this galaxy; otherwise the largest
    # object within 0.5 arcmin (the nearest entry can be a small background galaxy).
    key = re.sub(r"[^A-Z0-9]", "", name.upper())
    named = t[t["ANames"].fillna("").str.upper().str.replace(r"[^A-Z0-9 ]", "", regex=True)
              .apply(lambda a: any(re.sub(r"[^A-Z0-9]", "", x) == key for x in a.split("  ") + a.split()))]
    if not named.empty:
        t = named
    else:
        t = t[t["_r"] <= 0.5].sort_values("logD25", ascending=False)
    if t.empty:
        return dict(galaxy=gal, query=name, pgc=None, logD25=None, R25_arcsec=None, sep_arcmin=None, note="no logD25")
    row = t.iloc[0]
    return dict(galaxy=gal, query=name, pgc=int(row.PGC), logD25=float(row.logD25),
                R25_arcsec=0.5 * 0.1 * 10 ** float(row.logD25) * 60, sep_arcmin=float(row._r), note="")

if __name__ == "__main__":
    gals = sorted({os.path.basename(d.rstrip("/")) for root in (SD, FB) for d in glob.glob(f"{root}/*/")})
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(query, gals))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(rows)} galaxies")
