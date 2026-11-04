"""Re-download SkyView cutouts wide enough for the 2.8 R25 working grid (field = 3.0 R25, PGC R25).

The original 500-px SDSS cutouts (198 arcsec) were too small for 31 of 93 SDSS galaxies.
Writes $FITS_ROOT/wide_FITS/<galaxy>/<band>.fits; existing files are skipped.
Bands: SDSS g, i for SDSS galaxies; DSS2 Blue, WISE 3.4 for the others (max 1000 px).
"""
import os, sys, time, warnings
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import urllib.parse, urllib.request
from astropy.io import fits
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.environ.get("FITS_ROOT", os.path.join(HERE, "fits")), "wide_FITS")
NATIVE = {"SDSSg": 0.396, "SDSSi": 0.396, "SDSSu": 0.396, "DSS2 Blue": 1.0, "WISE 3.4": 1.6,  # SkyView returns no data for WISE at <= ~1.5"/px
          "GALEX Near UV": 1.5, "GALEX Far UV": 1.5}
r25 = pd.read_csv(f"{HERE}/data/r25_pgc.csv").set_index("galaxy").R25_arcsec
split = pd.read_csv(f"{HERE}/split.csv")

def fetch(task):
    gal, survey = task
    path = f"{OUT}/{gal}/{survey.replace(' ', '_')}.fits"
    if os.path.exists(path): return gal, survey, "skip"
    width = 3.0 * float(r25[gal])
    pix = int(min(max(width / NATIVE[survey], 200), 1000))
    for attempt in range(5):
        try:
            # direct SkyView query (astroquery's wrapper hangs on this machine)
            q = urllib.parse.urlencode(dict(Position=gal.replace("_", " "), Survey=survey, Pixels=pix,
                                            Size=width / 3600.0, Return="FITS"))
            data = urllib.request.urlopen(f"https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?{q}", timeout=300).read()
            if not data.startswith(b"SIMPLE"): return gal, survey, "none"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".part"; open(tmp, "wb").write(data)
            if fits.getheader(tmp).get("NAXIS", 0) < 2:          # SkyView "no data" reply: often transient
                os.remove(tmp); err = "empty image"; time.sleep(10); continue
            os.replace(tmp, path); return gal, survey, "ok"
        except Exception as e:
            err = str(e)[:80]; time.sleep(5)
    return gal, survey, f"fail: {err}"

tasks = []
for r in split.itertuples():
    bands = ["SDSSg", "SDSSi"] if r.batch == "sdss" else ["DSS2 Blue", "WISE 3.4"]
    tasks += [(r.galaxy, b) for b in bands]
print(f"{len(tasks)} downloads", flush=True)
n = 0; bad = []
with ThreadPoolExecutor(3) as ex:   # SkyView returns spurious empty images under heavier load
    for gal, s, st in ex.map(fetch, tasks):
        n += 1
        if st not in ("ok", "skip"): bad.append((gal, s, st))
        print(f"  {n:3d} {gal} {s}: {st}", flush=True)
        if n % 50 == 0: print(f"  {n}/{len(tasks)}  problems so far: {len(bad)}", flush=True)
print("done; problems:", len(bad))
for b in bad: print("  ", *b)
