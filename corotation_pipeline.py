"""Corotation pipeline (FITS-based), faster version of compare_est2.py.

Same science as compare_est2.py: WCS-resample both bands onto a common sky grid, deproject with
the second-moment ellipse of the redder band, take 34 rings over 0.08-1.25 R25 (width 0.04 R25),
fit A*cos(m*theta + c) + D to each band independently, store the phase, fit MSE and ring brightness.

Changes for speed only:
  * one BLAS/OpenMP thread per worker (the old run oversubscribed the machine);
  * ring profiles from a single np.bincount instead of 120 boolean masks per ring;
  * radius/angle maps computed once per galaxy.

R25 source (--r25):
  pgc       : independent catalogue value, r25_pgc.csv (PGC, Paturel et al. 2003) [default]
  benchmark : r_25 column of total_crs.csv (reproduces the draft's run)

Usage:  python3 corotation_pipeline.py --r25 pgc --out run/est_pgc.json [--workers 16]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse, glob, json, re, time, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "data", "total_crs.csv")
PGC = os.path.join(HERE, "r25_pgc.csv")
FITS_ROOT = os.environ.get("FITS_ROOT", os.path.join(HERE, "fits"))
SD = os.path.join(FITS_ROOT, "SDSS_FITS")
FB = os.path.join(FITS_ROOT, "Fallback_FITS")
M = 2; NR = 34; FR = (0.08, 1.25); NB = 120
TDEG = np.arange(NB) * (360 / NB)


def norm(n): return re.sub(r"[^A-Z0-9]+", "", str(n).upper())


def load_r25(source):
    R25 = {}
    if source == "pgc":
        for r in pd.read_csv(PGC).itertuples():
            if pd.notna(r.R25_arcsec) and r.R25_arcsec > 0:
                R25[norm(r.galaxy)] = float(r.R25_arcsec)
    else:
        lit = pd.read_csv(BENCH, sep=None, engine="python")
        for _, r in lit.iterrows():
            v = pd.to_numeric(r.get("r_25"), errors="coerce")
            if pd.notna(v) and v > 0:
                for k in (norm(r["obj_name"]), norm(r["article_name"])):
                    if k: R25.setdefault(k, float(v))
    return R25


def to_grid(path, npix, scale):
    h = fits.open(path)[0]; w = WCS(h.header); d = np.nan_to_num(h.data.astype(float))
    cy, cx = (h.header["NAXIS2"] - 1) / 2, (h.header["NAXIS1"] - 1) / 2
    ra0, dec0 = w.all_pix2world([[cx, cy]], 0)[0]
    y, x = np.mgrid[0:npix, 0:npix]
    dx = (x - npix / 2) * scale / 3600.0 / np.cos(np.radians(dec0)); dy = (y - npix / 2) * scale / 3600.0
    px, py = w.all_world2pix(np.stack([ra0 + dx.ravel(), dec0 + dy.ravel()], 1), 0).T
    return map_coordinates(d, [py.reshape(npix, npix), px.reshape(npix, npix)], order=1, mode="constant", cval=0.0)


def ellipse(img):
    v = img - np.median(img); v[v < 0] = 0
    if not (v > 0).any(): return None
    m = v > np.percentile(v[v > 0], 70)
    if m.sum() < 50: return None
    y, x = np.nonzero(m); w = v[m]
    cx, cy = np.average(x, weights=w), np.average(y, weights=w)
    xx = np.average((x - cx) ** 2, weights=w); yy = np.average((y - cy) ** 2, weights=w); xy = np.average((x - cx) * (y - cy), weights=w)
    ev, evec = np.linalg.eigh(np.array([[xx, xy], [xy, yy]]))
    return dict(cx=float(cx), cy=float(cy), ang=float(np.arctan2(evec[1, -1], evec[0, -1])),
                axr=float(min(max(np.sqrt(max(ev[0], 1e-9) / max(ev[-1], 1e-9)), 0.2), 1.0)))


def deproject(img, E, out):
    ca, sa = np.cos(-E["ang"]), np.sin(-E["ang"])
    A = np.array([[1.0, 0.0], [0.0, 1.0 / E["axr"]]]) @ np.array([[ca, -sa], [sa, ca]])
    Mt = np.zeros((2, 3)); Mt[:, :2] = A; Mt[:, 2] = np.array([out / 2, out / 2]) - A @ np.array([E["cx"], E["cy"]])
    return cv2.warpAffine(img, Mt, (out, out), flags=cv2.INTER_LINEAR, borderValue=0.0)


def ring(imgs, rr, binidx, R, w):
    """Mean-subtracted azimuthal profiles (one per image) and mean ring brightness, or None."""
    msk = np.abs(rr - R) < w / 2
    if msk.sum() < NB * 2: return None
    idx = binidx[msk]
    cnt = np.bincount(idx, minlength=NB)
    if (cnt == 0).any(): return None
    out = []
    for im in imgs:
        v = im[msk]
        prof = np.bincount(idx, weights=v, minlength=NB) / cnt
        out.append((prof - prof.mean(), float(np.mean(v))))
    return out


def _f(t, a, c, d): return a * np.cos(M * t * np.pi / 180 + c) + d


def fit(o):
    try:
        po, _ = curve_fit(_f, TDEG, o, p0=[float(np.std(o)), 0.0, 0.0],
                          bounds=([-np.inf, -np.pi, -np.inf], [np.inf, np.pi, np.inf]), maxfev=8000)
    except Exception:
        return None
    a, c, d = po
    if a < 0: a = -a; c = c + np.pi
    return float(-c / M), [float(x) for x in po]


def job(args):
    gal, pair, root, r25 = args
    try:
        span = 2.8 * r25
        npix = int(min(max(span / 0.6, 200), 900)); scale = span / npix
        ims = [to_grid(f"{root}/{gal}/{b}.fits", npix, scale) for b in pair]
        E = ellipse(ims[-1])
        if E is None: return gal, None
        dp = [deproject(im, E, npix) for im in ims]
        Y, X = np.mgrid[0:npix, 0:npix]
        rr = np.hypot(X - npix / 2, Y - npix / 2) * scale
        th = np.degrees(np.arctan2(-(Y - npix / 2), X - npix / 2)) % 360
        binidx = (th / (360 / NB)).astype(int) % NB
        out = {}
        for f in np.linspace(FR[0], FR[1], NR):
            R = f * r25; w = max(0.04 * r25, 2 * scale)
            rg = ring(dp, rr, binidx, R, w)
            if rg is None: continue
            (o0, b0), (o1, b1) = rg
            ia, ib = fit(o0), fit(o1)
            if not (ia and ib): continue
            m0 = float(np.mean((_f(TDEG, *ia[1]) - o0) ** 2)); m1 = float(np.mean((_f(TDEG, *ib[1]) - o1) ** 2))
            out[f"{f:.4f}"] = dict(ind=[ia[0], ib[0]], mse=[m0, m1], brt=[abs(b0), abs(b1)],
                                    amp=[abs(ia[1][0]), abs(ib[1][0])])
        return gal, out
    except Exception:
        return gal, None


def tasks_for(R25):
    tasks = []
    for d in sorted(glob.glob(f"{SD}/*/")):
        g = os.path.basename(d.rstrip("/"))
        if os.path.exists(d + "SDSSg.fits") and os.path.exists(d + "SDSSi.fits") and norm(g) in R25:
            tasks.append((g, ("SDSSg", "SDSSi"), SD, R25[norm(g)]))
    for d in sorted(glob.glob(f"{FB}/*/")):
        g = os.path.basename(d.rstrip("/"))
        if os.path.exists(d + "DSS2_Blue.fits") and os.path.exists(d + "WISE_3.4.fits") and norm(g) in R25:
            tasks.append((g, ("DSS2_Blue", "WISE_3.4"), FB, R25[norm(g)]))
    return tasks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r25", choices=["pgc", "benchmark"], default="pgc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    R25 = load_r25(a.r25); tasks = tasks_for(R25)
    t0 = time.time(); res = {}; meta = {}; r25used = {}
    with Pool(a.workers) as p:
        for g, o in p.imap_unordered(job, tasks):
            if o:
                res[g] = o
                meta[g] = "sdss" if os.path.exists(f"{SD}/{g}/SDSSg.fits") else "nonsdss"
                r25used[g] = R25[norm(g)]
    json.dump({"data": res, "batch": meta, "r25": r25used, "r25_source": a.r25}, open(a.out, "w"))
    print(f"{len(tasks)} galaxies, wrote {len(res)} to {a.out} in {time.time() - t0:.0f}s (R25 from {a.r25})")
