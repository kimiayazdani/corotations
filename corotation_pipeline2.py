"""Corotation pipeline v2 (FITS-based).  Extends corotation_pipeline.py with:

  * wide cutouts (wide_FITS/, 3 R25 across), falling back to the original cutouts if missing;
  * a coverage mask: rings touching any pixel outside the real image are skipped
    (the old version read zeros there, which adds a fake m=2 signal after deprojection);
  * selectable young-population band: g (default), u, nuv, fuv;
  * per-ring phase uncertainty from the cosine fit covariance, and the m=2 amplitude.

Usage:  python3 corotation_pipeline2.py --blue g --out run/v2_g.json
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse, glob, json, time, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import corotation_pipeline as cp
warnings.filterwarnings("ignore")
SMOOTH = 0.0   # profile smoothing (bins), set with --smooth

HERE = os.path.dirname(os.path.abspath(__file__))
WIDE = os.path.join(cp.FITS_ROOT, "wide_FITS")
ORIG = {"SDSSg": cp.SD, "SDSSi": cp.SD, "SDSSu": cp.SD, "DSS2_Blue": cp.FB, "WISE_3.4": cp.FB}
BLUE = {"sdss": {"g": "SDSSg", "u": "SDSSu", "nuv": "GALEX_Near_UV", "fuv": "GALEX_Far_UV"},
        "nonsdss": {"g": "DSS2_Blue", "u": "DSS2_Blue", "nuv": "GALEX_Near_UV", "fuv": "GALEX_Far_UV"}}
RED = {"sdss": "SDSSi", "nonsdss": "WISE_3.4"}


def band_path(gal, band, wide=True):
    p = f"{WIDE}/{gal}/{band}.fits"
    if wide and os.path.exists(p): return p
    if band in ORIG and os.path.exists(f"{ORIG[band]}/{gal}/{band}.fits"): return f"{ORIG[band]}/{gal}/{band}.fits"
    return None


def to_grid(path, npix, scale, ra0=None, dec0=None):
    """Resample onto a common sky grid; also return the coverage map (1 inside the image, 0 outside)."""
    h = fits.open(path)[0]; w = WCS(h.header).celestial; d = np.nan_to_num(np.squeeze(h.data).astype(float))
    if ra0 is None:
        cy, cx = (d.shape[0] - 1) / 2, (d.shape[1] - 1) / 2
        ra0, dec0 = w.all_pix2world([[cx, cy]], 0)[0]
    y, x = np.mgrid[0:npix, 0:npix]
    dx = (x - npix / 2) * scale / 3600.0 / np.cos(np.radians(dec0)); dy = (y - npix / 2) * scale / 3600.0
    px, py = w.all_world2pix(np.stack([ra0 + dx.ravel(), dec0 + dy.ravel()], 1), 0).T
    coords = [py.reshape(npix, npix), px.reshape(npix, npix)]
    img = map_coordinates(d, coords, order=1, mode="constant", cval=0.0)
    cov = map_coordinates(np.ones_like(d), coords, order=1, mode="constant", cval=0.0)
    return img, cov, ra0, dec0


def fit(o):
    try:
        po, pc = curve_fit(cp._f, cp.TDEG, o, p0=[float(np.std(o)), 0.0, 0.0],
                           bounds=([-np.inf, -np.pi, -np.inf], [np.inf, np.pi, np.inf]), maxfev=8000)
    except Exception:
        return None
    a, c, d = po
    sig = float(np.sqrt(max(pc[1, 1], 0)) / cp.M) if np.all(np.isfinite(pc)) else np.nan
    if a < 0: a = -a; c = c + np.pi
    return float(-c / cp.M), [float(x) for x in po], sig, float(a)


def job(args):
    gal, pair, r25, wide = args
    try:
        span = 2.8 * r25
        npix = int(min(max(span / 0.6, 200), 900)); scale = span / npix
        paths = [band_path(gal, b, wide) for b in pair]
        if None in paths: return gal, None
        red, rcov, ra0, dec0 = to_grid(paths[1], npix, scale)           # centre on the old-population image
        blue, bcov, _, _ = to_grid(paths[0], npix, scale, ra0, dec0)
        E = cp.ellipse(red * (rcov > 0.99))
        if E is None: return gal, None
        dp = [cp.deproject(im, E, npix) for im in (blue, red)]
        cov = np.minimum(cp.deproject(bcov, E, npix), cp.deproject(rcov, E, npix))
        Y, X = np.mgrid[0:npix, 0:npix]
        rr = np.hypot(X - npix / 2, Y - npix / 2) * scale
        th = np.degrees(np.arctan2(-(Y - npix / 2), X - npix / 2)) % 360
        binidx = (th / (360 / cp.NB)).astype(int) % cp.NB
        out = {}
        for f in np.linspace(cp.FR[0], cp.FR[1], cp.NR):
            R = f * r25; w = max(0.04 * r25, 2 * scale)
            msk = np.abs(rr - R) < w / 2
            if msk.any() and cov[msk].min() < 0.99: continue                # ring leaves the real image
            rg = cp.ring(dp, rr, binidx, R, w)
            if rg is None: continue
            (o0, b0), (o1, b1) = rg
            if SMOOTH > 0:                     # Gaussian smoothing of the azimuthal profile (as described in the paper)
                o0 = gaussian_filter1d(o0, SMOOTH, mode="wrap"); o1 = gaussian_filter1d(o1, SMOOTH, mode="wrap")
            ia, ib = fit(o0), fit(o1)
            if not (ia and ib): continue
            m0 = float(np.mean((cp._f(cp.TDEG, *ia[1]) - o0) ** 2)); m1 = float(np.mean((cp._f(cp.TDEG, *ib[1]) - o1) ** 2))
            out[f"{f:.4f}"] = dict(ind=[ia[0], ib[0]], sig=[ia[2], ib[2]], amp=[ia[3], ib[3]],
                                   mse=[m0, m1], brt=[abs(b0), abs(b1)])
        return gal, out
    except Exception:
        return gal, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blue", choices=["g", "u", "nuv", "fuv"], default="g")
    ap.add_argument("--orig", action="store_true", help="use the original (narrow) cutouts only")
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smooth", type=float, default=0.0, help="Gaussian smoothing of each ring profile, in bins")
    a = ap.parse_args(); SMOOTH = a.smooth
    R25 = cp.load_r25("pgc")
    split = pd.read_csv(os.path.join(HERE, "split.csv"))
    tasks = [(r.galaxy, (BLUE[r.batch][a.blue], RED[r.batch]), R25[cp.norm(r.galaxy)], not a.orig)
             for r in split.itertuples() if cp.norm(r.galaxy) in R25]
    batch = dict(zip(split.galaxy, split.batch))
    t0 = time.time(); res = {}; used = {}
    with Pool(a.workers) as p:
        for (g, pair, r25, _), (_, o) in zip(tasks, p.imap(job, tasks)):
            if o: res[g] = o; used[g] = r25
    json.dump({"data": res, "batch": {g: batch[g] for g in res}, "r25": used, "r25_source": "pgc",
               "blue": a.blue, "wide": not a.orig, "smooth": a.smooth}, open(a.out, "w"))
    nr = [len(v) for v in res.values()]
    print(f"{len(tasks)} galaxies, wrote {len(res)} to {a.out} in {time.time()-t0:.0f}s; "
          f"rings/galaxy median {np.median(nr):.0f}")
