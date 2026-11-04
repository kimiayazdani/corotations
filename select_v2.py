"""Candidate selection for corotation_pipeline2.py outputs, with every rule as an option.

  wrap      : sign change counts only if Delta-phi changes by < 45 deg between neighbouring rings.
  sense     : rotation sense for the direction rule
                "own"    -> from the winding of the old-population arm phase (s = -sign(d theta/dR))
                "agree"  -> own winding, galaxies where it disagrees with SpArcFiRe chirality dropped
                "sparc"  -> SpArcFiRe chirality (S-wise -> +1, the majority mapping on SDSS)
                None     -> no direction rule
  arms      : both bands' m=2 amplitude >= 5% of ring brightness on both bracketing rings.
  sigma     : interval = centre +/- (Delta-phi uncertainty / local slope), from the fit covariance,
              floored at half a ring step.  Otherwise the original adaptive-threshold interval.
  bar       : "exclude" -> drop crossings inside the bar; "add" -> also add a candidate at 1.2 R_bar
              (interval 1.0-1.4 R_bar).  The bar is detected from our own red-band data: the run of
              rings from the centre whose phase stays within 15 deg of the innermost ring's phase
              with m=2 amplitude >= 10% of the ring brightness, at least 3 rings long.
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LO, HI = 0.15, 0.60
WRAP = np.radians(45); AMIN = 0.05
BAR_DPHI = np.radians(15); BAR_AMIN = 0.10; BAR_MIN_RINGS = 3

_CHIR = None
def sparc_chirality():
    global _CHIR
    if _CHIR is None:
        w = pd.read_csv(os.path.join(HERE, "run", "winding_check.csv"))
        _CHIR = {r.g: (+1 if r.ch == "S-wise" else -1 if r.ch == "Z-wise" else 0) for r in w.itertuples()}
    return _CHIR


def bar_radius(fr, th_red, rel_red):
    """Outer radius of the bar (fraction of R25), or None."""
    n = 0
    for i in range(len(fr)):
        d = np.angle(np.exp(2j * (th_red[i] - th_red[0]))) / 2
        if abs(d) <= BAR_DPHI and rel_red[i] >= BAR_AMIN: n = i + 1
        else: break
    return fr[n - 1] if n >= BAR_MIN_RINGS else None


def candidates(gal, prof, wrap=True, sense="own", arms=True, sigma=False, bar=None, kernel=3, lo_hi=(LO, HI)):
    ks = [k for k in sorted(prof, key=float) if prof[k].get("ind")]
    if len(ks) < 15: return []
    fr = np.array([float(x) for x in ks])
    dp = np.array([prof[x]["ind"][0] - prof[x]["ind"][1] for x in ks])
    z = np.exp(2j * dp); k3 = np.ones(kernel) / kernel
    ds = np.angle(np.convolve(z.real, k3, "same") + 1j * np.convolve(z.imag, k3, "same")) / 2
    rel = np.array([min(prof[x]["amp"][0] / max(prof[x]["brt"][0], 1e-9), prof[x]["amp"][1] / max(prof[x]["brt"][1], 1e-9)) for x in ks])
    rel_red = np.array([prof[x]["amp"][1] / max(prof[x]["brt"][1], 1e-9) for x in ks])
    th_red = np.unwrap(2 * np.array([prof[x]["ind"][1] for x in ks])) / 2
    inner = (fr >= 0.15) & (fr <= 0.80)
    own = -np.sign(np.polyfit(fr[inner], th_red[inner], 1)[0]) if inner.sum() > 3 else 0
    if sense == "own": s = own
    elif sense == "agree":
        sp = sparc_chirality().get(gal, 0)
        s = own if (sp != 0 and own == sp) else None              # mapping: S-wise <-> own = +1
        if s is None: return []
    elif sense == "sparc": s = sparc_chirality().get(gal, 0)
    else: s = 0
    if sigma and "sig" in prof[ks[0]]:
        sd = np.array([np.hypot(*prof[x]["sig"]) for x in ks])
        sd_s = np.sqrt(np.convolve(sd ** 2, k3 ** 2, "same"))      # uncertainty of the 3-point average
    rbar = bar_radius(fr, th_red, rel_red) if bar else None
    grad = np.gradient(ds, fr); base = max(0.01 * float(ds.max() - ds.min()), 1e-8); eps = 1e-8
    step = float(np.median(np.diff(fr))); out = []
    for i in range(len(fr) - 1):
        y1, y2 = ds[i], ds[i + 1]
        if not (np.sign(y1) == 0 or np.sign(y1) * np.sign(y2) < 0): continue
        if wrap and abs(y2 - y1) >= WRAP: continue
        if s and np.sign(s * (y2 - y1)) != -1: continue           # s*Delta-phi must go + -> - outward
        if arms and min(rel[i], rel[i + 1]) < AMIN: continue
        c = fr[i] - y1 * (fr[i + 1] - fr[i]) / (y2 - y1) if y2 != y1 else fr[i]
        c = min(max(c, fr[i]), fr[i + 1])
        if not (lo_hi[0] <= c <= lo_hi[1]): continue
        if rbar is not None and bar in ("exclude", "add") and c <= rbar: continue
        if sigma and "sig" in prof[ks[0]]:
            slope = abs(y2 - y1) / (fr[i + 1] - fr[i])
            half = max(np.nanmax([sd_s[i], sd_s[i + 1]]) / max(slope, 1e-9), step / 2)
            lo, hi = c - half, c + half
        else:
            L = i
            while L >= 0 and abs(ds[L]) <= base * (1 + 1 / (abs(grad[L]) + eps)): L -= 1
            R = i + 1
            while R < len(ds) and abs(ds[R]) <= base * (1 + 1 / (abs(grad[R]) + eps)): R += 1
            lo, hi = sorted((fr[max(L + 1, 0)], fr[min(R - 1, len(fr) - 1)]))
            if hi - lo < step: lo, hi = c - step / 2, c + step / 2
        out.append((lo, hi, c))
    if bar == "add" and rbar is not None and LO <= 1.2 * rbar <= HI:
        out.append((1.0 * rbar, 1.4 * rbar, 1.2 * rbar))
    return out


def predictions(blob, **kw):
    """{galaxy: [(lo, hi, centre)] in arcsec} and {norm(galaxy): R25} for eval_lib.evaluate."""
    from eval_lib import norm
    RM = {norm(g): v for g, v in blob["r25"].items()}; P = {}
    for g, prof in blob["data"].items():
        iv = candidates(g, prof, **kw)
        if iv: P[g] = [(lo * RM[norm(g)], hi * RM[norm(g)], c * RM[norm(g)]) for lo, hi, c in iv]
    return P, RM
