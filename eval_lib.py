"""Shared evaluation for corotation predictions (same numbers as score_pipeline.py + scatter_test.py).

Predictions P: {galaxy: [(lo, hi, centre), ...]} in arcsec.  RM: {norm(galaxy): R25 used for the
measurement}.  The gold standard and the 0.10 R25 tolerance follow score_pipeline.py.
"""
import os, random
import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "score_est.py")).read().split("blob=json.load")[0])
RB = R25   # benchmark r_25: scoring yardstick


def evaluate(P, RM, n_null=400):
    P = {norm(g): v for g, v in P.items() if v and norm(g) in RM and norm(g) in RB}
    GOLD = {g: cons(g, papers) for g in P}
    ks = [g for g in P if GOLD[g]]
    GC = {g: [(x + y) / 2 for x, y in GOLD[g]] for g in ks}
    hit = lambda g, c: any(abs(c - v) <= 0.10 * RB[g] for v in GC[g])
    comp = float(np.mean([any(hit(g, c) for lo, hi, c in P[g]) for g in ks]))
    def purity(Q):
        t = c = 0
        for g in ks:
            for x in Q.get(g, []): t += 1; c += hit(g, x)
        return c / t if t else 0
    pur = purity({g: [c for lo, hi, c in P[g]] for g in ks})
    frac = {g: [c / RM[g] for lo, hi, c in P[g]] for g in ks}
    null = float(np.mean([purity({x: [f * RM[x] for f in frac[y]] for x, y in zip(ks, random.Random(i).sample(ks, len(ks)))})
                          for i in range(n_null)]))
    ok = ti = ci = 0
    for g in ks:
        ivs = [(lo, hi) for lo, hi, c in P[g]]; ti += len(ivs)
        h = [any(ov(x, y) for y in GOLD[g]) for x in ivs]; ci += sum(h); ok += any(h)
    xs, ys = [], []
    for g in ks:
        keep = [c for lo, hi, c in P[g] if (hi - lo) / RM[g] <= 0.15]
        if keep:
            ys.append(np.mean(keep) / RM[g]); xs.append(np.median(GC[g]) / RM[g])
    xs, ys = np.array(xs), np.array(ys)
    r = pearsonr(xs, ys)[0] if len(xs) > 2 else np.nan
    mad = float(np.median(np.abs(ys - xs))) if len(xs) else np.nan
    rng = np.random.default_rng(0)
    pnull = float(np.mean([np.median(np.abs(rng.permutation(ys) - xs)) <= mad for _ in range(2000)])) if len(xs) else np.nan
    f1 = 2 * comp * pur / (comp + pur) if comp + pur else 0
    return dict(n=len(ks), cand=float(np.mean([len(P[g]) for g in ks])), comp=comp, pur=pur, f1=f1,
                ratio=pur / null if null else 0, gal_ov=ok / len(ks), rcr_ov=ci / ti if ti else 0,
                n_sc=len(xs), r=r, mad=mad, p_shuffle=pnull)


HEADER = (f"{'method':44s} {'n':>3s} {'cand':>4s} {'comp':>6s} {'pur':>6s} {'F1':>5s} {'ratio':>6s} "
          f"{'ovGal':>6s} {'ovRCR':>6s} {'r':>6s} {'MAD':>6s} {'p_shuf':>6s}")


def row(label, m):
    return (f"{label:44s} {m['n']:3d} {m['cand']:4.2f} {100*m['comp']:5.1f}% {100*m['pur']:5.1f}% {m['f1']:5.2f} "
            f"{m['ratio']:5.2f}x {100*m['gal_ov']:5.1f}% {100*m['rcr_ov']:5.1f}% {m['r']:6.3f} {m['mad']:6.3f} {m['p_shuffle']:6.3f}")


def overlap_matched(P, widths=(0.05, 0.10, 0.20), min_gal=16):
    """Overlap criterion with every method's interval width set to the same value.

    Each prediction (ours: candidate centre; a published paper: midpoint of each reported interval)
    becomes centre +/- w/2, with w a fixed fraction of the benchmark R25, the same for every method.
    The gold standard is unchanged (published intervals; ours scored against all papers, each paper
    leave-one-out).  Returns {width: [(method, n_galaxies, galaxy_acc, per_rcr_acc), ...]} sorted
    by per-R_CR accuracy.
    """
    ours = {norm(g): [c for lo, hi, c in v] for g, v in P.items() if v}
    out = {}
    for w in widths:
        rows = []
        for p in list(papers) + ["THIS WORK"]:
            if p == "THIS WORK": cen, ex = ours, papers
            else: cen, ex = {g: [(a + b) / 2 for a, b in v] for g, v in paper_iv[p].items()}, papers - {p}
            ok = ti = ci = sc = 0
            for g, cs in cen.items():
                if g not in RB or not cs: continue
                ref = cons(g, ex)
                if not ref: continue
                hw = w * RB[g] / 2; sc += 1
                h = [any(ov((c - hw, c + hw), y) for y in ref) for c in cs]
                ti += len(h); ci += sum(h); ok += any(h)
            if sc >= min_gal: rows.append((p, sc, ok / sc, ci / ti))
        out[w] = sorted(rows, key=lambda r: -r[3])
    return out
