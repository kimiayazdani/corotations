"""Recall per corotation radius, alongside the paper's measures (final run, fixed literature tables).

recall  = fraction of gold-standard R_CR values (midpoints of the merged gold intervals) that have
          at least one predicted R_CR within 0.10 R25            -- "how many true radii were found"
purity  = fraction of predicted R_CR within 0.10 R25 of a gold value -- "how many predictions are right"
F1(R,P) = harmonic mean of recall and purity (the standard precision/recall F1)
Errors: bootstrap over galaxies (2000 resamples).  Published methods leave-one-out, >= 16 galaxies.
"""
import json
import numpy as np, pandas as pd
import eval_fixed
from eval_fixed import CR
from eval_lib import norm, cons, papers, RB
from select_v2 import predictions

b = json.load(open("run/v2_g_wide.json"))
P, RM = predictions(b, wrap=False, sense=None, arms=False, sigma=False, bar=None, kernel=1)
OURS = {norm(g): [c for lo, hi, c in v] for g, v in P.items() if norm(g) in RB}

def per_galaxy(cen, ex):
    rows = []
    for k, cs in cen.items():
        if k not in RB or not cs: continue
        ref = cons(k, ex)
        if not ref: continue
        gc = [(a + c) / 2 for a, c in ref]; tol = 0.10 * RB[k]
        hit_pred = [any(abs(c - v) <= tol for v in gc) for c in cs]      # each prediction correct?
        hit_gold = [any(abs(c - v) <= tol for c in cs) for v in gc]      # each gold radius recovered?
        rows.append((any(hit_pred), sum(hit_pred), len(hit_pred), sum(hit_gold), len(hit_gold)))
    return np.array(rows, dtype=float)

def stats(A):
    comp = A[:, 0].mean(); pur = A[:, 1].sum() / A[:, 2].sum(); rec = A[:, 3].sum() / A[:, 4].sum()
    f1 = 2 * pur * rec / (pur + rec) if pur + rec else 0
    return comp, pur, rec, f1

rows = []
for name, cen, ex in [("This work", OURS, papers)] + [(p, dict(CR[p]), papers - {p}) for p in sorted(papers)]:
    A = per_galaxy(cen, ex)
    if len(A) < 16: continue
    comp, pur, rec, f1 = stats(A)
    rng = np.random.default_rng(0)
    B = np.array([stats(A[rng.integers(0, len(A), len(A))]) for _ in range(2000)])
    rows.append(dict(method=name, n=len(A), gold_radii=int(A[:, 4].sum()), recovered=int(A[:, 3].sum()),
                     completeness=comp, purity=pur, recall=rec, recall_se=B[:, 2].std(ddof=1),
                     f1_recall_purity=f1, f1_se=B[:, 3].std(ddof=1)))
T = pd.DataFrame(rows).sort_values("recall", ascending=False)
T.to_csv("results/recall_test.csv", index=False)
print(f"{'method':32s} {'n':>3s} {'gold radii':>10s} {'found':>5s} | {'recall':>14s} | {'purity':>7s} | {'F1(recall,purity)':>17s} | {'completeness':>12s}")
for r in T.itertuples():
    print(f"{r.method[:32]:32s} {r.n:3d} {r.gold_radii:10d} {r.recovered:5d} | {100*r.recall:5.1f}% +/- {100*r.recall_se:4.1f} | {100*r.purity:6.1f}% | "
          f"{r.f1_recall_purity:6.2f} +/- {r.f1_se:.2f}     | {100*r.completeness:11.1f}%")
