"""Fix for the literature double-counting in score_est.py / compare_mine_vs_others_multipaper.py.

The original code files every row of total_crs.csv under BOTH norm(obj_name) and norm(article_name):
when the two keys are equal the value is stored twice; when they differ the row creates a second,
spurious "galaxy".  Importing this module rebuilds eval_lib's literature tables in place with each row
stored exactly once, under the catalogue name (obj_name) only.
"""
from collections import defaultdict
import pandas as pd
import eval_lib as E

_lit = pd.read_csv(f"{E.D}/total_crs.csv", sep=None, engine="python")
E.paper_iv.clear(); E.R25.clear()
for _, r in _lit.iterrows():
    k = E.norm(r["obj_name"])
    if not k: continue
    v = pd.to_numeric(r.get("r_25"), errors="coerce")
    if pd.notna(v) and v > 0: E.R25.setdefault(k, float(v))
    p = str(r.get("paper_name", "")).strip()
    iv = E.pti(r.get("cr_range", ""))
    if p and iv: E.paper_iv[p][k].append(iv)
E.papers.clear(); E.papers.update(E.paper_iv.keys())
CR = defaultdict(lambda: defaultdict(list))          # published point values, once per row
for _, r in _lit.iterrows():
    p = str(r.get("paper_name", "")).strip()
    if p and pd.notna(r.get("cr")) and E.norm(r["obj_name"]): CR[p][E.norm(r["obj_name"])].append(float(r["cr"]))
