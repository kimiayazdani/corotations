import os, json, numpy as np, pandas as pd, re, random
from collections import defaultdict
SP=None
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")
def norm(n): return re.sub(r"[^A-Z0-9]+","",str(n).upper())
def pti(s):
    if not isinstance(s,str) or not s.strip(): return None
    m=re.match(r"[\(\[\{]\s*([+-]?\d+(\.\d+)?)\s*,\s*([+-]?\d+(\.\d+)?)\s*[\)\]\}]",s.strip())
    return (min(float(m.group(1)),float(m.group(3))),max(float(m.group(1)),float(m.group(3)))) if m else None
def ov(a,b): return not (a[1]<b[0] or b[1]<a[0])
def isect(a,b): return (max(a[0],b[0]),min(a[1],b[1])) if ov(a,b) else None
def merge(i):
    if not i: return []
    s=sorted(i); o=[s[0]]
    for lo,hi in s[1:]:
        if lo<=o[-1][1]: o[-1]=(o[-1][0],max(o[-1][1],hi))
        else: o.append((lo,hi))
    return o
lit=pd.read_csv(f"{D}/total_crs.csv",sep=None,engine="python")
paper_iv=defaultdict(lambda: defaultdict(list)); R25={}
for _,r in lit.iterrows():
    ks=[k for k in (norm(r["obj_name"]),norm(r["article_name"])) if k]
    if pd.notna(r.get("r_25")):
        try: v=float(r["r_25"])
        except Exception: v=0
        if v>0:
            for k in ks: R25.setdefault(k,v)
    p=str(r.get("paper_name","")).strip()
    if not p: continue
    for c in ["cr_range","err_range"]:
        v=pti(r.get(c,""))
        if v:
            for k in ks: paper_iv[p][k].append(v)
papers=set(paper_iv)
def cons(g,ms):
    L=[list(paper_iv[m].get(g,[])) for m in ms if paper_iv[m].get(g)]
    if len(L)<2: return []
    c=[]
    for i in range(len(L)):
        for j in range(i+1,len(L)):
            for x in L[i]:
                for y in L[j]:
                    q=isect(x,y)
                    if q: c.append(q)
    return merge(c) if c else []
wr=lambda x:(x+np.pi/2)%np.pi-np.pi/2
LO,HI=0.15,0.60
def crossings(fr,dp,thr=0.01,alpha=1.0):
    z=np.exp(2j*dp); k=np.ones(3)/3
    ds=np.angle(np.convolve(z.real,k,'same')+1j*np.convolve(z.imag,k,'same'))/2
    grad=np.gradient(ds,fr); base=max(thr*float(ds.max()-ds.min()),1e-8); eps=1e-8
    sg=np.sign(ds); step=float(np.median(np.diff(fr))); out=[]
    for i in range(len(fr)-1):
        if not (sg[i]==0 or sg[i]*sg[i+1]<0): continue
        y1,y2=ds[i],ds[i+1]
        c=fr[i]-y1*(fr[i+1]-fr[i])/(y2-y1) if y2!=y1 else fr[i]
        c=min(max(c,fr[i]),fr[i+1])
        L=i
        while L>=0 and abs(ds[L])<=base*(1+alpha/(abs(grad[L])+eps)): L-=1
        Rr=i+1
        while Rr<len(ds) and abs(ds[Rr])<=base*(1+alpha/(abs(grad[Rr])+eps)): Rr+=1
        lo,hi=fr[max(L+1,0)],fr[min(Rr-1,len(fr)-1)]
        lo,hi=min(lo,hi),max(lo,hi)
        if hi-lo<step: lo,hi=c-step/2,c+step/2
        out.append((lo,hi,c))
    return out
blob=json.load(open(f"{SP}/est.json")); data=blob["data"]; batch=blob["batch"]
def build(key,subset=None):
    P={}; nrad=[]
    for g,prof in data.items():
        nk=norm(g)
        if nk not in R25: continue
        if subset and batch.get(g)!=subset: continue
        ks=[k for k in sorted(prof,key=float) if prof[k].get(key)]
        if len(ks)<15: continue
        nrad.append(len(ks))
        fr=np.array([float(x) for x in ks])
        if key=="fou": dp=np.array([wr(prof[x]["fou"][0]-prof[x]["fou"][1]) for x in ks])
        else:          dp=np.array([wr(prof[x][key][0]-prof[x][key][1]) for x in ks])
        iv=[t for t in crossings(fr,dp) if LO<=t[2]<=HI]
        if iv: P[nk]=iv
    return P,(np.mean(nrad) if nrad else 0)
def score(P,label,nrad):
    GC={}; 
    for g in P:
        c=cons(g,papers)
        if c: GC[g]=[(a+b)/2 for a,b in c]
    ks=[g for g in P if g in GC]
    if len(ks)<10: print(f"{label:34s} too few ({len(ks)})"); return
    tol=0.10
    rec=np.mean([any(any(abs(x[2]*R25[g]-v)<=tol*R25[g] for v in GC[g]) for x in P[g]) for g in ks])
    def hit(Q):
        t=c=0
        for g in ks:
            tt=tol*R25[g]
            for x in Q.get(g,[]):
                t+=1
                if any(abs(x[2]*R25[g]-v)<=tt for v in GC[g]): c+=1
        return c/t if t else 0
    pre=hit(P); nl=[hit({a:P[b] for a,b in zip(ks,random.Random(i).sample(ks,len(ks)))}) for i in range(300)]
    nu=float(np.mean(nl)); f1=2*rec*pre/(rec+pre) if rec+pre else 0
    ncand=np.mean([len(P[g]) for g in ks])
    print(f"{label:34s} {len(ks):5d} {nrad:6.1f} {ncand:6.2f} {rec:7.1%} {pre:8.1%} {f1:6.2f} {pre/nu if nu else 0:6.2f}x")
print(f"{'estimator':34s} {'nGal':>5s} {'radii':>6s} {'cand':>6s} {'recall':>7s} {'precis':>8s} {'F1':>6s} {'lift':>6s}")
print("-"*84)
for key,lab in [("fou","Fourier (closed form)"),("prj","curve_fit, project-style"),("ind","curve_fit, independent")]:
    P,nr=build(key); score(P,lab,nr)
print("\n-- SDSS subsample only (g-i) --")
for key,lab in [("fou","Fourier"),("prj","curve_fit project-style"),("ind","curve_fit independent")]:
    P,nr=build(key,"sdss"); score(P,lab,nr)
print("\n-- non-SDSS subsample only (B-W1) --")
for key,lab in [("fou","Fourier"),("prj","curve_fit project-style"),("ind","curve_fit independent")]:
    P,nr=build(key,"nonsdss"); score(P,lab,nr)
