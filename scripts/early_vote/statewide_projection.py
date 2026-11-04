"""Election night statewide projection of T (the statewide down-ballot judicial margin), every county.

For each county: swing from its 2024 judicial margin.
  complete counties (all precincts in, Election Day counted) -> observed final swing
  early-only counties -> early swing x k, where k = final/early swing learned from Texas counties that
                         are complete (shrunk toward a prior from North Carolina's complete counties, 0.6)
  counties with nothing -> predicted from a vote-weighted regression of the (estimated) final swing on
                         county demographics: Hispanic share of CVAP, white college share of CVAP,
                         2016-24 judicial swing, log 2024 votes
Projected 2026 county margin = 2024 margin + swing; statewide T = turnout-weighted mean.
Sources: state feed snapshot (all counties) overridden by county feeds (Clarity, Harris, Collin PDF).
    python scripts/early_vote/statewide_projection.py
"""
import glob
import gzip
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
EN = ROOT / "data" / "raw" / "election_night"
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
K_PRIOR, K_PRIOR_VOTES = 0.6, 400_000          # prior weight equivalent in judicial votes
CLARITY = {"TRAVIS": "Travis/127492", "TARRANT": "Tarrant/127297", "DALLAS": "Dallas/127396", "BEXAR": "Bexar/127291", "DENTON": "Denton/127398", "EL PASO": "El_Paso/127348", "WILLIAMSON": "Williamson/127488", "FORT BEND": "Fort_Bend/127480"}
TGRID = [(-8.0, 65.2, None), (-7.2, 66.0, 0.020), (-6.0, 67.3, 0.042), (-4.8, 68.7, 0.082), (-3.5, 70.4, None), (-2.0, 72.5, 0.262), (0.4, 76.2, 0.504), (2.4, 79.6, 0.708), (2.9, 80.4, 0.753), (4.0, 82.4, 0.835)]
JUD = re.compile(r"SUPREME COURT|CRIMINAL APPEALS", re.I)


def get(url, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, headers=HDR, timeout=90)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 + 2 * i)
    return None


# ---------- baselines and demographics ----------
e = pd.read_csv(ROOT / "data/raw/early_vote/county_panel2.csv", index_col=0)
z = zipfile.ZipFile(ROOT / "data/raw/_cvap_cache/CVAP_2020-2024_ACS_csv_files.zip")
cv = pd.read_csv(io.BytesIO(z.read("County.csv")), encoding="latin-1")
cv = cv[cv.geoid.str.startswith("0500000US48")]
cv["county"] = cv.geoname.str.replace(" County, Texas", "", regex=False).str.upper()
piv = cv.pivot_table(index="county", columns="lntitle", values="cvap_est", aggfunc="sum")
dem = pd.DataFrame({"cvap": piv["Total"], "hisp": piv["Hispanic or Latino"] / piv["Total"], "wnh": piv["White Alone"] / piv["Total"]})
ed = pd.read_csv(ROOT / "data/raw/_acs_cache/acsdt5y2024-c15002h.dat", sep="|", dtype=str)
ed = ed[ed.GEO_ID.str.startswith("0500000US48")]
geo2c = cv.drop_duplicates("geoid").set_index("geoid").county
ed["county"] = ed.GEO_ID.map(geo2c)
ed = ed.dropna(subset=["county"]).set_index("county")
dem["wcol"] = (ed.C15002H_E006.astype(float) + ed.C15002H_E011.astype(float)) / dem.cvap
base = e[["j2016", "j2024", "n2024", "ed2022", "tot2022", "ed2024", "tot2024"]].join(dem, how="left")
base["rsw"] = base.j2024 - base.j2016
base = base.dropna(subset=["j2024", "n2024"])

# ---------- current results: state feed, then county overrides ----------
res = {}   # county -> dict(De, Re, Dv, Rv, complete, src)
snaps = sorted(glob.glob(str(EN / "countyInfo_*.json.gz")), key=os.path.getmtime)
ci = json.load(gzip.open(snaps[-1], "rt", encoding="utf-8")) if snaps else {}
for c in ci.values():
    cn = {"LASALLE": "LA SALLE"}.get(c["N"].upper(), c["N"].upper())
    a = {"De": 0, "Re": 0, "Dv": 0, "Rv": 0, "pr": 0, "tp": 0}
    for r in c["Races"].values():
        if r.get("OT") == "SW" and JUD.search(r["N"]):
            for x in r["C"].values():
                if x["P"] == "DEM":
                    a["De"] += x.get("EV") or 0; a["Dv"] += x.get("V") or 0
                elif x["P"] == "REP":
                    a["Re"] += x.get("EV") or 0; a["Rv"] += x.get("V") or 0
            a["pr"], a["tp"] = r.get("PR", 0) or 0, r.get("TPR", r.get("TP", 0)) or 0
    a["complete"] = a["tp"] > 0 and a["pr"] >= a["tp"] and (a["Dv"] + a["Rv"]) > (a["De"] + a["Re"])
    a["src"] = "state"
    res[cn] = a
LIVE_CLARITY = False   # 9pm: Clarity returning 502s; county_poll.py is the one fetcher, its saved files are read below
for county, path in (CLARITY.items() if LIVE_CLARITY else []):
    b = f"https://results.enr.clarityelections.com/TX/{path}"
    r = get(f"{b}/current_ver.txt")
    if r is None or not r.text.strip().isdigit():
        continue
    zr = get(f"{b}/{r.text.strip()}/reports/detailxml.zip")
    try:
        x = zipfile.ZipFile(io.BytesIO(zr.content)).read("detail.xml").decode("utf-8", "replace")
    except Exception:
        continue
    ts = (re.findall(r"<Timestamp>([^<]*)</Timestamp>", x) or [""])[0]
    try:
        all_early = time.strptime(ts.replace(" CST", "").strip(), "%m/%d/%Y %I:%M:%S %p") < time.strptime("11/3/2026 8:00:00 PM", "%m/%d/%Y %I:%M:%S %p")
    except ValueError:
        all_early = False
    a = {"De": 0, "Re": 0, "Dv": 0, "Rv": 0, "pr": 0, "tp": 0}
    for cm in re.finditer(r'<Contest key="\d+" text="([^"]+)"([^>]*)>(.*?)</Contest>', x, re.S):
        if not JUD.search(cm.group(1)):
            continue
        prm = re.search(r'precinctsReported="(\d+)"', cm.group(2)); tpm = re.search(r'precinctsParticipating="(\d+)"', cm.group(2))
        if prm and tpm:
            a["pr"], a["tp"] = int(prm.group(1)), int(tpm.group(1))
        for ch in re.finditer(r'<Choice key="\d+" text="[^"]*" party="([^"]*)" totalVotes="(\d+)">(.*?)</Choice>', cm.group(3), re.S):
            p = ch.group(1)
            if p not in ("DEM", "REP"):
                continue
            for vt, n in re.findall(r'<VoteType name="([^"]+)" votes="(\d+)"', ch.group(3)):
                if re.search(r"overvote|undervote", vt, re.I):
                    continue
                eday = re.match(r"election", vt, re.I) and not all_early
                a[("D" if p == "DEM" else "R") + "v"] += int(n)
                if not eday:
                    a[("D" if p == "DEM" else "R") + "e"] += int(n)
    a["complete"] = a["tp"] > 0 and a["pr"] >= a["tp"] and (a["Dv"] + a["Rv"]) > (a["De"] + a["Re"])
    a["src"] = "county"
    if a["Dv"] + a["Rv"] > 0:
        res[county] = a
# Clarity counties the live fetch missed (502s): newest saved county_poll file {contest: {DEM: [early, eday, total], ...}}
for county in CLARITY:
    if res.get(county, {}).get("src") == "county":
        continue
    for f in sorted(glob.glob(str(EN / f"clarity_{county.replace(' ', '_')}_*.json")), key=os.path.getmtime, reverse=True):
        js = json.loads(Path(f).read_text())
        a = {"De": 0, "Re": 0, "Dv": 0, "Rv": 0, "pr": 0, "tp": 0, "complete": False, "src": "saved"}
        if isinstance(js, dict):
            for name, v in js.items():
                if JUD.search(name):
                    a["De"] += v["DEM"][0]; a["Re"] += v["REP"][0]
                    a["Dv"] += v["DEM"][0] + v["DEM"][1]; a["Rv"] += v["REP"][0] + v["REP"][1]
        elif isinstance(js, list):
            for cc in js:
                if JUD.search(cc["C"]):
                    D = sum(v for p, v in zip(cc["P"], cc["V"]) if p == "DEM"); Rr = sum(v for p, v in zip(cc["P"], cc["V"]) if p == "REP")
                    a["De"] += D; a["Re"] += Rr; a["Dv"] += D; a["Rv"] += Rr
        if a["Dv"] + a["Rv"] > 0:
            sos = res.get(county)
            # the state feed wins when it is at least as far along (it also carries precincts for completeness)
            if not (sos and sos.get("Dv", 0) + sos.get("Rv", 0) >= a["Dv"] + a["Rv"]):
                res[county] = a
            break
r = get(f"https://appfiles.harrisvotes.com/harrisvotes/prd/Data/1126/resultsbyballot.json?nocache={int(time.time() // 60 * 60000)}")
if r is not None:
    a = {"De": 0, "Re": 0, "Dv": 0, "Rv": 0, "pr": 0, "tp": 0}
    for c in r.json().get("contests", []):
        if JUD.search(c["t"]):
            for i, nm in enumerate(c["ca"]):
                p = "D" if "(DEM)" in nm else "R" if "(REP)" in nm else None
                if p:
                    a[p + "e"] += c["ev"][i] + c["ab"][i]; a[p + "v"] += c["tv"][i]
            a["pr"], a["tp"] = c.get("pr", 0), c.get("tp", 0)
    a["complete"] = a["tp"] > 0 and a["pr"] >= a["tp"] and (a["Dv"] + a["Rv"]) > (a["De"] + a["Re"])
    a["src"] = "county"
    if a["Dv"] + a["Rv"] > 0:
        res["HARRIS"] = a
cp = sorted(glob.glob(str(EN / "collin" / "*.txt")), key=os.path.getmtime)
if cp:
    txt, cur, tot = Path(cp[-1]).read_text(encoding="utf-8"), None, {"DEM": 0, "REP": 0}
    for line in txt.splitlines():
        s = line.strip()
        m = re.match(r"^(REP|DEM)\s+(.+?)\s+([\d,]+)\s+[\d.]+%", s)
        if m and cur:
            tot[m.group(1)] += int(m.group(3).replace(",", ""))
        elif re.search(r"[A-Za-z]", s) and not m and not re.match(r"^(Vote For|TOTAL|Mail|Total Votes|Over|Under|Contest|Summary|General|November|Election Summary|Report|Collin|Watermark|Statistics|Registered|Ballots|Voter)", s):
            cur = s if JUD.search(s) else None
    nm = Path(cp[-1]).name.lower()
    early = "early" in nm or "_ev_" in nm
    a = {"De": tot["DEM"] if early else 0, "Re": tot["REP"] if early else 0, "Dv": tot["DEM"], "Rv": tot["REP"], "pr": 0, "tp": 1, "complete": False, "src": "county"}
    if not early:   # an election-day-inclusive report: early-only numbers come from the early report
        evf = [f for f in cp if "_ev_" in Path(f).name.lower() or "early" in Path(f).name.lower()]
        if evf:
            cur2, t2 = None, {"DEM": 0, "REP": 0}
            for line in Path(evf[0]).read_text(encoding="utf-8").splitlines():
                s = line.strip()
                m = re.match(r"^(REP|DEM)\s+(.+?)\s+([\d,]+)\s+[\d.]+%", s)
                if m and cur2:
                    t2[m.group(1)] += int(m.group(3).replace(",", ""))
                elif re.search(r"[A-Za-z]", s) and not m and not re.match(r"^(Vote For|TOTAL|Mail|Total Votes|Over|Under|Contest|Summary|General|November|Election Summary|Report|Collin|Watermark|Statistics|Registered|Ballots|Voter)", s):
                    cur2 = s if JUD.search(s) else None
            a["De"], a["Re"] = t2["DEM"], t2["REP"]
    res["COLLIN"] = a

R = pd.DataFrame(res).T
for col in ("De", "Re", "Dv", "Rv", "pr", "tp"):
    R[col] = pd.to_numeric(R[col])
R["complete"] = R.complete.astype(bool)
mg = lambda d, r: np.where(d + r > 0, (d - r) / np.maximum(d + r, 1) * 100, np.nan)
R["m_early"] = mg(R.De, R.Re)
R["m_total"] = mg(R.Dv, R.Rv)
R["n_early"] = R.De + R.Re
R["n_total"] = R.Dv + R.Rv
t = base.join(R, how="left")
t["sw_early"] = t.m_early - t.j2024
# "complete" also needs a believable Election Day share (10:55pm): Harris shows every precinct "reporting" once its
# early vote is loaded, so a trickle of E-day votes made it look finished. Require E-day >= half its usual share.
_s = ((t.ed2022 / t.tot2022) + (t.ed2024 / t.tot2024)) / 2
t["complete"] = (t.complete == True) & (((t.n_total - t.n_early) / t.n_total) >= 0.5 * _s.fillna(0.2))
t["sw_final_obs"] = np.where(t.complete == True, t.m_total - t.j2024, np.nan)

# ---------- Election Day factor k ----------
c = t[(t.complete == True) & t.sw_early.notna() & (t.sw_early.abs() > 1)]
if len(c):
    w = c.n_total
    k_hat = float((w * c.sw_final_obs * c.sw_early).sum() / (w * c.sw_early ** 2).sum())
    k = (k_hat * w.sum() + K_PRIOR * K_PRIOR_VOTES) / (w.sum() + K_PRIOR_VOTES)
else:
    k_hat, k = np.nan, K_PRIOR
# partial Election Day evidence (added 9:58pm): a county with E-day votes counted but not complete is projected
# directly as (1 - s) x early margin + s x E-day margin so far, s = its usual E-day share (mean of 2022 and 2024
# ballots), trusted more as more of its expected E-day vote is in; the rest of the weight stays on k x early swing.
t["m_eday"] = mg(t.Dv - t.De, t.Rv - t.Re)
t["n_eday"] = t.n_total - t.n_early
t["s_hat"] = ((t.ed2022 / t.tot2022) + (t.ed2024 / t.tot2024)) / 2
# 5am Nov 4: fully counted counties' 2026 E-day share ran well under their 2022/24 share; scale s_hat by the median
# ratio so partial counties (Harris, Dallas) are not projected with too much weight on their E-day vote
_cc = t[(t.complete == True) & (t.n2024 > 20000) & (t.n_total > 0) & (t.s_hat > 0)]
ED_RATIO = float(np.median(((_cc.n_total - _cc.n_early) / _cc.n_total) / _cc.s_hat)) if len(_cc) >= 5 else 1.0
t["s26"] = (t.s_hat * ED_RATIO).clip(0.02, 0.8)
part = (t.complete != True) & (t.n_eday > 3000) & t.m_eday.notna() & t.m_early.notna() & t.s_hat.between(0.02, 0.8)
exp_eday = t.s26 / (1 - t.s26) * t.n_early
t["w_dir"] = np.where(part, np.clip(t.n_eday / exp_eday, 0, 1) ** 0.5, 0.0)
t["sw_dir"] = (1 - t.s26) * t.m_early + t.s26 * t.m_eday - t.j2024
t["sw_est"] = np.where(t.sw_final_obs.notna(), t.sw_final_obs,
                       np.where(part, t.w_dir * t.sw_dir + (1 - t.w_dir) * t.sw_early * k, t.sw_early * k))
t["status"] = np.where(t.sw_final_obs.notna(), "complete", np.where(part, "early + partial E-day", np.where(t.sw_early.notna(), "early only", "none")))
# metro factor (added 10:22pm): the big counties' partial Election Day batches imply a lower factor than the rural
# complete counties (Dallas ~0.66, Williamson ~0.64, Tarrant ~0.74, Bexar ~0.89, El Paso ~0.95). Early-only metros
# (2024 votes over 100k) take the pooled implied factor of the partial metros, weighted by size x evidence weight.
pm = t[(t.status == "early + partial E-day") & (t.n2024 > 100000) & (t.sw_early.abs() > 5)]
K_METRO = k
if len(pm):
    wt = pm.n2024 * pm.w_dir
    k_metro_hat = float(np.average(pm.sw_dir / pm.sw_early, weights=wt))
    K_METRO = (k_metro_hat * wt.sum() + k * 150_000) / (wt.sum() + 150_000)
    big_early = (t.status == "early only") & (t.n2024 > 100000)
    t.loc[big_early, "sw_est"] = t.loc[big_early, "sw_early"] * K_METRO
    # partial metros: the rest of their weight goes on the metro factor too, not the rural one
    pmask = (t.status == "early + partial E-day") & (t.n2024 > 100000)
    t.loc[pmask, "sw_est"] = t.loc[pmask, "w_dir"] * t.loc[pmask, "sw_dir"] + (1 - t.loc[pmask, "w_dir"]) * t.loc[pmask, "sw_early"] * K_METRO

# ---------- demographic regression for counties with nothing ----------
X = ["hisp", "wcol", "rsw", "lnv"]
t["lnv"] = np.log(t.n2024)
fit = t[(t.status != "none")].dropna(subset=X + ["sw_est"])
coef = None
if len(fit) >= 10:
    W = np.sqrt(fit.n2024.values)
    A = np.column_stack([np.ones(len(fit))] + [fit[x].values for x in X]) * W[:, None]
    coef, *_ = np.linalg.lstsq(A, fit.sw_est.values * W, rcond=None)
    pred = coef[0] + sum(coef[i + 1] * t[x] for i, x in enumerate(X))
    t["sw_pred_demo"] = pred
    t.loc[t.status == "none", "sw_est"] = t.loc[t.status == "none", "sw_pred_demo"]
    resid = fit.sw_est - (coef[0] + sum(coef[i + 1] * fit[x] for i, x in enumerate(X)))
    rmse = float(np.sqrt(np.average(resid ** 2, weights=fit.n2024)))
# ---------- turnout weights ----------
# weights: each county's expected 2026 vote (5am Nov 4 fix). Until then this used the 2024 vote, which put T about
# 1.6 points too R once counts were in: 2026 turnout fell much more in rural R counties than in the metros.
# complete: its count; partial / early only: early / (1 - 2026 E-day share), that share = the ratio fully counted
# counties ran against their 2022/24 share; none: 2024 vote x the median 2026/2024 turnout ratio. (n_* sum over the
# statewide judicial races, n2024 is per race, so turnout ratio = n_total / (races x n2024).)
_c = t[(t.status == "complete") & (t.n2024 > 20000) & (t.n_total > 0)]
NJ = 7
_turn = float(np.median(_c.n_total / (NJ * _c.n2024))) if len(_c) else 0.85
_s26 = t.s26.fillna(0.2 * ED_RATIO)
t["w"] = np.where(t.status == "complete", t.n_total / NJ,
                  np.where(t.status.isin(["early + partial E-day", "early only"]),
                           np.maximum(t.n_early / (1 - _s26), t.n_total) / NJ, t.n2024 * _turn))
t["w"] = t["w"].where(t["w"] > 0, t.n2024 * _turn)
W_NOTE = f"weights = 2026 votes (E-day share {ED_RATIO:.2f}x its 2022/24 level, turnout {_turn:.2f}x 2024)"
t["m26"] = t.j2024 + t.sw_est
T = float(np.average(t.m26.dropna(), weights=t.w[t.m26.notna()]))
sw = float(np.average(t.sw_est.dropna(), weights=t.w[t.sw_est.notna()]))
xs = [g[0] for g in TGRID]
house = float(np.interp(T, xs, [g[1] for g in TGRID]))
p76 = float(np.interp(T, [g[0] for g in TGRID if g[2] is not None], [g[2] for g in TGRID if g[2] is not None]))

# ---------- report ----------
share = t.groupby("status").w.sum() / t.w.sum() * 100
print(f"=== STATEWIDE PROJECTION {time.strftime('%H:%M:%S')} ===")
print(f"  expected vote by status: " + ", ".join(f"{s} {v:.0f}%" for s, v in share.items()) + f" | counties: " + ", ".join(f"{s} {n}" for s, n in t.status.value_counts().items()))
print(f"  Election Day factor k (final swing / early swing): Texas complete counties {k_hat:.2f} on {int(c.n_total.sum()) if len(c) else 0:,} votes ({len(c)} counties); used {k:.2f} (prior {K_PRIOR} from NC)")
print(f"  metro factor for big counties (2024 votes over 100k) without complete counts: {K_METRO:.2f}")
pp = t[t.status == "early + partial E-day"].sort_values("n2024", ascending=False)
if len(pp):
    print("  partial E-day counties (implied k = direct final swing / early swing, weight on direct): " + "; ".join(
        f"{c_.title()} {r_.sw_dir / r_.sw_early:.2f} (w {r_.w_dir:.2f})" for c_, r_ in pp.head(8).iterrows() if abs(r_.sw_early) > 1))
if coef is not None:
    print(f"  demographic fit of swing (n={len(fit)}, rmse {rmse:.1f}): const {coef[0]:+.1f}, per +10pt Hispanic {coef[1] * 0.1:+.1f}, per +10pt white-college {coef[2] * 0.1:+.1f}, per point of 2016-24 swing {coef[3]:+.2f}, per log-size {coef[4]:+.1f}")
print(f"  statewide swing from 2024 judicial {sw:+.1f} -> PROJECTED T {T:+.1f}   (2024 judicial R+16.2; model center -6.0)")
print(f"  {W_NOTE}")
for kk in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    tt = t.copy()
    tt["sw2"] = np.where(tt.status == "complete", tt.sw_final_obs,
                         np.where(tt.status == "early + partial E-day", tt.w_dir * tt.sw_dir + (1 - tt.w_dir) * tt.sw_early * kk,
                                  np.where(tt.status == "early only", tt.sw_early * kk, np.nan)))
    if coef is not None:
        f2 = tt[tt.sw2.notna()].dropna(subset=X)
        W2 = np.sqrt(f2.n2024.values)
        A2 = np.column_stack([np.ones(len(f2))] + [f2[x].values for x in X]) * W2[:, None]
        c2, *_ = np.linalg.lstsq(A2, f2.sw2.values * W2, rcond=None)
        tt.loc[tt.sw2.isna(), "sw2"] = c2[0] + sum(c2[i + 1] * tt.loc[tt.sw2.isna(), x] for i, x in enumerate(X))
    mm = tt.j2024 + tt.sw2
    T2 = float(np.average(mm.dropna(), weights=tt.w[mm.notna()]))
    print(f"    if k = {kk:.2f}: T {T2:+.1f}, TX House ~{np.interp(T2, xs, [g[1] for g in TGRID]):.1f} D seats")
print(f"  MODEL at projected T: TX House ~{house:.1f} D seats, P(76+) ~{p76:.0%}  (model at -6.0: 67.3 / 4%)")
big = ["HARRIS", "DALLAS", "TARRANT", "BEXAR", "TRAVIS", "COLLIN", "DENTON", "FORT BEND", "WILLIAMSON", "EL PASO", "HIDALGO", "CAMERON", "WEBB", "NUECES", "MONTGOMERY"]
tb = t.loc[[b for b in big if b in t.index], ["status", "j2024", "m_early", "m_total", "sw_early", "sw_est", "hisp", "wcol"]].copy()
print(tb.round(2).to_string())
t.to_csv(EN / f"projection_{time.strftime('%H%M%S')}.csv")
