"""Bundle the latest race board into one JSON rows file for the phone dashboard.

Rows carry an `id` (unique), a `kind` (summary | statewide | seat | alert) and display strings, so the
page shows exactly what the data holds. Reads poll.log (projection + house lines + alerts), the newest
statewide_races_latest.csv and the newest seat_report_*.csv.
    python scripts/early_vote/board_export.py  -> data/raw/election_night/board.json
"""
import glob
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

EN = Path(r"C:\Users\Brennan\code\tx-leg-model\data\raw\election_night")
log = (EN / "poll.log").read_text(encoding="utf-8", errors="replace")
fmt = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"D+{x:.1f}" if x >= 0 else f"R+{-x:.1f}")
rows = []

blk = log[log.rfind("=== STATEWIDE PROJECTION"):]
g = lambda pat, s=blk: (re.search(pat, s).group(1) if re.search(pat, s) else None)
T = float(g(r"PROJECTED T ([+-][\d.]+)") or "nan")
k_used = float(g(r"used ([\d.]+) \(prior") or "nan")
k_tx = g(r"Texas complete counties ([\d.]+|nan) on")
n_complete = g(r"\((\d+) counties\); used")
none_share = g(r"none (\d+)%")
early_share = str(100 - int(none_share or 0))   # every county with any returns in
hl = [l for l in log.splitlines() if "=== HOUSE PROJECTION" in l]
h = hl[-1] if hl else ""
seats = float(g(r"expected D seats ([\d.]+)", h) or "nan")
p76 = float(g(r"P\(76\+\) (\d+)%", h) or "nan") / 100
when = time.strftime("%-I:%M %p") if os.name != "nt" else time.strftime("%I:%M %p").lstrip("0")
KT = {float(a): float(b) for a, b in re.findall(r"if k = ([\d.]+): T ([+-][\d.]+)", blk)}
HK = {float(a): float(b) for a, b in re.findall(r"([\d.]+)->([\d.]+)", g(r"\[by k: ([^\]]+)\]", h) or "")}
T_rng = f"{fmt(KT[0.6])} to {fmt(KT[0.9])}" if 0.6 in KT and 0.9 in KT else "n/a"
H_rng = f"{HK[0.6]:.0f} to {HK[0.9]:.0f}" if 0.6 in HK and 0.9 in HK else "n/a"
summ = [("T", "Statewide down-ballot margin (T)", T, fmt(T)), ("house", "Texas House Democratic seats, expected", seats, f"{seats:.1f}"),
        ("p76", "Chance of a Democratic House majority (76+)", p76, f"{p76:.0%}"), ("k", "Election Day factor in use", k_used, f"{k_used:.2f}"),
        ("ktx", "Election Day factor, Texas counties fully counted", float(k_tx) if k_tx not in (None, "nan") else float("nan"), k_tx or "n/a"),
        ("ncomplete", "Counties fully counted", float(n_complete or 0), n_complete or "0"),
        ("reported", "Expected vote with early returns in", float(early_share or 0), f"{early_share or 0}%"),
        ("updated", "Updated", 0.0, when),
        ("T_range", "T if the Election Day factor is 0.6 to 0.9", None, T_rng), ("house_range", "House D seats if the factor is 0.6 to 0.9", None, H_rng),
        ("T_pre", "Model before the polls closed: T", -6.0, "R+6.0"), ("house_pre", "Model before the polls closed: House D seats", 67.3, "67.3"),
        ("p76_pre", "Model before the polls closed: majority chance", 0.042, "4%")]
for i, (id_, label, v, disp) in enumerate(summ):
    rows.append({"id": id_, "kind": "summary", "label": label, "value": None if isinstance(v, float) and np.isnan(v) else v, "disp": disp, "order": i})

sw = EN / "statewide_races_latest.csv"
NAMES = {"comptroller": "Comptroller", "land": "Land Commissioner", "agriculture": "Agriculture: Tucker v Sheets"}
if sw.exists():
    s = pd.read_csv(sw)
    s["jud"] = s.key.str.match(r"^(sc|cca|coa15)-")      # executive races first, then the courts
    s = s.sort_values(["jud", "proj"], ascending=[True, False])
    for i, r in enumerate(s.itertuples()):
        lab = NAMES.get(r.key, r.label)
        # called when even the low end of the range (Election Day factor 0.6) keeps the same leader
        call = ("Called D" if r.lo > 0 else "Called R") if (r.lo > 0) == (r.hi > 0) else "Uncalled"
        rows.append({"id": f"sw-{r.key}", "kind": "statewide", "race": lab, "proj": r.proj, "proj_disp": fmt(r.proj), "range_disp": f"{fmt(r.lo)} to {fmt(r.hi)}",
                     "lo": r.lo, "hi": r.hi, "call_disp": call,
                     "lead": "D" if r.proj >= 0 else "R", "order": i})

# seats (5:45am Nov 4): split into called (99%+ either way) and uncalled, with the leader's name and chance.
# Called rows: every flip, plus holds decided by under 3 points. Uncalled rows: everything else short of 99%.
files = sorted(glob.glob(str(EN / "seat_report_*.csv")), key=os.path.getmtime)
if files:
    t = pd.read_csv(files[-1])
    allseats = t.copy()
    nodata = t[t.pproj.isna()]
    t = t[t.pproj.notna()]
    t["called"] = (t.pproj >= 0.99) | (t.pproj <= 0.01)
    t["isflip"] = ((t.held == "R") & (t.pproj > 0.5)) | ((t.held == "D") & (t.pproj < 0.5))
    t = t[(~t.called) | t.isflip | (t.proj.abs() < 3)].sort_values("pproj", ascending=False)

    def names(race):
        m = re.match(r"^(.*?) \(R\) v (.*?) \(D\)$", str(race))
        return (m.group(1).replace("*", ""), m.group(2).replace("*", "")) if m else ("R", "D")

    for i, r in enumerate(t.itertuples()):
        a, b = (int(x) for x in str(r.pieces).split("/"))
        flip = bool(r.isflip)
        lab = ("HD " if r.seat.startswith("hd") else "SD ") + r.seat[2:]
        anchor = r.model - 0.95 * (-6.0 + 16.2)          # same anchor as seat_report.py
        lo, hi = anchor + 0.6 * (r.early - anchor), anchor + 0.9 * (r.early - anchor)
        race = str(r.race).replace("*", " (inc.)")
        rn, dn = names(r.race)
        lead_d = r.pproj > 0.5
        who = f"{dn} (D)" if lead_d else f"{rn} (R)"
        final = str(getattr(r, "final", False)) == "True"
        votes = abs(r.now) / 100 * r.votes if not np.isnan(r.now) else np.nan
        margin = f"by {votes:,.0f} votes" if final and abs(r.now) < 3 else fmt(r.proj)
        result = f"{who} {'flip' if flip else 'hold'}, {margin}"
        lead = f"{who} {max(r.pproj, 1 - r.pproj):.0%}"
        note = ("fully counted, " + f"{'D' if r.now > 0 else 'R'} {margin}") if final else f"count now {fmt(r.now)}" + (f", {a} of {b} counties in" if a < b else "")
        rows.append({"id": r.seat, "kind": "called" if r.called else "uncalled", "seat_label": lab, "race": race, "seat_race": f"{lab} · {race} · {r.held}-held", "held": f"{r.held}-held",
                     "early_disp": fmt(r.early), "proj_disp": fmt(r.proj), "range_disp": f"{fmt(lo)} to {fmt(hi)}", "lo": round(lo, 2), "hi": round(hi, 2), "proj": round(float(r.proj), 2),
                     "pd": round(float(r.pproj), 3), "pd_disp": f"{r.pproj:.0%}", "model_disp": f"{r.p:.0%}",
                     "result_disp": result, "lead_disp": lead, "note_disp": note,
                     "status": ("Called: " if r.called else "Uncalled, leaning ") + ("flip" if flip else "hold") + (", fully counted" if final else "") + (f", partial {a} of {b}" if a < b else ""),
                     "flip": flip, "order": i})
    for j, r in enumerate(nodata.itertuples()):
        lab = ("HD " if r.seat.startswith("hd") else "SD ") + r.seat[2:]
        race = str(r.race).replace("*", " (inc.)")
        rows.append({"id": r.seat, "kind": "uncalled", "seat_label": lab, "race": race, "seat_race": f"{lab} · {race} · {r.held}-held",
                     "lead_disp": "No votes reported yet", "note_disp": f"model had {fmt(r.model)}", "pd": None, "order": 1000 + j})
    # tallies over every seat, only when the newest seat file covers all of them (SEATS=all)
    if allseats.seat.str.startswith("hd").sum() == 150:
        h = allseats[allseats.seat.str.startswith("hd")]
        cd, cr = int((h.pproj >= 0.99).sum()), int((h.pproj <= 0.01).sum())
        rows.append({"id": "house_called", "kind": "summary", "label": "Texas House seats called", "value": cd, "disp": f"{cd} D · {cr} R · {150 - cd - cr} uncalled", "order": 50})
        rows.append({"id": "majority_call", "kind": "summary", "label": "House majority", "value": None,
                     "disp": "Called for Democrats" if cd >= 76 else "Called for Republicans" if cr >= 76 else "Not called yet", "order": 51})
        s_ = allseats[allseats.seat.str.startswith("sd")]
        dist = pd.read_csv(Path(r"C:\Users\Brennan\code\tx-leg-model\data\processed\districts_2026.csv"))
        hold_r = int(((dist.chamber.str.lower().str.startswith("s")) & (dist.up_in_2026 != True) & (dist.incumbent_party == "R")).sum())
        sd_, sr_ = int((s_.pproj >= 0.99).sum()), int((s_.pproj <= 0.01).sum())
        ctl = "Republicans keep control" if hold_r + sr_ >= 16 else "Not called yet"
        rows.append({"id": "senate_call", "kind": "summary", "label": "Texas Senate", "value": None,
                     "disp": f"{ctl} · {sd_ + sr_} of {len(s_)} seats on the ballot called, no flips" if not (((s_.held == "R") & (s_.pproj >= .99)).any() or ((s_.held == "D") & (s_.pproj <= .01)).any()) else f"{ctl} · {sd_ + sr_} of {len(s_)} called", "order": 52})

al = [l.replace("BOARD| ", "").strip() for l in log.splitlines() if "ALERT" in l]
seen, uniq = set(), []
for l in reversed(al):
    if l not in seen:
        seen.add(l)
        uniq.append(l)
for i, l in enumerate(uniq[:20]):
    rows.append({"id": f"alert-{i}", "kind": "alert", "text": l.replace("ALERT ", ""), "order": i})

# Austin council and propositions, from the browser poller's Travis file
lt = EN / "local_TRAVIS_latest.json"
if lt.exists():
    L = json.loads(lt.read_text())["contests"]

    def counted(c):
        ed = sum(ch["eday"] for ch in c["choices"])
        if not ed:
            return "early vote only"
        return f"Election Day {c['pr']} of {c['tp']} precincts" if c.get("tp") else "some Election Day in"

    for i, (name, c) in enumerate(sorted(((n, c) for n, c in L.items() if n.startswith("City of Austin, City Council Member")), key=lambda x: int(re.search(r"(\d+)$", x[0]).group(1)))):
        chs = sorted(c["choices"], key=lambda ch: -ch["total"])
        tot = sum(ch["total"] for ch in chs)
        if not tot:
            continue
        lead, sec = chs[0], chs[1] if len(chs) > 1 else None
        sh = lead["total"] / tot * 100
        # 5am Nov 4: Travis's election-night count is complete (state feed 309 of 309), so 50% is the line
        status = ("Over 50%, wins outright" if sh > 50 else "Under 50%, runoff") + " · Travis count " + json.loads(lt.read_text())["timestamp"].split(" ", 1)[1].replace(" CST", "")
        rows.append({"id": f"council-{i}", "kind": "council", "race": "District " + re.search(r"(\d+)$", name).group(1),
                     "leader_disp": f"{lead['name']} {sh:.1f}%", "second_disp": f"{sec['name']} {sec['total'] / tot * 100:.1f}%" if sec else "", "status": status, "order": i})
    props = {}
    for n, c in L.items():
        m = re.match(r"City of Austin Proposition ([A-Z])", n)
        if m:
            a = props.setdefault(m.group(1), {"for": 0, "against": 0, "c": c})
            for ch in c["choices"]:
                if ch["name"].lower().startswith("for"):
                    a["for"] += ch["total"]
                elif ch["name"].lower().startswith("against"):
                    a["against"] += ch["total"]
            if "Full Purpose" in n:
                a["c"] = c
    for i, (k, a) in enumerate(sorted(props.items())):
        tot = a["for"] + a["against"]
        if not tot:
            continue
        sh = a["for"] / tot * 100
        rows.append({"id": f"prop-{k}", "kind": "prop", "prop": f"Prop {k}", "for_disp": f"{sh:.1f}%", "status": ("Passing" if sh > 50 else "Failing") + f" · {counted(a['c'])}", "order": i})

(EN / "board.json").write_text(json.dumps(rows, indent=0))
print(f"board.json: {len(rows)} rows; T {T:+.1f}, seats {seats:.1f}, p76 {p76:.0%}")
