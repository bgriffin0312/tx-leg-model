"""Election-night poller for the Texas SOS results feed (Civix ENR), 2026-11-03.

Polls the state feed, and each time it updates, reads every county's EARLY votes (the EV field) in the
statewide judicial races (Supreme Court + Court of Criminal Appeals, both parties on the ballot) and
compares them with the model's early-only benchmark: 2024 judicial lean + county shift + early-to-final
drift (same construction as ev_night_benchmarks.py). Implied T for a county = its early judicial margin
minus that benchmark base. Prints event lines for a Monitor; snapshots go to data/raw/election_night/.

    python scripts/early_vote/election_night_poll.py            # loop
    python scripts/early_vote/election_night_poll.py --once     # one pass, verbose parse check
"""
import base64
import datetime as dt
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
S = ROOT / "data" / "raw" / "early_vote"
OUT = ROOT / "data" / "raw" / "election_night"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://goelect.txelections.civixapps.com/api-ivis-system/api/s3/enr"
EID = 53815
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
K = 0.9
POLL = 45
ONCE = "--once" in sys.argv

BIG9 = ["HARRIS", "DALLAS", "TARRANT", "BEXAR", "TRAVIS", "COLLIN", "DENTON", "FORT BEND", "WILLIAMSON"]
WATCH = BIG9 + ["HAYS", "NUECES", "EL PASO", "HIDALGO", "CAMERON", "WEBB", "MONTGOMERY", "BRAZORIA", "GALVESTON", "BELL", "LUBBOCK"]
BORDER = ["HIDALGO", "CAMERON", "WEBB", "STARR", "MAVERICK", "EL PASO", "NUECES", "ZAPATA", "WILLACY", "VAL VERDE", "JIM HOGG", "BROOKS", "DUVAL", "KLEBERG", "ZAVALA", "DIMMIT", "KINNEY", "PRESIDIO", "REEVES", "LA SALLE", "JIM WELLS", "SAN PATRICIO", "BEE", "UVALDE", "HUDSPETH", "CULBERSON"]
# whole-election model, T read -> TX House D seats, P(76+), Talarico chance (docs/whole-election-model.md)
TGRID = [(-8.0, 65.2, None, 0.37), (-7.2, 66.0, 0.020, 0.47), (-6.0, 67.3, 0.042, 0.62), (-4.8, 68.7, 0.082, 0.75), (-3.5, 70.4, None, 0.86), (3.1, 80.8, 0.77, 0.999)]


def log(msg):
    print(msg, flush=True)


# ---------- benchmarks for every county ----------
e = pd.read_csv(S / "county_panel2.csv", index_col=0)
Ed = json.load(open(S / "coldspark_eday_2026-10-30.json"))
ok = e.j2024.notna() & e.n2024.notna()
J24 = float(np.average(e.j2024[ok], weights=e.n2024[ok]))
nb = e.sort_values("n2024").tail(60).index.difference(BORDER)
w = e.t2026[nb]
sig = {n: -(e.loc[nb, n] - np.average(e.loc[nb, n], weights=w)) for n in ("d24", "adj")}
cshift = (0.5 * (sig["d24"] + sig["adj"]) / 2).reindex(e.index).fillna(0.0)
base = {}
for c in e.index[ok]:
    drifts = []
    for y in ("2022", "2024"):
        p, t = Ed["p"][y].get(c), Ed["y"][y].get(c)
        if not p or not t or not t[0]:
            continue
        rem = np.array(p[:3], float) - np.array(p[3:], float)
        if rem.sum() <= 0:
            continue
        drifts.append(((rem[0] - rem[1]) / rem.sum() * 100, t[1] / t[0]))
    if drifts and not np.isnan(e.E2026[c]):
        eds = np.mean([d[1] for d in drifts])
        drift = np.mean([eds * (pool - e.E2026[c]) for pool, _ in drifts])
    else:
        drift = 0.0
    base[c] = {"lean": e.j2024[c] - J24, "cshift": float(cshift[c]), "drift": K * drift, "base": e.j2024[c] - J24 + float(cshift[c]) + K * drift}
B = pd.DataFrame(base).T

# model race means at T -6.0 with county shifts (final, not early-only)
rs = pd.read_csv(S / "race_shift.csv").set_index("race")
SEATS = [r for r in rs.index if (r.startswith(("hd", "sd", "tx")) and 0.04 < rs.loc[r, "p1"] < 0.96)] + ["sd9"]
CJ = {"tarrant": "TARRANT", "williamson": "WILLIAMSON", "collin": "COLLIN", "denton": "DENTON", "fort_bend": "FORT BEND", "harris": "HARRIS", "bexar": "BEXAR", "el_paso": "EL PASO", "montgomery": "MONTGOMERY"}


def fmt(m):
    return "n/a" if m is None or np.isnan(m) else (f"D+{m:.1f}" if m >= 0 else f"R+{-m:.1f}")


def interp(T):
    xs = [g[0] for g in TGRID]
    house = float(np.interp(T, xs, [g[1] for g in TGRID]))
    tal = float(np.interp(T, xs, [g[3] for g in TGRID]))
    return house, tal


# ---------- feed ----------
def dec(b64):
    raw = base64.b64decode(b64)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def get(path):
    r = requests.get(f"{API}/{path}", headers=HDR, timeout=60)
    r.raise_for_status()
    return r.json()


def party_votes(race):
    d = {"DEM": [0, 0], "REP": [0, 0]}
    for cand in race.get("C", {}).values():
        p = cand.get("P")
        if p in d:
            d[p][0] += cand.get("EV") or 0
            d[p][1] += cand.get("V") or 0
    return d


RX = {"hd": re.compile(r"STATE REPRESENTATIVE\D*(\d+)"), "sd": re.compile(r"STATE SENATOR\D*(\d+)"), "tx": re.compile(r"(?:U\. ?S\. ?|UNITED STATES )REPRESENTATIVE\D*(\d+)")}


def parse(ci):
    rows, seats = [], {}
    names_seen = set()
    for cid, cty in ci.items():
        cn = cty.get("N", "").upper().strip()
        cn = {"LASALLE": "LA SALLE"}.get(cn, cn)
        acc = {"county": cn, "jD": 0, "jR": 0, "jDv": 0, "jRv": 0}
        for race in cty.get("Races", {}).values():
            nm = (race.get("N") or "").upper().strip()
            names_seen.add(nm)
            pv = party_votes(race)
            both = all(pv[p][1] > 0 or any(c.get("P") == p for c in race.get("C", {}).values()) for p in ("DEM", "REP"))
            if race.get("OT") == "SW" and re.search(r"SUPREME COURT|CRIMINAL APPEALS", nm) and both:
                acc["jD"] += pv["DEM"][0]; acc["jR"] += pv["REP"][0]; acc["jDv"] += pv["DEM"][1]; acc["jRv"] += pv["REP"][1]
            key = None
            if nm == "ATTORNEY GENERAL":
                key = "ag"
            elif nm == "LIEUTENANT GOVERNOR":
                key = "ltg"
            elif nm == "GOVERNOR":
                key = "gov"
            elif "SENATOR" in nm and "STATE" not in nm:
                key = "sen"
            if key:
                acc[key + "D"], acc[key + "R"] = pv["DEM"][0], pv["REP"][0]
                acc[key + "Dv"], acc[key + "Rv"] = pv["DEM"][1], pv["REP"][1]
            seat = None
            for pre, rx in RX.items():
                m = rx.search(nm)
                if m:
                    seat = f"{pre}{int(m.group(1))}"
                    break
            if seat is None and nm == "COUNTY JUDGE":
                seat = cn.lower().replace(" ", "_")
            if seat:
                s = seats.setdefault(seat, {"D": 0, "R": 0, "Dv": 0, "Rv": 0, "n": 0, "rep": 0})
                s["D"] += pv["DEM"][0]; s["R"] += pv["REP"][0]; s["Dv"] += pv["DEM"][1]; s["Rv"] += pv["REP"][1]
                s["n"] += 1
                s["rep"] += int(pv["DEM"][1] + pv["REP"][1] > 0)
        rows.append(acc)
    return pd.DataFrame(rows).set_index("county"), seats, names_seen


def m(D, R):
    return (D - R) / (D + R) * 100 if (D + R) > 0 else np.nan


def summarize(df, seats, home, first=False):
    now = dt.datetime.now().strftime("%H:%M:%S")
    cr = home.get("CountiesReporting", {})
    df = df.copy()
    df["jm"] = [m(a, b) for a, b in zip(df.jD, df.jR)]
    df["jmv"] = [m(a, b) for a, b in zip(df.jDv, df.jRv)]
    df["n"] = df.jD + df.jR
    df = df.join(B, how="left")
    df["T_imp"] = df.jm - df.base
    rep = df[df.n > 0]
    lines = [f"[{now}] SOS feed {home.get('LastUpdatedTime')} | counties {cr.get('CR')}/{cr.get('CT')} | counties with early judicial votes {len(rep)}"]
    for c in WATCH:
        if c in rep.index:
            r = rep.loc[c]
            sen = m(r.get("senD", 0), r.get("senR", 0)); ag = m(r.get("agD", 0), r.get("agR", 0)); gov = m(r.get("govD", 0), r.get("govR", 0))
            lines.append(f"  {c}: judicial early {fmt(r.jm)} vs bench(T-6) {fmt(r.base - 6.0)} -> dev {r.jm - (r.base - 6.0):+.1f}, implied T {r.T_imp:+.1f} | AG {fmt(ag)} | Sen {fmt(sen)} (Sen-AG {sen - ag:+.1f}) | Gov {fmt(gov)} | {int(r.n):,} early jud votes")
    def comp(idx, lab):
        x = rep.loc[rep.index.intersection(idx)].dropna(subset=["T_imp"])
        if len(x) == 0:
            return None
        T = float(np.average(x.T_imp, weights=x.n))
        return T, len(x), f"{lab}: implied T {T:+.1f} ({len(x)} counties, {int(x.n.sum()):,} votes)"
    nonb = rep.index.difference(BORDER)
    parts = [comp(BIG9, "big-9"), comp(nonb, "all non-border"), comp(rep.index.intersection(BORDER), "border")]
    parts = [p for p in parts if p]
    if parts:
        lines.append("  COMPOSITE " + " | ".join(p[2] for p in parts))
        ref = parts[0][0] if parts[0][2].startswith("big-9") and parts[0][1] >= 5 else (parts[1][0] if len(parts) > 1 and "non-border" in parts[1][2] else parts[0][0])
        h, tal = interp(ref)
        lines.append(f"  MODEL at that T (rough): TX House D ~{h:.1f} seats; Talarico ~{tal:.0%} (model at T-6.0: 67.3 / 62%)")
    seat_lines = []
    for s in SEATS + list(CJ):
        if s in seats and (seats[s]["D"] + seats[s]["R"]) > 0:
            v = seats[s]
            mm = m(v["D"], v["R"])
            mod = rs.loc[s] if s in rs.index else None
            modtxt = f"model final {fmt(mod.m1)} ({mod.p1:.0%})" if mod is not None else ""
            seat_lines.append(f"    {s}: early {fmt(mm)} [{v['rep']}/{v['n']} county pieces in] {modtxt}")
    if seat_lines:
        lines.append("  SEATS (early votes):")
        lines += seat_lines
    return "\n".join(lines), df


def main():
    last_upd, last_sig = None, None
    errors = 0
    log(f"poller start {dt.datetime.now():%H:%M:%S}; benchmarks for {len(B)} counties; watching {len(SEATS)} seats; poll {POLL}s")
    while True:
        try:
            el = get(f"election/{EID}")
            home = json.loads(dec(el["Home"]))
            upd = home.get("LastUpdatedTime")
            if upd != last_upd or ONCE:
                ci_raw = get(f"election/countyInfo/{EID}")
                ci_txt = dec(ci_raw["upload"])
                ci = json.loads(ci_txt)
                stamp = dt.datetime.now().strftime("%H%M%S")
                with gzip.open(OUT / f"countyInfo_{stamp}.json.gz", "wt", encoding="utf-8") as f:
                    f.write(ci_txt)
                (OUT / f"home_{stamp}.json").write_text(json.dumps(home))
                df, seats, names = parse(ci)
                if ONCE:
                    log(f"counties {len(df)}; matched counties in benchmarks {df.index.isin(B.index).sum()}")
                    log("unmatched county names: " + ", ".join(sorted(set(df.index) - set(B.index)))[:400])
                    j = sorted(n for n in names if re.search(r"SUPREME COURT|CRIMINAL APPEALS", n))
                    log(f"judicial race names ({len(j)}): " + "; ".join(j))
                    log("senate-like: " + "; ".join(sorted(n for n in names if "SENATOR" in n and "STATE" not in n)))
                    log(f"seat keys found ({len(seats)}): " + ", ".join(sorted(seats)[:60]))
                    log("missing watched seats: " + ", ".join(s for s in SEATS + list(CJ) if s not in seats))
                    log(f"Home: {home}")
                    return
                text, full = summarize(df, seats, home)
                sig_now = text.split("\n", 1)[1] if "\n" in text else ""
                full.assign(feed_time=upd, polled=stamp).to_csv(OUT / "county_reads.csv", mode="a", header=not (OUT / "county_reads.csv").exists())
                if sig_now != last_sig:
                    log("=== UPDATE ===\n" + text)
                    last_sig = sig_now
                else:
                    log(f"[{dt.datetime.now():%H:%M:%S}] feed updated ({upd}), no change in watched numbers")
                last_upd = upd
            errors = 0
        except Exception as ex:  # keep polling through transient failures
            errors += 1
            if errors in (1, 5) or errors % 20 == 0:
                log(f"POLL ERROR #{errors}: {type(ex).__name__}: {str(ex)[:200]}")
            if ONCE:
                raise
        time.sleep(POLL)


if __name__ == "__main__":
    main()
