"""
whole_election_model.py

One electorate, every race. Each simulation draws a single Texas party vote and one local
electorate for every county and House district; every race on the ballot is read from those
same voters. Two races differ in a run only because:

  - they cover different places (same draws, different weights),
  - some voters stop before the race (incomplete ballots), or
  - some voters cross over in it (ticket splitters).

    margin = T + place + incomplete ballots + splitters

Race means come from the existing layers: the statewide + congressional model and its poll
conditioning, the legislative formula, the county model. What is new is the error structure
and the one T. A race's gap to T is held fixed, so the two versions differ only in where T
centers:

    statewide     R+4.7, the statewide posterior
    legislative   R+7.2, what D+9.2 implies in the legislative formula

  python scripts/whole_election_estimates.py        # the measured rates this reads
  python src/whole_election_model.py                # both versions
  python src/whole_election_model.py --query "us_senate & !tarrant" --env statewide
  python src/whole_election_model.py --query "tarrant" --given "us_senate" --env legislative
  python src/whole_election_model.py --query "house>=70 & seats>=12"

Names: us_senate, governor, ... ; tx1..tx38; hd1..hd150; sd1.. (the 16 seats up); tarrant,
williamson, collin, denton, fort_bend, harris, bexar, montgomery, el_paso, bexar_da. Each
means the Democrat wins; ! means the Democrat loses. house, senate and seats count D wins in
the Texas House, the Texas Senate (holdovers included) and the U.S. House delegation.

Output: output/whole_election_<env>_2026.csv, .npz; output/whole_election_summary_2026.csv
"""
import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "output"

T_LEGISLATIVE = -7.2      # model docs: D+9.2 in the differenced legislative formula
ROLL_SD = 0.35            # cycle-to-cycle spread in the D-R drop-off gap, 2022 vs 2024; two post-straight-ticket cycles only
LAYER_COUNTY = 1.7        # county model's cycle miss, margin
TOP_SLOTS = {"us_senate", "governor"}


def electorate():
    """Weights from each race to the county and House district cells that vote in it, from
    2024 presidential votes on the VTDs. PlanC2333 uses the exact TLC assignment."""
    import statewide_results as sr
    d24, d22 = sr.load(2024), sr.load(2022)
    top = d24[d24.Office == "President"].groupby(["cntyvtd", "County"]).Votes.sum().reset_index(name="v")
    top["county"] = top.County.str.lower().str.strip()
    # TLC's precinct-to-district file, not the race rows: an uncontested race has no VTD rows,
    # which left 54 House districts with no place draw
    pm = pd.read_excel(ROOT / "data" / "raw" / "_capitol_data_cache" / "precincts24g_districts.xlsx", dtype=str).set_index("PCTKEY")
    hd = d24[d24.Office.str.match(r"^State Rep \d+$")].groupby("cntyvtd").Office.first().str.extract(r"(\d+)$")[0].astype(int)
    top["hd"] = top.cntyvtd.astype(str).map(pm.PlanH2316.astype(float)).fillna(top.cntyvtd.map(hd))
    sd = d22[d22.Office.str.match(r"^State Sen \d+$")].groupby("cntyvtd").Office.first().str.extract(r"(\d+)$")[0].astype(int)
    top["sd"] = top.cntyvtd.astype(str).map(pm.PlanS2168.astype(float)).fillna(top.cntyvtd.map(sd))
    counties = sorted(top.county.unique())
    hds = list(range(1, 151))
    ci = {c: i for i, c in enumerate(counties)}
    hi = {h: i for i, h in enumerate(hds)}

    def weights(frame, key, labels):
        wc = np.zeros((len(labels), len(counties)))
        wh = np.zeros((len(labels), len(hds)))
        idx = {x: i for i, x in enumerate(labels)}
        for row in frame.itertuples():
            i = idx.get(getattr(row, key))
            if i is None:
                continue
            wc[i, ci[row.county]] += row.v
            if row.hd == row.hd:
                wh[i, hi[int(row.hd)]] += row.v
        return wc, wh

    pieces = sr.vtd_assignment(d24).merge(top, on="cntyvtd")
    pieces["v"] = pieces.v * pieces.w
    return {"top": top, "counties": counties, "hds": hds, "pieces": pieces, "weights": weights}


def unit(w):
    """L2-normalize, so a race's local draw has unit variance whatever its geography."""
    n = np.sqrt((w ** 2).sum(1, keepdims=True))
    return np.divide(w, n, out=np.zeros_like(w), where=n > 0)


def rolloff_table():
    """Drop-off effect on the margin relative to the judicial vote T is measured on, from
    2022, the one midterm without straight-ticket voting. Top of the ticket drops nobody."""
    r = pd.read_csv(OUT / "whole_election_rolloff.csv")
    x = r[r.year == 2022]
    j = x[x.tier == "judicial"].rolloff_effect.mean()
    t = {k: float(g.rolloff_effect.mean() - j) for k, g in x.groupby("tier")}
    by = {o: float(v - j) for o, v in zip(x.office, x.rolloff_effect)}
    out = {"j": float(j), "tier": t,
           "statewide": {"us_senate": -j, "governor": -j, "lt_governor": by["Lt. Governor"], "attorney_general": by["Attorney Gen"],
                         "comptroller": by["Comptroller"], "land_commissioner": by["Land Comm"], "ag_commissioner": by["Ag Comm"],
                         "railroad_commissioner": by["RR Comm 1"]}}
    return out


def load_layers():
    import statewide_pollster_history  # noqa: F401  (statewide_model reads what it writes)
    import statewide_model as sw
    import county_judge_model as cj
    import model as lg
    import model_config as mc
    with contextlib.redirect_stdout(io.StringIO()):
        b = sw.build()
        f = sw.forecast(*(b[k] for k in ("R", "C", "dp", "Tmu", "Tsd", "c_sd", "h_sd", "e_sd", "sds")), sw.POLL_B_MEAN, quiet=True)
        c = cj.main()
        df = lg.load_districts()
        linear = lg.build_linear_predictions(df, lg.CURRENT_ENV, lg.RACE_GENERIC_BALLOT_D_SHARE).values
    return {"f": f, "cj": cj, "c": c, "lg": lg, "mc": mc, "df": df.reset_index(drop=True), "linear": np.asarray(linear, float)}


class Whole:
    def __init__(self, seed=7):
        self.est = json.loads((OUT / "whole_election_estimates.json").read_text())
        self.ro = rolloff_table()
        self.L = load_layers()
        self.E = electorate()
        f = self.L["f"]
        self.n = len(f["T"])
        self.Tc = float(f["T"].mean())
        rng = np.random.default_rng(seed)
        self.Uc = rng.standard_normal((self.n, len(self.E["counties"])), dtype=np.float32)
        self.Uh = rng.standard_normal((self.n, len(self.E["hds"])), dtype=np.float32)
        self.kappa = rng.normal(0, ROLL_SD, self.n).astype(np.float32)   # D-R drop-off gap this cycle, relative to 2022
        self.layer = {k: rng.normal(0, s, self.n).astype(np.float32) for k, s in self.est["gap_tier_sd"].items()}
        self.layer["county"] = rng.normal(0, LAYER_COUNTY, self.n).astype(np.float32)
        self.layer["state_house"] = rng.normal(0, self.est["competitive_sd"], self.n).astype(np.float32)
        self.layer["state_senate"] = self.layer["state_house"]
        self.eps = rng.standard_normal((self.n, 200), dtype=np.float32)
        self.groups = rng.standard_normal((self.n, 5), dtype=np.float32)
        self.seed = seed
        # place share of a race's own error: county-wide, then the rest of the local share
        self.lc = self.est["local_share"] * self.est["county_icc"]
        self.lh = self.est["local_share"] * (1 - self.est["county_icc"])
        self.beta = self.est["slope"]
        self._geo()

    def _geo(self):
        E = self.E
        wf = E["weights"]
        top, pieces = E["top"], E["pieces"]
        df = self.L["df"]
        house = df.chamber_lower == "house"
        self.h_ids = df.district[house].astype(int).tolist()
        self.s_ids = df.district[~house].astype(int).tolist()
        self.is_house = house.values
        hc, hh = wf(top, "hd", self.h_ids)
        sc, sh = wf(top, "sd", self.s_ids)
        cc, ch = wf(pieces, "dist", list(self.L["f"]["C"].index))
        self.leg_c = np.zeros((len(df), hc.shape[1]))
        self.leg_h = np.zeros((len(df), hh.shape[1]))
        self.leg_c[house.values], self.leg_h[house.values] = unit(hc), unit(hh)
        self.leg_c[~house.values], self.leg_h[~house.values] = unit(sc), unit(sh)
        self.cd_c, self.cd_h = unit(cc), unit(ch)
        names = list(self.L["c"]["r"].county.unique())
        wc, wh = wf(top, "county", names)
        self.cj_names = names
        self.cj_c, self.cj_h = unit(wc), unit(wh)
        # Hispanic exposure of a county race: its House districts' centered Hispanic share
        groups = self.L["lg"]._GROUPS_EDU if self.L["mc"].GROUP_ERROR_EDUCATION else self.L["lg"]._GROUPS
        self.group_names = groups
        cen = self.L["lg"].centered_group_shares(df, self.L["mc"].GROUP_ERROR_EDUCATION)
        self.cen = cen
        hi = cen[:, list(groups).index("hispanic")]
        byhd = pd.Series(hi[house.values] * 10, index=self.h_ids).reindex(E["hds"]).fillna(0).values
        share = wh / wh.sum(1, keepdims=True)
        self.cj_ss = share @ byhd

    def local(self, wc, wh):
        return np.sqrt(self.lc) * (self.Uc @ wc.T.astype(np.float32)) + np.sqrt(self.lh) * (self.Uh @ wh.T.astype(np.float32))

    def run(self, env, layer_extra=0.0):
        L, f = self.L, self.L["f"]
        center = self.Tc if env == "statewide" else T_LEGISLATIVE
        T = f["T"] - self.Tc + center                    # one party vote per run
        h = f["h"]
        hd = h - h.mean()
        ro = self.ro
        out, marg = {}, {}

        # statewide: T + the race's own gap, as conditioned on its polls
        R = f["R"]
        dev = f["sw_y"] - f["T"][:, None]
        names = list(R.index)
        roll = np.array([ro["statewide"][k] for k in names])
        sw = T[:, None] + dev + self.kappa[:, None] * roll
        for i, k in enumerate(names):
            marg[k] = sw[:, i]

        # U.S. House: T + c + Hispanic slope + district, with the place share made common
        C = f["C"]
        rest = f["cd_y"] - f["T"][:, None] - f["c"][:, None] - f["h"][:, None] * C.ss.values[None, :]
        mu = rest.mean(0)
        sd = rest.std(0)
        delta = (rest - mu) * np.sqrt(1 - self.lc - self.lh)
        loc = self.local(self.cd_c, self.cd_h)
        cd = T[:, None] + f["c"][:, None] + f["h"][:, None] * C.ss.values[None, :] + mu + delta + loc * sd + self.kappa[:, None] * ro["tier"]["us_house"]
        for j, d in enumerate(C.index):
            marg[f"tx{d}"] = cd[:, j]

        # Legislature: the shipped formula, read relative to its own T center
        lg, mc = L["lg"], L["mc"]
        df = L["df"]
        lin = L["linear"] * 200 - 100
        sig = np.array([mc.GROUP_ERROR_SIGMA[g] for g in self.group_names]) * 200
        gi = list(self.group_names).index("hispanic")
        ge = self.groups[:, :len(sig)] * sig.astype(np.float32)
        ge[:, gi] = hd.astype(np.float32) * 10   # h is margin per 10 pts, cen is a share
        gerr = ge @ self.cen.T.astype(np.float32)
        ig = float(lg.SIGMA_IDIO ** 2 - ((self.cen ** 2) @ (sig / 200) ** 2).mean())
        idio_sd = np.sqrt(max(ig, 0.0)) * 200
        rng = np.random.default_rng(self.seed + 1)
        eps = rng.standard_normal((self.n, len(df)), dtype=np.float32)
        lloc = self.local(self.leg_c, self.leg_h)
        idio = idio_sd * (np.sqrt(1 - self.lc - self.lh) * eps + lloc)
        lay = np.where(self.is_house[None, :], self.layer["state_house"][:, None], self.layer["state_senate"][:, None])
        extra = 0.0
        if layer_extra:
            extra = np.random.default_rng(self.seed + 2).normal(0, layer_extra, self.n)[:, None]
        rl = np.where(self.is_house, ro["tier"]["state_house"], ro["tier"]["state_senate"])
        leg = lin + self.beta * (T - T_LEGISLATIVE)[:, None] + lay + extra + gerr + idio + self.kappa[:, None] * rl
        forced = df.forced_d_win.values if "forced_d_win" in df else np.full(len(df), np.nan)
        lw = leg > 0
        lw[:, forced == 1.0] = True
        lw[:, forced == 0.0] = False
        for k, (ch, d) in enumerate(zip(df.chamber_lower, df.district)):
            nm = f"hd{int(d)}" if ch == "house" else f"sd{int(d)}"
            marg[nm] = leg[:, k]
        self.leg_forced = forced

        # County races, same draws; their formula also centers on the legislative T
        c = L["c"]
        shift = self.beta * (center - T_LEGISLATIVE)
        shared = (self.beta * (T - T.mean()) + self.layer["county"]) / 200
        loc = self.local(self.cj_c, self.cj_h)
        local = {k: (loc[:, i], self.lc + self.lh, hd * self.cj_ss[i]) for i, k in enumerate(self.cj_names)}
        keep, km = {}, {}
        rr = c["r"]
        with contextlib.redirect_stdout(io.StringIO()):
            o = L["cj"].run(rr, c["k_open"], c["k_held"], float(shared.std()), c["within"], shift=shift, shared_draws=shared,
                            seed=self.seed + 3, keep=keep, keep_margin=km, local=local)
        cwins, cnames = [], []
        for (county, office), w in keep.items():
            nm = "bexar_da" if office == "district_attorney" else county.replace(" ", "_")
            cnames.append(nm)
            cwins.append(w)
            marg[nm] = km[(county, office)] + self.kappa * ro["tier"]["county_judge"]
        names_all = list(marg)
        M = np.column_stack([marg[k] for k in names_all]).astype(np.float32)
        W = M > 0
        idx = {k: i for i, k in enumerate(names_all)}
        for k, (ch, d) in enumerate(zip(df.chamber_lower, df.district)):
            nm = f"hd{int(d)}" if ch == "house" else f"sd{int(d)}"
            if forced[k] == forced[k]:
                W[:, idx[nm]] = forced[k] == 1.0
        for nm, w in zip(cnames, cwins):
            W[:, idx[nm]] = w
        hold = lg._SENATE_D_HOLDOVER
        house_cols = [idx[f"hd{d}"] for d in self.h_ids]
        sen_cols = [idx[f"sd{d}"] for d in self.s_ids]
        cd_cols = [idx[f"tx{d}"] for d in C.index]
        return {"T": T, "names": names_all, "M": M, "W": W, "house": W[:, house_cols].sum(1), "senate": W[:, sen_cols].sum(1) + hold,
                "seats": W[:, cd_cols].sum(1), "kappa": self.kappa, "hold": hold}


def race_table(res, wh, env):
    ro = wh.ro
    T = float(res["T"].mean())
    rows = []
    for i, k in enumerate(res["names"]):
        m = float(res["M"][:, i].mean())
        if k in ro["statewide"]:
            kind, r = "statewide", ro["statewide"][k]
        elif k.startswith("tx"):
            kind, r = "us_house", ro["tier"]["us_house"]
        elif k.startswith("hd"):
            kind, r = "state_house", ro["tier"]["state_house"]
        elif re.fullmatch(r"sd\d+", k):
            kind, r = "state_senate", ro["tier"]["state_senate"]
        else:
            kind, r = "county", ro["tier"]["county_judge"]
        rows.append({"env": env, "race": k, "kind": kind, "mean": m, "sd": float(res["M"][:, i].std()), "p_d": float(res["W"][:, i].mean()),
                     "T": T, "gap_to_T": m - T, "incomplete_ballots": r, "splitters_and_place": m - T - r})
    return pd.DataFrame(rows)


def p(res, *names, neg=()):
    idx = {k: i for i, k in enumerate(res["names"])}
    x = np.ones(len(res["T"]), bool)
    for n in names:
        x &= res["W"][:, idx[n]]
    for n in neg:
        x &= ~res["W"][:, idx[n]]
    return float(x.mean())


def save(res, env):
    np.savez_compressed(OUT / f"whole_election_{env}_2026.npz", wins=res["W"], names=np.array(res["names"]),
                        house=res["house"].astype(np.int16), senate=res["senate"].astype(np.int8), seats=res["seats"].astype(np.int8))


def mask(expr, W, names, counts):
    idx = {n: i for i, n in enumerate(names)}
    out = np.ones(len(W), bool)
    for tok in re.split(r"\s*&\s*", expr.strip()):
        s = re.fullmatch(r"(seats|house|senate)\s*(>=|<=|==)\s*(\d+)", tok)
        if s:
            x, v = counts[s.group(1)], int(s.group(3))
            out &= {">=": x >= v, "<=": x <= v, "==": x == v}[s.group(2)]
            continue
        neg = tok.startswith("!")
        name = tok.lstrip("!").strip()
        if name not in idx:
            sys.exit(f"unknown name {name}")
        out &= ~W[:, idx[name]] if neg else W[:, idx[name]]
    return out


def query(a):
    for env in ([a.env] if a.env != "both" else ["statewide", "legislative"]):
        d = np.load(OUT / f"whole_election_{env}_2026.npz")
        W, names = d["wins"], [str(x) for x in d["names"]]
        counts = {k: d[k] for k in ("seats", "house", "senate")}
        g = mask(a.given, W, names, counts) if a.given else np.ones(len(W), bool)
        for e in a.query:
            m = mask(e, W, names, counts)
            tail = f" given {a.given}" if a.given else ""
            print(f"[{env}] P({e}{tail}) = {m[g].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", nargs="*")
    ap.add_argument("--given")
    ap.add_argument("--env", default="both", choices=["both", "statewide", "legislative"])
    a = ap.parse_args()
    if a.query:
        query(a)
        return
    wh = Whole()
    print(f"T center {wh.Tc:.2f}; sd {wh.L['f']['T'].std():.2f}; slope {wh.beta:.2f}; place share of race error: county {wh.lc:.3f}, district {wh.lh:.3f}")
    print("incomplete ballots relative to T (2022):", {k: round(v, 2) for k, v in wh.ro["tier"].items()}, {k: round(v, 2) for k, v in wh.ro["statewide"].items()})
    tabs, summ = [], []
    KEY = ["us_senate", "governor", "lt_governor", "attorney_general", "comptroller", "land_commissioner", "ag_commissioner", "railroad_commissioner",
           "tx15", "tx23", "tx28", "tx34", "tx35", "tarrant", "williamson", "collin", "denton", "fort_bend", "harris", "bexar", "bexar_da", "el_paso", "montgomery"]
    for env in ("statewide", "legislative"):
        res = wh.run(env)
        save(res, env)
        t = race_table(res, wh, env)
        tabs.append(t)
        t.to_csv(OUT / f"whole_election_{env}_2026.csv", index=False)
        print(f"\n===== {env} environment: T {res['T'].mean():+.1f} =====")
        print(t[t.race.isin(KEY)].set_index("race").loc[KEY, ["mean", "sd", "p_d", "gap_to_T", "incomplete_ballots", "splitters_and_place"]].round(2).to_string())
        h, s, cd = res["house"], res["senate"], res["seats"]
        print(f"TX House D {h.mean():.1f} (10-90 {np.percentile(h, 10):.0f}-{np.percentile(h, 90):.0f}), P(76+) {(h >= 76).mean():.3f}; "
              f"TX Senate D {s.mean():.1f}, P(16+) {(s >= 16).mean():.3f}; U.S. House D {cd.mean():.1f} (10-90 {np.percentile(cd, 10):.0f}-{np.percentile(cd, 90):.0f})")
        rec = {"env": env, "T": float(res["T"].mean()), "house": float(h.mean()), "house_76": float((h >= 76).mean()), "senate": float(s.mean()), "senate_16": float((s >= 16).mean()), "us_house": float(cd.mean()),
               "talarico_simmons": p(res, "us_senate", "tarrant"), "talarico_no_simmons": p(res, "us_senate", neg=["tarrant"]), "paxton_simmons": p(res, "tarrant", neg=["us_senate"]),
               "both_lose": p(res, neg=["us_senate", "tarrant"]), "talarico_abbott": p(res, "us_senate", neg=["governor"]), "paxton_hinojosa": p(res, "governor", neg=["us_senate"]),
               "sen_gov": p(res, "us_senate", "governor"), "tarrant_williamson": p(res, "tarrant", "williamson"), "any_statewide_d": float(res["W"][:, :8].any(1).mean())}
        idx = {k: i for i, k in enumerate(res["names"])}
        for a_, b_ in (("us_senate", "governor"), ("us_senate", "tarrant"), ("us_senate", "tx35"), ("tarrant", "tx12"), ("tarrant", "hd92"), ("tx35", "hd118")):
            if a_ in idx and b_ in idx:
                x, y = res["M"][:, idx[a_]], res["M"][:, idx[b_]]
                rec[f"corr_{a_}_{b_}"] = float(np.corrcoef(x, y)[0, 1])
        gap = res["M"][:, idx["us_senate"]] - res["M"][:, idx["governor"]]
        rec["sen_minus_gov"] = float(gap.mean())
        rec["net_crossover_pct"] = float(gap.mean() / 2)
        summ.append(rec)
        print(pd.Series({k: v for k, v in rec.items() if k != "env"}).round(3).to_string())
    pd.DataFrame(summ).to_csv(OUT / "whole_election_summary_2026.csv", index=False)
    pd.concat(tabs).to_csv(OUT / "whole_election_races_2026.csv", index=False)
    # the legislative shared error as shipped, as a sensitivity: 6.7 margin in total
    lg = wh.L["lg"]
    want = lg.SIGMA_NATIONAL * 200
    extra = np.sqrt(max(want ** 2 - wh.L["f"]["T"].std() ** 2 * wh.beta ** 2 - wh.est["competitive_sd"] ** 2, 0.0))
    for env in ("statewide", "legislative"):
        r = wh.run(env, layer_extra=extra)
        print(f"sensitivity {env}: legislative shared error at the shipped {want:.1f}: House {r['house'].mean():.1f}, P(76+) {(r['house'] >= 76).mean():.3f}, Senate P(16+) {(r['senate'] >= 16).mean():.3f}")


if __name__ == "__main__":
    main()
