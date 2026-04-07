"""
update_polling.py

Automatically fetch race-specific generic ballot data from Decision Desk HQ
and update model_config.py.

DATA SOURCE: Decision Desk HQ (data.ddhq.io JSON API)
===========
DDHQ publishes a public JSON API at data.ddhq.io with average.json endpoints
for each poll question. The generic ballot questions with racial breakdowns are:

  ID   5  — topline generic ballot (LV, all races)
  ID 446  — Generic Congressional Ballot (African American)
  ID 448  — Generic Congressional Ballot (Hispanic)
  ID 452  — Generic Congressional Ballot (White)

No DDHQ question exists for Asian/Other. That group is solved algebraically:
  other_2p = (topline_2p − Σ[known_weight × known_2p]) / other_weight
This constrains the weighted average of all four groups to match the topline.
If the algebraic result is implausible (< 0.45 or > 0.85), it falls back to
a uniform shift from the stored "other" value instead.

UPDATE TRIGGERS
===============
  - Run this script whenever the topline generic ballot shifts ≥ 1pp (default).
  - DDHQ updates their averages daily, so re-runs at any time will reflect
    the latest data.

USAGE
=====
  python src/update_polling.py              # update if ≥ 1pp topline shift
  python src/update_polling.py --force      # always update
  python src/update_polling.py --check      # show current data, don't update
  python src/update_polling.py --dry-run    # show what would change
  python src/update_polling.py --threshold 2.0  # custom trigger (default 1.0pp)
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_CONFIG_PATH = Path(__file__).parent / "model_config.py"

USER_AGENT = "TX-Legislature-Model/1.0 (research; contact via GitHub)"
TIMEOUT = 20

# DDHQ JSON API base
DDHQ_API = "https://data.ddhq.io/polls/v1/production/{qid}/average.json"

# Question IDs — discovered by inspecting embedded JSON in DDHQ page HTML
DDHQ_QUESTIONS = {
    "topline":   5,    # Generic Congressional Ballot (LV, all)
    "black_nh":  446,  # Generic Congressional Ballot (African American)
    "hispanic":  448,  # Generic Congressional Ballot (Hispanic)
    "white_nh":  452,  # Generic Congressional Ballot (White)
    # No DDHQ question for Asian/Other — derived algebraically from others
}

# National demographic weights (from model_config.py — must match)
NATIONAL_DEMO_WEIGHTS = {
    "white_nh": 0.61,
    "black_nh":  0.12,
    "hispanic":  0.15,
    "other":     0.12,
}

# Plausible range for the "other" algebraic solution
OTHER_D_2P_MIN = 0.45
OTHER_D_2P_MAX = 0.85


# ---------------------------------------------------------------------------
# DDHQ API fetch
# ---------------------------------------------------------------------------

def _fetch_ddhq_question(qid: int) -> dict | None:
    """
    Fetch and parse a DDHQ average.json for one poll question.
    Returns {title, d_pct, r_pct, d_2p, date} or None on failure.
    """
    url = DDHQ_API.format(qid=qid)
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT},
                         timeout=TIMEOUT, verify=False)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, json.JSONDecodeError) as exc:
        print(f"    ERROR fetching ID {qid}: {exc}")
        return None

    title = data.get("title", f"ID {qid}")

    # Build cand_id → poll_ag_name map from option_candidate_map
    cand_party = {}   # cand_id (int) → "Democrat" | "Republican"
    for opt_id, cands in data.get("option_candidate_map", {}).items():
        for cand_id_str, cand in cands.items():
            label = cand.get("poll_ag_name", "")
            if "Democrat" in label:
                cand_party[int(cand_id_str)] = "D"
            elif "Republican" in label:
                cand_party[int(cand_id_str)] = "R"

    # Most recent timeseries entry
    ts = data.get("timeseries", [])
    if not ts:
        print(f"    No timeseries data for '{title}'")
        return None

    latest = ts[-1]
    latest_date = latest.get("date", "?")
    d_pct = r_pct = None

    for entry in latest.get("data", []):
        cid = entry.get("cand_id")
        val = entry.get("value")
        if val is None:
            continue
        party = cand_party.get(cid)
        if party == "D":
            d_pct = val
        elif party == "R":
            r_pct = val

    if d_pct is None or r_pct is None:
        print(f"    Could not extract D/R values for '{title}'")
        return None

    denom = d_pct + r_pct
    d_2p = d_pct / denom if denom > 0 else None

    return {
        "title": title,
        "d_pct": d_pct,
        "r_pct": r_pct,
        "d_2p": d_2p,
        "date": latest_date,
        "qid": qid,
    }


def fetch_all_ddhq() -> dict | None:
    """
    Fetch topline + all three DDHQ racial questions.
    Returns dict keyed by model_config race key → d_2p share,
    plus 'topline_d_2p' and 'data_date'.
    """
    print("Fetching from DDHQ JSON API (data.ddhq.io)...")
    results = {}

    for key, qid in DDHQ_QUESTIONS.items():
        q = _fetch_ddhq_question(qid)
        if q is None:
            print(f"  FAILED: {key} (ID {qid})")
            return None
        results[key] = q
        label = key if key != "topline" else "topline (LV)"
        print(f"  {label:12s}  D={q['d_pct']:.1f}%  R={q['r_pct']:.1f}%  "
              f"→  D 2p: {q['d_2p']*100:.1f}%  [{q['date']}]")

    topline_2p   = results["topline"]["d_2p"]
    white_2p     = results["white_nh"]["d_2p"]
    black_2p     = results["black_nh"]["d_2p"]
    hispanic_2p  = results["hispanic"]["d_2p"]

    # Solve for "other" algebraically:
    #   topline_2p = Σ(weight × 2p)
    #   topline_2p = w_wh×white + w_bl×black + w_hi×hisp + w_ot×other
    #   other = (topline_2p - w_wh×white - w_bl×black - w_hi×hisp) / w_ot
    w = NATIONAL_DEMO_WEIGHTS
    other_2p = (
        topline_2p
        - w["white_nh"] * white_2p
        - w["black_nh"] * black_2p
        - w["hispanic"] * hispanic_2p
    ) / w["other"]

    if OTHER_D_2P_MIN <= other_2p <= OTHER_D_2P_MAX:
        print(f"  {'other':12s}                        "
              f"→  D 2p: {other_2p*100:.1f}%  [solved from topline constraint]")
        other_source = "solved"
    else:
        # Implausible result — fall back to stored + uniform shift
        other_2p = None
        print(f"  other: algebraic solution {other_2p} out of plausible range "
              f"[{OTHER_D_2P_MIN}, {OTHER_D_2P_MAX}] — will use uniform shift for this group")
        other_source = "shift"

    return {
        "white_nh":       round(white_2p, 4),
        "black_nh":       round(black_2p, 4),
        "hispanic":       round(hispanic_2p, 4),
        "other":          round(other_2p, 4) if other_2p is not None else None,
        "topline_d_2p":   round(topline_2p, 4),
        "topline_margin": _margin_label(topline_2p),
        "data_date":      results["topline"]["date"],
        "other_source":   other_source,
    }


def _margin_label(d_2p: float) -> str:
    margin_pp = (d_2p - 0.5) * 200
    sign = "D" if margin_pp >= 0 else "R"
    return f"{sign}+{abs(margin_pp):.1f}"


# ---------------------------------------------------------------------------
# Read and update model_config.py
# ---------------------------------------------------------------------------

def read_current_config() -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_config", MODEL_CONFIG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "race_generic":  getattr(mod, "RACE_GENERIC_BALLOT_D_SHARE", {}),
        "source":        getattr(mod, "GENERIC_BALLOT_SOURCE", ""),
        "updated":       getattr(mod, "GENERIC_BALLOT_UPDATED", ""),
        "topline_d_2p":  getattr(mod, "GENERIC_BALLOT_TOPLINE_D_2P", None),
    }


def compute_weighted_topline(race_shares: dict) -> float:
    return sum(NATIONAL_DEMO_WEIGHTS[k] * race_shares.get(k, 0)
               for k in NATIONAL_DEMO_WEIGHTS)


def update_model_config(new_shares: dict, new_topline: float,
                        source: str, margin_label: str,
                        dry_run: bool = False) -> bool:
    config_text = MODEL_CONFIG_PATH.read_text(encoding="utf-8")
    today = str(date.today())

    # Update each racial D share value — only within RACE_GENERIC_BALLOT_D_SHARE block
    # Strategy: replace the entire dict literal so we don't touch NATIONAL_DEMO_WEIGHTS
    def replace_in_race_dict(text: str, key: str, value: float) -> str:
        """Replace a key's value only within the RACE_GENERIC_BALLOT_D_SHARE dict."""
        # Match from RACE_GENERIC_BALLOT_D_SHARE up to its closing }
        # Then within that match, replace the key's value
        block_pat = r'(RACE_GENERIC_BALLOT_D_SHARE\s*:[^=]*=\s*\{[^}]*\})'

        def replace_key_in_block(m):
            block = m.group(1)
            key_pat = rf'("{re.escape(key)}"\s*:\s*)([\d.]+)'
            new_block = re.sub(key_pat, rf'\g<1>{value:.4f}', block)
            return new_block

        new_text = re.sub(block_pat, replace_key_in_block, text, flags=re.DOTALL)
        return new_text

    for key, value in new_shares.items():
        new_text = replace_in_race_dict(config_text, key, value)
        if new_text == config_text:
            print(f"  WARNING: Could not find '{key}' in RACE_GENERIC_BALLOT_D_SHARE")
        config_text = new_text

    # Update metadata strings
    config_text = re.sub(
        r'(GENERIC_BALLOT_SOURCE\s*=\s*)["\'].*?["\']',
        rf'\1"{source}"',
        config_text,
    )
    config_text = re.sub(
        r'(GENERIC_BALLOT_UPDATED\s*=\s*)["\'].*?["\']',
        rf'\1"{today}"',
        config_text,
    )

    # Update or insert GENERIC_BALLOT_TOPLINE_D_2P
    topline_val = round(new_topline, 4)
    topline_pattern = r'(GENERIC_BALLOT_TOPLINE_D_2P\s*:\s*float\s*=\s*)[\d.]+'
    if re.search(topline_pattern, config_text):
        config_text = re.sub(topline_pattern, rf'\g<1>{topline_val}', config_text)
    else:
        insert_after = r'(GENERIC_BALLOT_UPDATED\s*=\s*"[^"]*"\s*\n)'
        config_text = re.sub(
            insert_after,
            rf'\1\nGENERIC_BALLOT_TOPLINE_D_2P: float = {topline_val}  # {margin_label}\n',
            config_text, count=1,
        )

    # Update "Topline at time of update" comment if present
    config_text = re.sub(
        r'(#\s*Topline at time of update:).*',
        rf'\1 {margin_label} ({today})',
        config_text,
    )

    if dry_run:
        print("\n  [DRY RUN] Changes that would be written to model_config.py:")
        print(f"    GENERIC_BALLOT_TOPLINE_D_2P = {topline_val}  # {margin_label}")
        print(f"    GENERIC_BALLOT_UPDATED      = '{today}'")
        print(f"    GENERIC_BALLOT_SOURCE       = '{source}'")
        for k, v in new_shares.items():
            print(f"    RACE_GENERIC_BALLOT_D_SHARE['{k}'] = {v:.4f}")
        return False

    MODEL_CONFIG_PATH.write_text(config_text, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch DDHQ generic ballot by race and update model_config.py"
    )
    parser.add_argument("--force", action="store_true",
                        help="Update even if topline shift < threshold")
    parser.add_argument("--check", action="store_true",
                        help="Print current config only; don't fetch or update")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch data and show proposed changes; don't save")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Topline shift (pp) to trigger update (default 1.0)")
    args = parser.parse_args()

    print("=" * 60)
    print("TX LEGISLATURE MODEL — POLLING UPDATE")
    print("Source: Decision Desk HQ (data.ddhq.io)")
    print("=" * 60)

    # Read current stored values
    print("\nCurrent model_config.py:")
    current = read_current_config()
    old_shares   = current["race_generic"]
    old_topline  = current.get("topline_d_2p")

    if old_topline is None:
        old_topline = compute_weighted_topline(old_shares)
        print("  (GENERIC_BALLOT_TOPLINE_D_2P not yet in config — computed from racial shares)")

    print(f"  Stored topline:  {old_topline*100:.2f}% D 2p  ({_margin_label(old_topline)})")
    print(f"  Last updated:    {current['updated']}")
    print(f"  Source:          {current['source']}")
    print(f"  Racial D shares: "
          f"white={old_shares.get('white_nh',0):.3f}  "
          f"black={old_shares.get('black_nh',0):.3f}  "
          f"hisp={old_shares.get('hispanic',0):.3f}  "
          f"other={old_shares.get('other',0):.3f}")

    if args.check:
        print("\n(--check mode: done)")
        return

    # Fetch from DDHQ
    print(f"\n{'─'*40}")
    new_data = fetch_all_ddhq()
    if new_data is None:
        print("\nFetch failed. Check your internet connection.")
        return

    new_topline = new_data["topline_d_2p"]
    shift_pp    = (new_topline - old_topline) * 100
    shift_abs   = abs(shift_pp)
    shift_dir   = "D" if shift_pp >= 0 else "R"

    print(f"\n{'─'*40}")
    print(f"Topline shift: {shift_dir}+{shift_abs:.2f}pp  "
          f"({old_topline*100:.2f}% → {new_topline*100:.2f}%)")

    if not args.force and shift_abs < args.threshold:
        print(f"Shift ({shift_abs:.2f}pp) < threshold ({args.threshold:.1f}pp). No update needed.")
        print("Run with --force to update anyway, or --threshold to lower the trigger.")
        return

    # Build final shares — handle missing "other" with uniform shift fallback
    new_shares = {
        "white_nh": new_data["white_nh"],
        "black_nh": new_data["black_nh"],
        "hispanic": new_data["hispanic"],
    }

    if new_data["other"] is not None:
        new_shares["other"] = new_data["other"]
        other_note = "solved from topline constraint"
    else:
        # Uniform shift for "other"
        delta = new_topline - old_topline
        other_new = round(max(0.0, min(1.0, old_shares.get("other", 0.57) + delta)), 4)
        new_shares["other"] = other_new
        other_note = f"uniform shift ({delta:+.4f}) from stored value"

    # Summary
    print(f"\n{'─'*40}")
    print("Final race-specific D 2p shares:")
    print(f"  {'Group':12s}  {'Old':>7}  {'New':>7}  {'Change':>8}  Notes")
    for k in ["white_nh", "black_nh", "hispanic", "other"]:
        old_v = old_shares.get(k, 0)
        new_v = new_shares[k]
        delta = new_v - old_v
        notes = other_note if k == "other" else "DDHQ"
        print(f"  {k:12s}  {old_v:.4f}  {new_v:.4f}  {delta:+.4f}  {notes}")

    implied_avg = compute_weighted_topline(new_shares)
    print(f"\n  Weighted-avg check: {implied_avg*100:.2f}% D 2p  "
          f"(DDHQ topline: {new_topline*100:.2f}%)")

    source_str = (
        f"DDHQ data.ddhq.io ({new_data['data_date']}); "
        f"white/black/hispanic from IDs 452/446/448; other {other_note}"
    )

    # Update model_config.py
    print(f"\n{'─'*40}")
    if args.dry_run:
        update_model_config(new_shares, new_topline, source_str,
                            new_data["topline_margin"], dry_run=True)
    else:
        ok = update_model_config(new_shares, new_topline, source_str,
                                  new_data["topline_margin"])
        if ok:
            print("model_config.py updated.")
            print("Re-run projections:  python src/model.py")


if __name__ == "__main__":
    main()
