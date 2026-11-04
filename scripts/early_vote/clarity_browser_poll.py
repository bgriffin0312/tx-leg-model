"""Election night: read Clarity county results through a real (headless) browser.

Clarity returns 502 to plain scripted requests tonight but serves a browser. Every few minutes this opens a
Clarity page, fetches each county's detailxml.zip from inside it, and saves:
  - clarity_<COUNTY>_<ver>.json   partisan contests as {contest: {DEM: [early, eday, total], REP: [...]}}
                                  (the format county_poll.py writes, read by the projection and seat report)
  - local_<COUNTY>_latest.json    every contest's choices with early / Election Day / total, for local races
    python -u scripts/early_vote/clarity_browser_poll.py [--once]
"""
import base64
import datetime as dt
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
EN = ROOT / "data" / "raw" / "election_night"
COUNTIES = {"TRAVIS": "Travis/127492", "TARRANT": "Tarrant/127297", "DALLAS": "Dallas/127396", "BEXAR": "Bexar/127291",
            "DENTON": "Denton/127398", "WILLIAMSON": "Williamson/127488", "EL PASO": "El_Paso/127348", "FORT BEND": "Fort_Bend/127480"}
EVERY = 180
ONCE = "--once" in sys.argv
JS = """async (path) => {
  const vr = await fetch('/TX/' + path + '/current_ver.txt', {cache: 'no-store'});
  const v = (await vr.text()).trim();
  if (!/^[0-9]+$/.test(v)) return {err: 'ver ' + vr.status + ' ' + v.slice(0, 60)};
  const r = await fetch('/TX/' + path + '/' + v + '/reports/detailxml.zip', {cache: 'no-store'});
  if (!r.ok) return {err: 'zip ' + r.status, ver: v};
  const buf = new Uint8Array(await r.arrayBuffer());
  let s = '';
  for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
  return {ver: v, b64: btoa(s)};
}"""


def log(s):
    print(s, flush=True)


def parse(xml):
    ts = (re.findall(r"<Timestamp>([^<]*)</Timestamp>", xml) or [""])[0]
    try:
        all_early = dt.datetime.strptime(ts.replace(" CST", "").strip(), "%m/%d/%Y %I:%M:%S %p") < dt.datetime(2026, 11, 3, 20, 0)
    except ValueError:
        all_early = False
    races, local = {}, {}
    for c in re.finditer(r'<Contest key="\d+" text="([^"]+)"([^>]*)>(.*?)</Contest>', xml, re.S):
        name, attrs, body = c.group(1), c.group(2), c.group(3)
        pr = re.search(r'precinctsReported="(\d+)"', attrs)
        tp = re.search(r'precinctsParticipating="(\d+)"', attrs)
        part = {"DEM": [0, 0, 0], "REP": [0, 0, 0]}
        choices = []
        for ch in re.finditer(r'<Choice key="\d+" text="([^"]*)" party="([^"]*)" totalVotes="(\d+)">(.*?)</Choice>', body, re.S):
            early = eday = 0
            for vt, n in re.findall(r'<VoteType name="([^"]+)" votes="(\d+)"', ch.group(4)):
                if re.search(r"overvote|undervote", vt, re.I):
                    continue
                if re.match(r"election", vt, re.I) and not all_early:
                    eday += int(n)
                else:
                    early += int(n)
            p = ch.group(2).upper()
            if p in part:
                part[p][0] += early; part[p][1] += eday; part[p][2] += int(ch.group(3))
            choices.append({"name": ch.group(1), "party": p, "total": int(ch.group(3)), "early": early, "eday": eday})
        if part["DEM"][2] or part["REP"][2]:
            races[name] = part
        local[name] = {"choices": choices, "pr": int(pr.group(1)) if pr else None, "tp": int(tp.group(1)) if tp else None}
    return ts, races, local


def cycle(page, last):
    for county, path in COUNTIES.items():
        try:
            res = page.evaluate(JS, path)
        except Exception as ex:
            log(f"CLARITYBR {county} error {type(ex).__name__}: {str(ex)[:120]}")
            continue
        if res.get("err"):
            log(f"CLARITYBR {county} {res['err']}")
            continue
        if last.get(county) == res["ver"]:
            continue
        xml = zipfile.ZipFile(io.BytesIO(base64.b64decode(res["b64"]))).read("detail.xml").decode("utf-8", "replace")
        ts, races, local = parse(xml)
        key = county.replace(" ", "_")
        if county == "BEXAR" and (EN / "clarity_BEXAR_383808.json").exists() and res["ver"] != "383808":
            # Bexar files early in-person votes under "Election Day". Its 7:36pm version (383808) was early votes
            # only, so Election Day = current total minus that early total, whatever the labels say.
            base = json.loads((EN / "clarity_BEXAR_383808.json").read_text())
            for n, v in races.items():
                b = base.get(n)
                for p in ("DEM", "REP"):
                    early = b[p][0] if b else v[p][0]
                    v[p][0], v[p][1] = early, max(v[p][2] - early, 0)
        (EN / f"clarity_{key}_{res['ver']}.json").write_text(json.dumps(races))
        (EN / f"local_{key}_latest.json").write_text(json.dumps({"county": county, "ver": res["ver"], "timestamp": ts, "contests": local}))
        last[county] = res["ver"]
        eday = sum(v["DEM"][1] + v["REP"][1] for n, v in races.items() if re.search(r"Supreme Court|Criminal Appeals", n, re.I))
        log(f"CLARITYBR {county} new version {res['ver']} [{ts}] judicial Election Day votes {eday:,}")


def main():
    last = {}
    # 11:37pm: the browser is launched and closed every cycle so it holds no memory between fetches (the laptop
    # ran low on RAM at 11pm with a long-lived browser)
    while True:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                ctx = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36")
                page = ctx.new_page()
                page.goto("https://results.enr.clarityelections.com/TX/Travis/", wait_until="domcontentloaded", timeout=60000)
                cycle(page, last)
            except Exception as ex:
                log(f"CLARITYBR cycle error {type(ex).__name__}: {str(ex)[:150]}")
            finally:
                browser.close()
        if ONCE:
            break
        time.sleep(EVERY)


if __name__ == "__main__":
    main()
