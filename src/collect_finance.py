"""
collect_finance.py

Collect Texas legislative candidate campaign finance totals (total raised)
by district for historical off-year cycles: 2002, 2006, 2010, 2014, 2018, 2022.

Strategy (two-tier fallback):

  Tier 1 — Texas Ethics Commission (TEC) bulk data
    TEC publishes annual totals summary ZIP files at:
      https://www.ethics.state.tx.us/data/search/cf/{year}tottab.zip
    Each ZIP contains a pipe-delimited text file (TEC_CF_CSV format) with one
    row per candidate filing. We filter for state legislative offices, aggregate
    total contributions per candidate, then match to election districts.

  Tier 2 — FollowTheMoney.org API
    If TEC download fails, and FOLLOWTHEMONEY_API_KEY is set in .env, use the
    FollowTheMoney API to pull candidate-level totals.
    API endpoint: https://api.followthemoney.org/
    Register for a free key at: https://www.followthemoney.org/

  Tier 3 — Placeholder
    If both fail, write placeholder CSVs with MANUAL_NEEDED=True and document
    exactly what fields need filling.

OUTPUT: data/raw/historical/tx_finance_{year}.csv
Columns:
  year, chamber, district, dem_candidate, rep_candidate,
  dem_raised, rep_raised, dem_fundraising_share, log_challenger_fundraising,
  challenger_viability_flag, incumbent_fundraising, name_match_confidence,
  data_source, MANUAL_NEEDED

VIABILITY THRESHOLDS (can be adjusted in build_phase1_dataset.py):
  House challenger: > $100,000
  Senate challenger: > $250,000

MANUAL DATA ENTRY NOTES:
  If automated collection fails, the best manual sources are:
  1. TEC search: https://www.ethics.state.tx.us/search/cf/
     - Search by filer type "Legislative Candidate"
     - Filter by election year and office
  2. FollowTheMoney bulk download: https://www.followthemoney.org/bulk-data/
     - Select Texas, state legislative, the relevant year
  3. For each MANUAL_NEEDED row, fill in dem_raised and rep_raised (total
     contributions received through the general election), then set
     MANUAL_NEEDED=False and recalculate the derived columns.

Usage:
  python src/collect_finance.py
  python src/collect_finance.py --year 2022
  python src/collect_finance.py --no-tec --no-ftm  # placeholder only
"""

import argparse
import csv
import io
import math
import os
import re
import struct
import sys
import time
import zlib
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_HIST = DATA_RAW / "historical"
DATA_HIST.mkdir(parents=True, exist_ok=True)
FINANCE_CACHE = DATA_HIST / "_finance_cache"
FINANCE_CACHE.mkdir(exist_ok=True)

USER_AGENT = "TXLegislativeModel/1.0 (academic research; non-commercial)"

# TEC master ZIP — one file containing all TX campaign finance data since 2000
TEC_ZIP_URL = "https://prd.tecprd.ethicsefile.com/public/cf/public/TEC_CF_CSV.zip"
TEC_ENCODING = "utf-8"  # TEC CSV files are UTF-8

# FollowTheMoney API
FTM_API_URL = "https://api.followthemoney.org/"
FTM_API_KEY = os.getenv("FOLLOWTHEMONEY_API_KEY", "")

# Office keywords for filtering TEC filings
HOUSE_KEYWORDS = ["state representative", "state rep", "tx house", "house dist"]
SENATE_KEYWORDS = ["state senator", "state senate", "tx senate", "senate dist"]

# Viability thresholds (dollars raised)
VIABILITY_THRESHOLD = {"house": 100_000, "senate": 250_000}

YEARS = [2002, 2006, 2010, 2014, 2018, 2022]


# ---------------------------------------------------------------------------
# Tier 1: TEC bulk data (selective extraction from master ZIP via HTTP ranges)
# ---------------------------------------------------------------------------
# TEC publishes a single ~1 GB ZIP with all TX campaign finance data since 2000.
# We extract only filers.csv and cover.csv using HTTP range requests against
# the ZIP central directory — no full download needed.
#
# Key files inside TEC_CF_CSV.zip:
#   filers.csv  — one row per filer: filerIdent, filerName, filerSeekOfficeCd,
#                  filerSeekOfficeDistrict, filerSeekOfficePlace
#   cover.csv   — one row per cover-sheet report: filerIdent, reportInfoIdent,
#                  totalContribAmount, totalExpendAmount, periodStartDt, periodEndDt
#
# Election cycle grouping: sum all cover.csv totalContribAmount for a filer
# where periodEndDt falls within Jan 1 of the election year through general
# election date (~Nov 8 of that year).
# ---------------------------------------------------------------------------

TEC_OFFICE_CODES = {"STATEREP": "house", "STATESEN": "senate"}

# Election cycle end dates (day after general election — contributions through this date)
ELECTION_CYCLE_END = {
    2002: "20021106",
    2006: "20061108",
    2010: "20101103",
    2014: "20141105",
    2018: "20181107",
    2022: "20221109",
}
# Cycle start = Jan 1 of election year (captures full-year fundraising)
ELECTION_CYCLE_START = {yr: f"{yr}0101" for yr in ELECTION_CYCLE_END}


def _tec_range_get(url: str, start: int, end: int) -> bytes | None:
    """HTTP range request. Returns bytes or None on error.

    Note: the TEC server has been observed to silently ignore Range headers
    and return HTTP 200 with the full file body. Callers that depend on
    receiving only the requested slice must detect this themselves.
    """
    headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
    expected = end - start + 1
    try:
        r = requests.get(url, headers=headers, timeout=120)
        if r.status_code == 200 and len(r.content) > expected * 2:
            # Server ignored Range and returned the whole file
            print(f"  TEC range request returned full file (server ignored Range)")
            return None
        if r.status_code not in (200, 206):
            print(f"  TEC range request failed: HTTP {r.status_code}")
            return None
        return r.content
    except requests.RequestException as exc:
        print(f"  TEC range request error: {exc}")
        return None


# Local-ZIP fallback. When TEC range requests fail (broken HEAD, server ignoring
# Range), we download the full ~1 GB ZIP once and serve subsequent reads from it.
_TEC_LOCAL_ZIP_PATH = FINANCE_CACHE / "TEC_CF_CSV.zip"
_TEC_LOCAL_ZIP_HANDLE: zipfile.ZipFile | None = None


def _download_tec_zip_full(url: str) -> Path | None:
    """Stream the full TEC ZIP to a local cache file. Returns path or None."""
    if _TEC_LOCAL_ZIP_PATH.exists() and _TEC_LOCAL_ZIP_PATH.stat().st_size > 100_000_000:
        print(f"  TEC: using cached local ZIP at {_TEC_LOCAL_ZIP_PATH} "
              f"({_TEC_LOCAL_ZIP_PATH.stat().st_size:,} bytes)")
        return _TEC_LOCAL_ZIP_PATH

    tmp_path = _TEC_LOCAL_ZIP_PATH.with_suffix(".zip.part")
    print(f"  TEC: downloading full ZIP to {_TEC_LOCAL_ZIP_PATH} (one-time, ~1 GB)...")
    try:
        with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)
            written = 0
            last_print = 0
            with open(tmp_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    written += len(chunk)
                    if written - last_print >= 100 * 1024 * 1024:
                        pct = f" ({100*written/total:.0f}%)" if total else ""
                        print(f"    downloaded {written/1024/1024:,.0f} MB{pct}")
                        last_print = written
    except Exception as exc:
        print(f"  TEC: full ZIP download failed: {exc}")
        if tmp_path.exists():
            tmp_path.unlink()
        return None

    tmp_path.replace(_TEC_LOCAL_ZIP_PATH)
    print(f"  TEC: downloaded {_TEC_LOCAL_ZIP_PATH.stat().st_size:,} bytes")
    return _TEC_LOCAL_ZIP_PATH


def _open_local_tec_zip(url: str) -> zipfile.ZipFile | None:
    """Return a ZipFile handle for the local TEC ZIP, downloading if needed."""
    global _TEC_LOCAL_ZIP_HANDLE
    if _TEC_LOCAL_ZIP_HANDLE is not None:
        return _TEC_LOCAL_ZIP_HANDLE
    path = _download_tec_zip_full(url)
    if path is None:
        return None
    try:
        _TEC_LOCAL_ZIP_HANDLE = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as exc:
        print(f"  TEC: local ZIP is corrupt ({exc}); deleting so next run re-downloads")
        path.unlink(missing_ok=True)
        return None
    return _TEC_LOCAL_ZIP_HANDLE


def _tec_central_dir_from_local(url: str) -> dict | None:
    """Fallback: open the locally-downloaded ZIP and emit the same dict shape
    as the range-request path. ``local_offset``/``comp_size``/``compression``
    are placeholders here because _tec_extract_file will go through the
    local zipfile path instead of doing range reads."""
    zf = _open_local_tec_zip(url)
    if zf is None:
        return None
    files = {}
    for info in zf.infolist():
        files[info.filename] = {
            "local_offset": info.header_offset,
            "comp_size": info.compress_size,
            "uncomp_size": info.file_size,
            "compression": info.compress_type,
            "_local": True,  # signal to _tec_extract_file
        }
    print(f"  TEC ZIP (local): {len(files)} files found")
    return files


def _tec_zip_central_dir(url: str) -> dict | None:
    """
    Read ZIP central directory from the end of the file without full download.
    Returns dict of {filename: {local_offset, comp_size, uncomp_size, compression}}.
    Falls back to downloading the full ZIP if HEAD or range requests look broken.
    """
    # Step 1: get file size
    file_size = None
    try:
        r = requests.head(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        file_size = int(r.headers["Content-Length"])
    except Exception as exc:
        print(f"  TEC HEAD failed: {exc}")

    # If HEAD reports a suspiciously small size, it's a broken proxy/redirect
    # wrapper — go straight to the full-download fallback. The TEC ZIP is
    # always hundreds of MB.
    if file_size is None or file_size < 100_000_000:
        print(f"  TEC HEAD returned suspect size ({file_size}); using local-ZIP fallback")
        return _tec_central_dir_from_local(url)

    # Step 2: download last 65KB to find End-of-Central-Directory (EOCD)
    eocd_chunk_size = min(65558, file_size)
    eocd_start = file_size - eocd_chunk_size
    eocd_data = _tec_range_get(url, eocd_start, file_size - 1)
    if not eocd_data:
        print("  TEC: range request failed; using local-ZIP fallback")
        return _tec_central_dir_from_local(url)

    sig = b"\x50\x4b\x05\x06"
    idx = eocd_data.rfind(sig)
    if idx == -1:
        print("  TEC: EOCD signature not found; using local-ZIP fallback")
        return _tec_central_dir_from_local(url)

    # Parse EOCD: signature(4) disk(2) disk_start(2) entries_disk(2) entries_total(2)
    #             cd_size(4) cd_offset(4) comment_len(2)
    eocd = eocd_data[idx:]
    if len(eocd) < 22:
        return None
    _, _, _, _, _, cd_size, cd_offset, _ = struct.unpack_from("<4sHHHHIIH", eocd)

    # Step 3: download central directory
    cd_data = _tec_range_get(url, cd_offset, cd_offset + cd_size - 1)
    if not cd_data:
        return None

    # Step 4: parse central directory entries
    files = {}
    pos = 0
    cd_sig = b"\x50\x4b\x01\x02"
    while pos + 46 <= len(cd_data):
        if cd_data[pos:pos + 4] != cd_sig:
            break
        # ZIP central directory record: 17 fields (46 bytes fixed)
        # sig(4s) ver_made(H) ver_need(H) flags(H) compression(H) mod_time(H) mod_date(H)
        # crc32(I) comp_size(I) uncomp_size(I) fname_len(H) extra_len(H) comment_len(H)
        # disk_start(H) int_attrs(H) ext_attrs(I) local_offset(I)
        (_, _, _, flags, compression, _, _,
         _, comp_size, uncomp_size, fname_len, extra_len, comment_len,
         _, _, _, local_offset) = struct.unpack_from("<4sHHHHHHIIIHHHHHII", cd_data, pos)
        fname = cd_data[pos + 46: pos + 46 + fname_len].decode("utf-8", errors="replace")
        files[fname] = {
            "local_offset": local_offset,
            "comp_size": comp_size,
            "uncomp_size": uncomp_size,
            "compression": compression,
        }
        pos += 46 + fname_len + extra_len + comment_len

    print(f"  TEC ZIP central directory: {len(files)} files found")
    return files


def _tec_extract_file(url: str, file_info: dict, fname: str) -> bytes | None:
    """
    Extract a single file from a remote ZIP using HTTP range requests.
    Reads the local file header to find the exact data offset, then downloads
    and decompresses the compressed data.
    """
    cache_path = FINANCE_CACHE / f"tec_{fname.replace('/', '_')}"
    if cache_path.exists() and cache_path.stat().st_size > 100:
        print(f"  TEC cache hit: {fname}")
        return cache_path.read_bytes()

    # Local-ZIP fallback path: read straight from the downloaded ZIP.
    if file_info.get("_local"):
        zf = _open_local_tec_zip(url)
        if zf is None:
            return None
        try:
            data = zf.read(fname)
        except KeyError:
            print(f"  TEC: {fname} not found in local ZIP")
            return None
        cache_path.write_bytes(data)
        print(f"  TEC: extracted {fname} from local ZIP ({len(data):,} bytes)")
        return data

    local_offset = file_info["local_offset"]
    comp_size = file_info["comp_size"]
    uncomp_size = file_info["uncomp_size"]
    compression = file_info["compression"]

    # Read local file header (30 bytes) to get actual variable-length fields
    header_bytes = _tec_range_get(url, local_offset, local_offset + 29)
    if not header_bytes or header_bytes[:4] != b"\x50\x4b\x03\x04":
        print(f"  TEC: bad local file header for {fname}")
        return None

    fname_len = struct.unpack_from("<H", header_bytes, 26)[0]
    extra_len = struct.unpack_from("<H", header_bytes, 28)[0]
    data_start = local_offset + 30 + fname_len + extra_len

    print(f"  TEC: downloading {fname} ({comp_size:,} compressed → {uncomp_size:,} bytes)...")
    compressed = _tec_range_get(url, data_start, data_start + comp_size - 1)
    if not compressed:
        return None

    if compression == 0:  # stored (no compression)
        data = compressed
    elif compression == 8:  # deflate
        try:
            data = zlib.decompress(compressed, wbits=-15)
        except zlib.error as exc:
            print(f"  TEC: decompression error for {fname}: {exc}")
            return None
    else:
        print(f"  TEC: unsupported compression method {compression} for {fname}")
        return None

    cache_path.write_bytes(data)
    print(f"  TEC: extracted {fname} ({len(data):,} bytes)")
    return data


def _parse_tec_filers(filers_data: bytes) -> dict:
    """
    Parse TEC filers.csv to extract TX state legislative candidates.

    Actual column names (confirmed from TEC CSV):
      ctaSeekOfficeCd        — STATEREP or STATESEN
      ctaSeekOfficeDistrict  — district number (numeric string)
      filerIdent             — unique filer ID
      filerName              — full name, "Last, First Middle (Title)" format

    Returns dict: {filerIdent: {name, chamber, district}}
    """
    filers = {}
    text = filers_data.decode(TEC_ENCODING, errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    for row in reader:
        office_cd = row.get("ctaSeekOfficeCd", "").strip().upper()
        chamber = TEC_OFFICE_CODES.get(office_cd)
        if chamber is None:
            continue

        filer_id = row.get("filerIdent", "").strip()
        if not filer_id:
            continue

        dist_raw = row.get("ctaSeekOfficeDistrict", "").strip()
        dist_clean = re.sub(r"\D", "", dist_raw)
        if not dist_clean:
            continue
        district = int(dist_clean)

        name = row.get("filerName", "").strip()

        filers[filer_id] = {
            "name": name,
            "chamber": chamber,
            "district": district,
        }

    print(f"  TEC filers.csv: {len(filers)} legislative candidates found")
    return filers


def _parse_tec_cover_direct(cover_data: bytes, years: list[int]) -> dict:
    """
    Parse TEC cover.csv filtering directly by filerSeekOfficeCd (STATEREP/STATESEN).
    This catches all historical candidates, including those who have since deregistered
    (who would be missing from the current-snapshot filers.csv).

    Returns by_district:
      {year: {(chamber, district): {filer_id: (name, total)}}}
    """
    by_district = {}
    cycle_windows = {yr: (ELECTION_CYCLE_START[yr], ELECTION_CYCLE_END[yr]) for yr in years}

    text = cover_data.decode(TEC_ENCODING, errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    rows_kept = 0
    for row in reader:
        office_cd = row.get("filerSeekOfficeCd", "").strip().upper()
        chamber = TEC_OFFICE_CODES.get(office_cd)
        if chamber is None:
            continue

        dist_raw = row.get("filerSeekOfficeDistrict", "").strip()
        dist_clean = re.sub(r"\D", "", dist_raw)
        if not dist_clean:
            continue
        district = int(dist_clean)
        if district < 1 or district > (150 if chamber == "house" else 31):
            continue

        period_end = re.sub(r"\D", "", row.get("periodEndDt", "").strip())[:8]
        if len(period_end) < 8:
            continue

        total_raw = row.get("totalContribAmount", "0").strip().replace(",", "").replace("$", "")
        try:
            total = float(total_raw)
        except ValueError:
            continue
        if total <= 0:
            continue

        # Determine which election cycle this report belongs to
        matched_yr = None
        for yr, (start, end) in cycle_windows.items():
            if start <= period_end <= end:
                matched_yr = yr
                break
        if matched_yr is None:
            continue

        filer_id = row.get("filerIdent", "").strip()
        name = row.get("filerName", "").strip()

        if matched_yr not in by_district:
            by_district[matched_yr] = {}
        dk = (chamber, district)
        if dk not in by_district[matched_yr]:
            by_district[matched_yr][dk] = {}
        key = filer_id or name  # use name as fallback ID if no filer_id
        existing = by_district[matched_yr][dk].get(key)
        if existing:
            by_district[matched_yr][dk][key] = (name, existing[1] + total)
        else:
            by_district[matched_yr][dk][key] = (name, total)
        rows_kept += 1

    print(f"  TEC cover.csv (direct): {rows_kept} legislative cover rows matched to election cycles")
    for yr in sorted(by_district):
        n = sum(len(filers) for filers in by_district[yr].values())
        print(f"    {yr}: {n} filer-district records across {len(by_district[yr])} districts")
    return by_district


def load_tec_data(years: list[int]) -> dict | None:
    """
    Extract cover.csv from TEC master ZIP via HTTP range requests.
    Filters directly by filerSeekOfficeCd — covers all historical candidates
    regardless of current registration status.
    Returns by_district: {year: {(chamber, district): {filer_id: (name, total)}}}
    or None on failure.
    """
    print("  TEC: reading ZIP central directory...")
    central_dir = _tec_zip_central_dir(TEC_ZIP_URL)
    if not central_dir:
        return None

    def find_file(name):
        for fname in central_dir:
            if fname.lower().endswith(name.lower()):
                return fname
        return None

    cover_fname = find_file("cover.csv")
    if not cover_fname:
        print(f"  TEC: cover.csv not found in ZIP. Available: {list(central_dir.keys())[:10]}")
        return None

    cover_data = _tec_extract_file(TEC_ZIP_URL, central_dir[cover_fname], cover_fname)
    if not cover_data:
        return None

    return _parse_tec_cover_direct(cover_data, years)


def match_tec_parties(tec_by_district: dict, years: list[int]) -> dict:
    """
    TEC doesn't include party in filers.csv.
    Cross-reference against election results CSVs we already collected to assign
    D/R to each TEC filer by name matching.
    Returns same structure but with party keys replacing filer_id keys.
    """
    # Load historical election results to get D/R candidate names
    # Actual file name pattern: tx_house_results_{yr}.csv / tx_senate_results_{yr}.csv
    result_cache = {}
    for yr in years:
        for chamber, fname_prefix in [("house", "tx_house_results"), ("senate", "tx_senate_results")]:
            result_path = DATA_HIST / f"{fname_prefix}_{yr}.csv"
            if result_path.exists():
                with open(result_path, encoding="utf-8-sig") as f:  # utf-8-sig handles BOM
                    for row in csv.DictReader(f):
                        dist_raw = row.get("district", "0").strip()
                        try:
                            dist = int(dist_raw)
                        except ValueError:
                            continue
                        result_cache[(chamber, dist, yr)] = {
                            "d": _normalize_name(row.get("d_candidate", "")),
                            "r": _normalize_name(row.get("r_candidate", "")),
                        }

    matched = {}
    unmatched_total = 0

    # Also load winner_party for single-filer fallback (pre-2018 cycles have no names)
    winner_cache = {}
    for yr in years:
        for chamber, fname_prefix in [("house", "tx_house_results"), ("senate", "tx_senate_results")]:
            result_path = DATA_HIST / f"{fname_prefix}_{yr}.csv"
            if result_path.exists():
                with open(result_path, encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        dist_raw = row.get("district", "0").strip()
                        try:
                            dist = int(dist_raw)
                        except ValueError:
                            continue
                        winner_cache[(chamber, dist, yr)] = row.get("winner_party", "").strip()

    for yr, districts in tec_by_district.items():
        matched[yr] = {}
        for (chamber, district), filers in districts.items():
            known = result_cache.get((chamber, district, yr), {})
            d_name = known.get("d", "")
            r_name = known.get("r", "")

            party_map = {}
            for filer_id, (name, total) in filers.items():
                norm = _normalize_name(name)
                if d_name and _name_match(norm, d_name):
                    party = "D"
                elif r_name and _name_match(norm, r_name):
                    party = "R"
                else:
                    party = None
                    unmatched_total += 1

                if party:
                    existing = party_map.get(party)
                    if existing is None or total > existing[1]:
                        party_map[party] = (name, total)

            # Single-filer fallback: if exactly one filer and no name match,
            # assign to winner_party if known (catches pre-2018 cycles with no candidate names)
            if not party_map and len(filers) == 1:
                winner_party = winner_cache.get((chamber, district, yr), "")
                if winner_party in ("D", "R"):
                    (fid, (name, total)) = next(iter(filers.items()))
                    party_map[winner_party] = (name, total)

            if party_map:
                matched[yr][(chamber, district)] = party_map

    if unmatched_total:
        print(f"  TEC party matching: {unmatched_total} filers could not be matched to D/R")
    return matched


def _normalize_name(name: str) -> str:
    """Normalize a candidate name for fuzzy matching: uppercase, strip titles/suffixes."""
    name = name.upper().strip()
    # Remove parenthetical content like "(Mr.)" or "(The Honorable)"
    name = re.sub(r"\([^)]*\)", " ", name)
    # Remove common titles and suffixes
    for token in [" JR", " SR", " II", " III", " IV", " MR", " MRS", " MS", " DR",
                  " THE HONORABLE", " HONORABLE", ",", "."]:
        name = name.replace(token, "")
    # Keep only alpha characters and spaces
    name = re.sub(r"[^A-Z ]", " ", name)
    return " ".join(name.split())


def _name_match(a: str, b: str) -> bool:
    """
    Fuzzy name match tolerant of "LAST FIRST" vs "FIRST LAST" ordering.
    TEC stores names as "VANDEAVER GARY W" (after normalize strips comma).
    Results store names as "GARY VANDEAVER".
    Strategy: check if significant tokens overlap across both name strings.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    tokens_a = set(t for t in a.split() if len(t) >= 4)
    tokens_b = set(t for t in b.split() if len(t) >= 4)
    if not tokens_a or not tokens_b:
        return False
    # Exact token overlap
    common = tokens_a & tokens_b
    if len(common) >= 2:
        return True
    if len(common) == 1 and len(next(iter(common))) >= 5:
        return True
    # Prefix match: handles nickname/formal pairs like TRENT/TRENTON, STEVE/STEVEN
    for ta in tokens_a:
        for tb in tokens_b:
            min_len = min(len(ta), len(tb))
            if min_len >= 5 and (ta.startswith(tb[:min_len]) or tb.startswith(ta[:min_len])):
                return True
    return False


# ---------------------------------------------------------------------------
# Tier 2: FollowTheMoney API (bulk fetch per year+chamber)
# ---------------------------------------------------------------------------

# c-r-ot office codes confirmed from FTM API discovery:
FTM_CHAMBER_CODE = {"house": "H", "senate": "S"}


def _ftm_val(rec: dict, field_name: str) -> str:
    """
    Extract the display value from a nested FTM record field.
    FTM returns every grouped field as a nested dict:
      {"token": "...", "id": "...", "FieldName": "display value"}
    So we access rec[field_name][field_name] to get the display value.
    """
    field = rec.get(field_name)
    if field is None:
        return ""
    if isinstance(field, dict):
        return str(field.get(field_name, ""))
    return str(field)


def _ftm_get(params: dict) -> dict | None:
    """Single FTM API call. Returns parsed JSON or None."""
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(FTM_API_URL, params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"  FTM HTTP {resp.status_code}")
            return None
        return resp.json()
    except Exception as exc:
        print(f"  FTM request error: {exc}")
        return None


def _parse_ftm_district(office_sought: str) -> int | None:
    """Parse 'HOUSE DISTRICT 021' or 'SENATE DISTRICT 14' → int district number."""
    m = re.search(r'district\s+(\d+)', office_sought, re.IGNORECASE)
    return int(m.group(1)) if m else None


def fetch_ftm_all_candidates(year: int, chamber: str) -> list[dict]:
    """
    Bulk fetch all TX legislative candidates for a year+chamber from FTM API.
    Uses a single call per page (typically 1-5 pages) rather than one call per district.
    Returns list of raw record dicts from the API.
    """
    if not FTM_API_KEY:
        return []

    chamber_code = FTM_CHAMBER_CODE[chamber]
    base_params = {
        "dt": "1",
        "y": str(year),
        "s": "TX",
        "c-r-ot": chamber_code,
        "gro": "c-t-id,c-r-id",
        "APIKey": FTM_API_KEY,
        "mode": "json",
    }

    all_records = []
    page = 0
    max_page = 0

    while page <= max_page:
        params = {**base_params, "p": str(page)}
        data = _ftm_get(params)
        if not data:
            break

        if page == 0:
            paging = data.get("metaInfo", {}).get("paging", {})
            max_page = int(paging.get("maxPage", 0))
            total = paging.get("totalRecords", "?")
            print(f"  FTM {year} {chamber}: {total} candidates, {max_page + 1} page(s)")

        records = data.get("records", [])
        all_records.extend(records)
        page += 1
        if page <= max_page:
            time.sleep(0.3)

    return all_records


def aggregate_ftm_by_district(records: list[dict]) -> dict:
    """
    Convert FTM raw candidate records into by_district dict.
    Keeps only general election candidates (excludes primary-only losers).
    Returns: {district: {"D": (name, total), "R": (name, total)}}
    """
    by_district = {}

    for rec in records:
        election_status = _ftm_val(rec, "Election_Status")
        # Keep only candidates who appeared in the general election
        if "General" not in election_status and "Default" not in election_status:
            continue

        party_raw = _ftm_val(rec, "General_Party")
        if "Republican" in party_raw:
            party = "R"
        elif "Democrat" in party_raw:
            party = "D"
        else:
            continue  # skip third-party / unknown

        office = _ftm_val(rec, "Office_Sought")
        district = _parse_ftm_district(office)
        if district is None:
            continue

        total_str = _ftm_val(rec, "Total_$").replace(",", "").replace("$", "").strip()
        try:
            total = float(total_str)
        except ValueError:
            total = 0.0

        candidate = _ftm_val(rec, "Candidate")

        if district not in by_district:
            by_district[district] = {}

        # If multiple same-party candidates reach the general (rare), keep highest fundraiser
        existing = by_district[district].get(party)
        if existing is None or total > existing[1]:
            by_district[district][party] = (candidate, total)

    return by_district


# ---------------------------------------------------------------------------
# Derive finance variables
# ---------------------------------------------------------------------------

def derive_finance_vars(dem_raised: float, rep_raised: float,
                        chamber: str, incumbency: str = "") -> dict:
    """
    Compute fundraising-derived variables for the regression.

    incumbency: "D_incumbent", "R_incumbent", or "open"
    """
    total = dem_raised + rep_raised
    threshold = VIABILITY_THRESHOLD[chamber.lower()]

    if total > 0:
        dem_share = round(dem_raised / total, 6)
    else:
        dem_share = None

    # Challenger = whoever is NOT the incumbent (or lower fundraiser for open seats)
    if incumbency == "D_incumbent":
        challenger_raised = rep_raised
    elif incumbency == "R_incumbent":
        challenger_raised = dem_raised
    else:  # open seat or unknown
        challenger_raised = min(dem_raised, rep_raised)

    incumbent_raised = max(dem_raised, rep_raised) if incumbency != "open" else None

    log_challenger = round(math.log1p(challenger_raised), 6) if challenger_raised >= 0 else None
    viability_flag = int(challenger_raised >= threshold) if challenger_raised >= 0 else None

    return {
        "dem_fundraising_share": dem_share,
        "log_challenger_fundraising": log_challenger,
        "challenger_viability_flag": viability_flag,
        "incumbent_fundraising": round(incumbent_raised, 2) if incumbent_raised else None,
    }


# ---------------------------------------------------------------------------
# Build output rows
# ---------------------------------------------------------------------------

def build_rows_from_district_map(year: int, chamber: str,
                                  by_district: dict, source: str) -> list[dict]:
    """
    Convert a by_district map {district: {party: (name, total)}} into output rows.
    Fills in placeholder rows for districts missing from the map.
    """
    rows = []
    max_d = 150 if chamber == "house" else 31
    confidence = "tec_aggregate" if source == "tec_bulk" else "ftm_api"

    for district in range(1, max_d + 1):
        parties = by_district.get(district, {})
        dem_info = parties.get("D", ("", 0.0))
        rep_info = parties.get("R", ("", 0.0))

        if not parties:
            rows.append({
                "year": year, "chamber": chamber.title(), "district": district,
                "dem_candidate": "", "rep_candidate": "",
                "dem_raised": None, "rep_raised": None,
                "dem_fundraising_share": None, "log_challenger_fundraising": None,
                "challenger_viability_flag": None, "incumbent_fundraising": None,
                "name_match_confidence": "no_match",
                "data_source": source, "MANUAL_NEEDED": True,
            })
            continue

        dem_raised = dem_info[1]
        rep_raised = rep_info[1]
        finance = derive_finance_vars(dem_raised, rep_raised, chamber)

        rows.append({
            "year": year,
            "chamber": chamber.title(),
            "district": district,
            "dem_candidate": dem_info[0],
            "rep_candidate": rep_info[0],
            "dem_raised": round(dem_raised, 2),
            "rep_raised": round(rep_raised, 2),
            **finance,
            "name_match_confidence": confidence,
            "data_source": source,
            "MANUAL_NEEDED": False,
        })

    return rows


def build_rows_from_tec(year: int, by_district: dict) -> list[dict]:
    """Convert aggregated TEC data into output row format."""
    rows = []
    for (chamber, district), parties in by_district.items():
        dem_info = parties.get("D", ("", 0.0))
        rep_info = parties.get("R", ("", 0.0))

        dem_raised = dem_info[1]
        rep_raised = rep_info[1]
        finance = derive_finance_vars(dem_raised, rep_raised, chamber)

        rows.append({
            "year": year,
            "chamber": chamber.title(),
            "district": district,
            "dem_candidate": dem_info[0],
            "rep_candidate": rep_info[0],
            "dem_raised": round(dem_raised, 2),
            "rep_raised": round(rep_raised, 2),
            **finance,
            "name_match_confidence": "tec_aggregate",
            "data_source": "tec_bulk",
            "MANUAL_NEEDED": False,
        })
    return rows


def build_placeholder_rows(year: int) -> list[dict]:
    """Build placeholder rows for all TX House + Senate districts."""
    rows = []
    for chamber, max_d in [("house", 150), ("senate", 31)]:
        for district in range(1, max_d + 1):
            rows.append({
                "year": year,
                "chamber": chamber.title(),
                "district": district,
                "dem_candidate": "",
                "rep_candidate": "",
                "dem_raised": None,
                "rep_raised": None,
                "dem_fundraising_share": None,
                "log_challenger_fundraising": None,
                "challenger_viability_flag": None,
                "incumbent_fundraising": None,
                "name_match_confidence": "no_match",
                "data_source": "placeholder",
                "MANUAL_NEEDED": True,
            })
    return rows


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    fields = ["year", "chamber", "district", "dem_candidate", "rep_candidate",
              "dem_raised", "rep_raised", "dem_fundraising_share",
              "log_challenger_fundraising", "challenger_viability_flag",
              "incumbent_fundraising", "name_match_confidence",
              "data_source", "MANUAL_NEEDED"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manual = sum(1 for r in rows if r["MANUAL_NEEDED"])
    print(f"  Wrote {len(rows)} rows → {path.name}  ({manual} MANUAL_NEEDED)")


# ---------------------------------------------------------------------------
# Main collection
# ---------------------------------------------------------------------------

def collect_year_from_tec(year: int, tec_matched: dict) -> list[dict]:
    """Build output rows for one year from pre-loaded TEC party-matched data."""
    year_data = tec_matched.get(year, {})
    rows = []
    for chamber in ["house", "senate"]:
        # by_district for this chamber: {district: {"D": (name, total), "R": (name, total)}}
        chamber_map = {
            dist: parties
            for (ch, dist), parties in year_data.items()
            if ch == chamber
        }
        chamber_rows = build_rows_from_district_map(year, chamber, chamber_map, "tec_csv")
        filled = sum(1 for r in chamber_rows if not r["MANUAL_NEEDED"])
        print(f"  TEC {year} {chamber}: {filled}/{len(chamber_rows)} districts filled")
        rows.extend(chamber_rows)
    return rows


def collect_year(year: int, tec_matched: dict | None = None, use_ftm: bool = True) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Finance: TX {year}")
    print(f"{'='*60}")

    rows = []

    # Tier 1: TEC (pre-loaded across all years)
    if tec_matched is not None:
        rows = collect_year_from_tec(year, tec_matched)

    # Tier 2: FollowTheMoney bulk API (one call per page, not per district)
    if not rows and use_ftm and FTM_API_KEY:
        print(f"  Trying FollowTheMoney API (bulk mode)...")
        ftm_rows = []
        for chamber in ["house", "senate"]:
            raw_records = fetch_ftm_all_candidates(year, chamber)
            if raw_records:
                by_district = aggregate_ftm_by_district(raw_records)
                chamber_rows = build_rows_from_district_map(
                    year, chamber, by_district, source="followthemoney_api"
                )
                ftm_rows.extend(chamber_rows)
                filled = sum(1 for r in chamber_rows if not r["MANUAL_NEEDED"])
                print(f"  FTM {chamber}: {filled}/{len(chamber_rows)} districts filled")
        if ftm_rows:
            rows = ftm_rows

    # Tier 3: Placeholder
    if not rows:
        print(f"  No automated source succeeded — writing placeholders.")
        if not FTM_API_KEY:
            print(f"    Set FOLLOWTHEMONEY_API_KEY in .env to enable FTM API.")
        rows = build_placeholder_rows(year)

    return rows


def main(years=None, use_tec: bool = True, use_ftm: bool = True):
    years = years or YEARS

    # TEC: load once for all years (range-extracts filers.csv + cover.csv from master ZIP)
    tec_matched = None
    if use_tec:
        print("\nLoading TEC master data (range-extracting from ZIP)...")
        tec_raw = load_tec_data(years)
        if tec_raw:
            tec_matched = match_tec_parties(tec_raw, years)
            total_filled = sum(
                sum(1 for parties in yr_data.values() if parties)
                for yr_data in tec_matched.values()
            )
            print(f"  TEC: {total_filled} district-years with D/R matched")

    for year in years:
        rows = collect_year(year, tec_matched=tec_matched, use_ftm=use_ftm)
        out = DATA_HIST / f"tx_finance_{year}.csv"
        write_csv(rows, out)

    print(f"\n{'='*60}")
    print("Finance collection complete.")
    print(f"Output directory: {DATA_HIST}")
    print("\nNOTE: Finance data quality varies significantly by year.")
    print("  2018 and 2022 TEC data is generally complete and accurate.")
    print("  Older years (2002-2010) may have lower filing compliance.")
    print("  Verify unusual amounts (very high or $0) against TEC filings directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect TX legislative campaign finance data")
    parser.add_argument("--year", type=int, choices=YEARS)
    parser.add_argument("--no-tec", action="store_true", help="Skip TEC bulk download")
    parser.add_argument("--no-ftm", action="store_true", help="Skip FollowTheMoney API")
    args = parser.parse_args()

    target_years = [args.year] if args.year else None
    main(years=target_years, use_tec=not args.no_tec, use_ftm=not args.no_ftm)
