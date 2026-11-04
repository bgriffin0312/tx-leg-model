#!/usr/bin/env bash
# Election-night race board (2026-11-03). Every ~5 minutes: statewide projection -> statewide races ->
# close legislative seats -> seat alerts, plus any new Collin County report. One block per cycle, every
# line prefixed "BOARD|", appended to poll.log in a single write so a monitor gets it as one event.
cd /c/Users/Brennan/code/tx-leg-model || exit 1
LOG=data/raw/election_night/poll.log
COL=data/raw/election_night/collin
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
SIGF=data/raw/election_night/.board_sig
while true; do
  tmp=$(mktemp)
  # Collin posts PDFs; pull any report not seen yet
  for href in $(curl -sL -m 20 -A "$UA" https://www.collincountytx.gov/elections/election-results | grep -oiE 'href="/docs/default-source/elections/election-results/[^"]*\.pdf[^"]*"' | grep -vi write-in | sed 's/href="//; s/"$//'); do
    name=$(basename "${href%%\?*}" .pdf)
    case "$name" in *early-voting*) out="$COL/collin_ev_summary";; *) out="$COL/collin_${name}";; esac
    if [ ! -f "$out.txt" ]; then
      curl -sL -m 60 -A "$UA" -o "$out.pdf" "https://www.collincountytx.gov$href" && \
      python -c "import pdfplumber,sys; p=pdfplumber.open(sys.argv[1]); open(sys.argv[2],'w',encoding='utf-8').write('\n'.join((pg.extract_text() or '') for pg in p.pages))" "$out.pdf" "$out.txt" && \
      echo "NEW COLLIN REPORT: $name" >> "$tmp.c"
    fi
  done
  python scripts/early_vote/statewide_projection.py 2>&1 | head -15 > "$tmp.p"
  cat "$tmp.p" >> "$LOG"     # the race and seat scripts read the newest projection from poll.log
  python scripts/early_vote/statewide_races.py 2>&1 | head -20 > "$tmp.s"
  python scripts/early_vote/seat_report.py 2>&1 | sed -n '/HOUSE PROJECTION/,/Flagged legislative/p' | grep -v "^Flagged" > "$tmp.h"
  python scripts/early_vote/seat_diff.py > "$tmp.d" 2>&1
  { cat "$tmp.c" 2>/dev/null; grep -E "expected vote|Election Day factor|PROJECTED T|Traceback|Error" "$tmp.p"; cat "$tmp.s" "$tmp.h" "$tmp.d"; } > "$tmp.body"
  # post only when a number moved: signature ignores the clock stamps in the section headers
  # alerts are new by construction (seat_diff only prints changes), so they post on their own and stay out of the signature
  # the county-count and factor lines change every cycle without moving any projection, so they stay out too
  sig=$(grep -v -E "ALERT|expected vote by status|Election Day factor|=== STATEWIDE RACES" "$tmp.body" | sed -E 's/[0-9]{2}:[0-9]{2}(:[0-9]{2})?//g' | md5sum | cut -c1-32)
  if [ "$sig" != "$(cat "$SIGF" 2>/dev/null)" ] || grep -q "ALERT" "$tmp.body"; then
    { echo "=== RACE BOARD $(date +%H:%M) ==="; cat "$tmp.body"; } | sed 's/^/BOARD| /' > "$tmp.b"
    cat "$tmp.b" >> "$LOG"
    echo "$sig" > "$SIGF"
  else
    echo "race board $(date +%H:%M): no change" >> "$LOG"
  fi
  rm -f "$tmp" "$tmp".*
  sleep 240
done
