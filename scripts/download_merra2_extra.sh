#!/bin/bash
# Download the MERRA-2 "trapping" variable subsets used by
# notebooks/merra2_trapping_pipeline.ipynb.
#
#   PBLH              from M2T1NXFLX  -> data/raw/merra2_flx/
#   T850, SLP, TQV    from M2T1NXSLV  -> data/raw/merra2_slv_extra/
#
# Both are OPeNDAP subsets of the same 3x3 grid box over LA used everywhere else in
# this project, so each granule is ~30-50 KB instead of ~400 MB.
#
# Requires a ~/.netrc entry for urs.earthdata.nasa.gov (chmod 600).
#
# Usage:  bash scripts/download_merra2_extra.sh [parallelism]
#
# Idempotent — re-run it to fill in anything that failed. Run it a second time if the
# summary reports failures; transient 500s from GES DISC are common.

set -u
cd "$(dirname "$0")/.."

PAR="${1:-4}"
LAT="247:249"
LON="98:100"
BASE="https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2"

SLV_DIR="data/raw/merra2_slv"
FLX_DIR="data/raw/merra2_flx"
SLVX_DIR="data/raw/merra2_slv_extra"
mkdir -p "$FLX_DIR" "$SLVX_DIR"

if [ ! -d "$SLV_DIR" ] || [ -z "$(ls -A "$SLV_DIR" 2>/dev/null)" ]; then
  echo "error: $SLV_DIR is empty. Download the base SLV granules first (see README)," >&2
  echo "       since this script takes its date list and stream numbers from them." >&2
  exit 1
fi

# MERRA-2 processing-stream numbers (400/401) are irregular and interleave, so we take
# the authoritative mapping from the filenames of the granules already on disk.
MAP=$(mktemp)
trap 'rm -f "$MAP"' EXIT
ls "$SLV_DIR"/*.nc4 \
  | sed -E 's/.*MERRA2_([0-9]+)\.tavg1_2d_slv_Nx\.([0-9]{8})\.nc4/\2 \1/' \
  | sort > "$MAP"
echo "$(wc -l < "$MAP") dates to fetch, parallelism $PAR"

fetch() {
  d="$1"; prod="$2"; map="$3"
  s=$(awk -v D="$d" '$1==D{print $2}' "$map")
  [ -z "$s" ] && return 1
  y="${d:0:4}"; m="${d:4:2}"
  if [ "$prod" = "flx" ]; then
    out="data/raw/merra2_flx/flx_${d}.nc4"
    url="$4/M2T1NXFLX.5.12.4/$y/$m/MERRA2_${s}.tavg1_2d_flx_Nx.${d}.nc4.nc4"
    ce="PBLH%5B0:23%5D%5B$5%5D%5B$6%5D,time,lat%5B$5%5D,lon%5B$6%5D"
  else
    out="data/raw/merra2_slv_extra/slvx_${d}.nc4"
    url="$4/M2T1NXSLV.5.12.4/$y/$m/MERRA2_${s}.tavg1_2d_slv_Nx.${d}.nc4.nc4"
    ce="T850%5B0:23%5D%5B$5%5D%5B$6%5D,SLP%5B0:23%5D%5B$5%5D%5B$6%5D,TQV%5B0:23%5D%5B$5%5D%5B$6%5D,time,lat%5B$5%5D,lon%5B$6%5D"
  fi
  [ -s "$out" ] && return 0
  # per-process cookie jar: a shared one races during the Earthdata OAuth redirect
  ck=$(mktemp)
  for attempt in 1 2 3; do
    code=$(curl -sS -L -n -c "$ck" -b "$ck" --max-time 180 -o "$out" -w "%{http_code}" "${url}?${ce}" 2>/dev/null)
    if [ "$code" = "200" ] && [ -s "$out" ]; then rm -f "$ck"; return 0; fi
    sleep $((attempt * 2))
  done
  rm -f "$out" "$ck"
  return 1
}
export -f fetch
export BASE LAT LON

for prod in flx slvx; do
  echo "--- $prod ---"
  awk '{print $1}' "$MAP" \
    | xargs -P "$PAR" -I{} bash -c 'fetch "$@" || echo "FAIL {}"' _ {} "$prod" "$MAP" "$BASE" "$LAT" "$LON"
done

echo
echo "PBLH granules:         $(ls "$FLX_DIR"  2>/dev/null | wc -l)"
echo "T850/SLP/TQV granules: $(ls "$SLVX_DIR" 2>/dev/null | wc -l)"
echo "expected:              $(wc -l < "$MAP")"
echo
echo "If the counts are short, just run this script again — it skips what it already has."
