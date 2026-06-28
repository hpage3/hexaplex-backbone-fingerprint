#!/usr/bin/env bash
# Always read inputs from ./input_data and write outputs to ./output
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python3}"
PYFILE="${SCRIPT_DIR}/planes_from_backbone_ortho_boxes.py"
INPUT_DIR="${SCRIPT_DIR}/input_data"
OUTDIR="${SCRIPT_DIR}/output"

# Defaults that are always applied
DEFAULTS=(--csv --plot --color-ss --force-chain A --as-sticks --stick-radius 0.15)

usage() {
  cat <<'USAGE'
Usage:
#	bash boxed.sh --all       
  ./run_boxes_input_data.sh <name1[.pdb]> [name2[.pdb] ...]
  ./run_boxes_input_data.sh --all

Notes:
  • Inputs are resolved inside ./input_data (no paths needed).
  • Outputs go to ./output/<name>_boxes.* (directory auto-created).
USAGE
}

[[ -f "$PYFILE" ]] || { echo "[error] Can't find $PYFILE" >&2; exit 2; }
mkdir -p "$INPUT_DIR" "$OUTDIR"

process_one() {
  local name="$1"
  case "$name" in
    *.pdb|*.PDB) : ;;
    *) name="${name}.pdb" ;;
  esac

  local in_abs="${INPUT_DIR}/${name}"
  if [[ ! -f "$in_abs" ]]; then
    echo "[warn] Missing: ${in_abs} — skipping" >&2
    return 1
  fi

  local stem out_pdb
  stem="${name%.*}"
  out_pdb="${OUTDIR}/${stem}_boxes.pdb"

  echo "[info] Processing: ${in_abs}"

  # Copy the original input PDB into the output directory for reference
  # (keeps the original coordinates unmodified).
  cp "$in_abs" "${OUTDIR}/${stem}_input.pdb"

  set -x
  "$PY" "$PYFILE" --outdir "$OUTDIR" --output "$out_pdb" "$in_abs" "${DEFAULTS[@]}"
  { set +x; } 2>/dev/null
}

if [[ $# -eq 0 ]]; then usage; exit 1; fi

if [[ "$1" == "--all" ]]; then
  shift
  shopt -s nullglob
  files=( "$INPUT_DIR"/*.pdb "$INPUT_DIR"/*.PDB )
  shopt -u nullglob
  to_run=()
  for f in "${files[@]}"; do
    bn="$(basename "$f")"
    stem="${bn%.*}"
    low="$(printf '%s' "$stem" | tr '[:upper:]' '[:lower:]')"
    [[ "$low" == *_boxes* ]] && continue
    to_run+=( "$bn" )
  done
  if (( ${#to_run[@]} == 0 )); then
    echo "[info] No plain *.pdb files found in ${INPUT_DIR}/"
    exit 0
  fi
  for n in "${to_run[@]}"; do process_one "$n" || true; done
  exit 0
fi

for n in "$@"; do process_one "$n" || true; done
