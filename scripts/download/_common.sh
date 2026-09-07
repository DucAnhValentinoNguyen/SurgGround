# Sourced by scripts/download/*.sh — shared helpers. Not executable on its own.
set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT first  ->  source scripts/env_4090.sh}"
RAW="$DATA_ROOT/raw"
mkdir -p "$RAW"

have() { command -v "$1" >/dev/null 2>&1; }

# fetch <url> <dest-file>  — resumable; aria2c if available, else wget -c
fetch() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if have aria2c; then
    aria2c -c -x8 -s8 --dir "$(dirname "$dest")" --out "$(basename "$dest")" "$url"
  else
    wget -c -O "$dest" "$url"
  fi
}

note() { printf '\n\033[1m[%s]\033[0m %s\n' "${SCRIPT:-download}" "$*"; }
die()  { printf '\n\033[1;31m[%s] %s\033[0m\n' "${SCRIPT:-download}" "$*" >&2; exit 1; }

note "DATA_ROOT=$DATA_ROOT  RAW=$RAW  host=$(hostname)"
