#!/usr/bin/env bash
#
# allsky360_api.sh
# Abruf von Allsky360 API-Daten und Bildern (aktuelles Kamerabild / Panorama).
#
# Beispiele:
#   ./scripts/allsky360_api.sh --api-url http://192.168.3.62:3000 --target latest_image --mode json
#   ./scripts/allsky360_api.sh --api-url http://192.168.3.62:3000 --target latest_image --mode download --output latest.jpg
#   ./scripts/allsky360_api.sh --api-url http://192.168.3.62:3000 --target panorama --mode download --output panorama.jpg
#   ./scripts/allsky360_api.sh --api-url http://192.168.3.62:3000 --target config --mode json
#

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

API_URL=""
TARGET=""
MODE="json"
OUTPUT=""

usage() {
  cat <<EOF
Verwendung:
  $SCRIPT_NAME --api-url <URL> --target <ziel> [--mode json|download] [--output <datei>] [--date YYYY-MM-DD]

Parameter:
  --api-url   Basis-URL der API (z. B. http://192.168.3.62:3000)
  --target    Ziel:
              latest_image | panorama | config | image | meteo | aurora | aircraft | aircraft_status
  --mode      json (Standard) oder download
  --output    Ausgabedatei bei Download (optional)
  --date      Datum für image/meteo/aurora im Format YYYY-MM-DD
  -h, --help  Diese Hilfe anzeigen

Hinweise:
  - latest_image unterstützt --mode json und --mode download.
  - panorama unterstützt --mode download.
EOF
}

error() {
  echo "Fehler: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || error "Benötigtes Kommando fehlt: $1"
}

trim_trailing_slash() {
  local value="$1"
  echo "${value%/}"
}

api_get_json() {
  local endpoint="$1"
  curl --silent --show-error --fail "${API_URL}${endpoint}"
}

download_file() {
  local endpoint="$1"
  local output_file="$2"
  curl --silent --show-error --fail --location --output "$output_file" "${API_URL}${endpoint}"
}

json_value() {
  local json="$1"
  local key="$2"
  printf '%s' "$json" | jq -r --arg k "$key" '.[$k] // empty'
}

DATE_VALUE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-url)
      API_URL="${2:-}"
      shift 2
      ;;
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT="${2:-}"
      shift 2
      ;;
    --date)
      DATE_VALUE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      error "Unbekannter Parameter: $1 (siehe --help)"
      ;;
  esac
done

[[ -n "$API_URL" ]] || error "--api-url ist erforderlich"
[[ -n "$TARGET" ]] || error "--target ist erforderlich"

API_URL="$(trim_trailing_slash "$API_URL")"

case "$MODE" in
  json|download) ;;
  *) error "--mode muss 'json' oder 'download' sein" ;;
esac

require_command curl

case "$TARGET" in
  latest_image)
    if [[ "$MODE" == "json" ]]; then
      api_get_json "/api/latest_image"
    elif [[ "$MODE" == "download" ]]; then
      require_command jq

      latest_json="$(api_get_json "/api/latest_image")"
      date_part="$(json_value "$latest_json" "date")"
      image_time="$(json_value "$latest_json" "image")"
      filetype="$(json_value "$latest_json" "filetype")"

      [[ -n "$date_part" && -n "$image_time" && -n "$filetype" ]] || error "Ungültige Antwort von /api/latest_image"

      datetime="${date_part}${image_time}"
      endpoint="/images/${date_part}/image-${datetime}.${filetype}"
      default_name="image-${datetime}.${filetype}"
      output_file="${OUTPUT:-$default_name}"
      download_file "$endpoint" "$output_file"
      echo "{\"status\":\"ok\",\"downloaded\":\"$output_file\",\"source\":\"${API_URL}${endpoint}\"}"
    else
      error "Ungültiger Modus für '$TARGET': $MODE"
    fi
    ;;

  config)
    [[ "$MODE" == "json" ]] || error "Target '$TARGET' unterstützt nur --mode json"
    api_get_json "/api/config"
    ;;

  image|meteo|aurora)
    [[ "$MODE" == "json" ]] || error "Target '$TARGET' unterstützt nur --mode json"
    [[ -n "$DATE_VALUE" ]] || error "--date ist für Target '$TARGET' erforderlich"
    [[ "$DATE_VALUE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || error "--date muss im Format YYYY-MM-DD sein"
    api_get_json "/api/${TARGET}?date=${DATE_VALUE}"
    ;;

  aircraft)
    [[ "$MODE" == "json" ]] || error "Target '$TARGET' unterstützt nur --mode json"
    api_get_json "/api/aircraft"
    ;;

  aircraft_status)
    [[ "$MODE" == "json" ]] || error "Target '$TARGET' unterstützt nur --mode json"
    api_get_json "/api/aircraft/status"
    ;;

  panorama)
    [[ "$MODE" == "download" ]] || error "Target '$TARGET' unterstützt nur --mode download"
    require_command jq

    latest_json="$(api_get_json "/api/latest_image")"
    date_part="$(json_value "$latest_json" "date")"
    image_time="$(json_value "$latest_json" "image")"
    filetype="$(json_value "$latest_json" "filetype")"

    [[ -n "$date_part" && -n "$image_time" && -n "$filetype" ]] || error "Ungültige Antwort von /api/latest_image"

    datetime="${date_part}${image_time}"
    endpoint="/images/${date_part}/panorama-${datetime}.${filetype}"
    default_name="panorama-${datetime}.${filetype}"

    output_file="${OUTPUT:-$default_name}"
    download_file "$endpoint" "$output_file"
    echo "{\"status\":\"ok\",\"downloaded\":\"$output_file\",\"source\":\"${API_URL}${endpoint}\"}"
    ;;

  *)
    error "Unbekanntes Target: $TARGET"
    ;;
esac
