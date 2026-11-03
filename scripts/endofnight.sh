#!/bin/bash
#
# endofnight.sh
# End of night tasks for Allsky360
#
# Usage:
#   endofnight.sh
#
# This script is designed to be executed via systemd-run.
#
# Copyright (c) 2025 Sven Kreiensen
#
# You can use this software under the terms of the MIT software license
# (see LICENSE.md)
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.
#

# Configuration
BINDIR="/opt/allsky360/bin"
CONFIG_FILE="${ALLSKY360_CONFIG:-/home/allskyuser/config.ini}"
WIDTH=800
HEIGHT=800
FPS=5

# The daemon spawns this hook detached with stdout/stderr on /dev/null, so
# everything has to be captured here. /var/log/allsky360/ is what logrotate
# covers.
LOG="${ALLSKY360_ENDOFNIGHT_LOG:-/var/log/allsky360/endofnight.log}"
mkdir -p "$(dirname "$LOG")" 2>/dev/null || LOG=/dev/null
exec >> "$LOG" 2>&1
echo "===== $(date '+%Y-%m-%d %H:%M:%S') end of night run ====="

# Building a whole night of timelapses takes minutes - never run two at once.
LOCKFILE="/run/allsky360/endofnight.lock"
if command -v flock >/dev/null 2>&1; then
  mkdir -p "$(dirname "$LOCKFILE")" 2>/dev/null || LOCKFILE="/tmp/allsky360-endofnight.lock"
  exec 9>"$LOCKFILE"
  if ! flock -n 9; then
    echo "WARNING: another end-of-night run is still active, aborting."
    exit 0
  fi
fi

# Read paths from config.ini
read_config() {
  local key="$1"
  local default="$2"
  if [ -f "$CONFIG_FILE" ]; then
    local val
    val=$(grep -E "^${key}=" "$CONFIG_FILE" 2>/dev/null | cut -d'=' -f2- | tr -d '[:space:]')
    echo "${val:-$default}"
  else
    echo "$default"
  fi
}

# The daemon exports these; config.ini is the fallback when the script is
# started by hand.
IMAGEDIR="${ALLSKY360_IMAGE_DIRECTORY:-$(read_config "image_directory" "/opt/allsky360/images")}"
DATABASEDIR="${ALLSKY360_DATABASE_DIRECTORY:-$(read_config "database_directory" "/opt/allsky360/database")}"
VIDEODIR="${ALLSKY360_VIDEO_DIRECTORY:-$(read_config "video_directory" "/opt/allsky360/allsky-api/public/videos")}"

if [ ! -x "$BINDIR/allsky360-timelapse" ]; then
  echo "ERROR: $BINDIR/allsky360-timelapse not found or not executable."
  exit 1
fi

# Current date and time in UTC: images/YYYYMMDD and database_YYYYMMDD.db are
# named after the UTC day (independent of the system time zone and of
# summer/winter time). The night is found via the timestamps in today's and
# yesterday's database, so it may span midnight.
DATE=$(date -u +%Y%m%d)
TIME=$(date -u +%H%M%S)
YESTERDAY=$(date -u -d "yesterday" +%Y%m%d 2>/dev/null || date -u -v-1d +%Y%m%d 2>/dev/null || echo "")

# Get the latest image
LATEST_IMAGE=$(ls -t ${IMAGEDIR}/*.jpg 2>/dev/null | head -n 1)

# Function to get night images from database
get_night_images() {
  local FILES_LIST=""
  local DB_TODAY="${DATABASEDIR}/database_${DATE}.db"
  local DB_YESTERDAY=""
  
  # Check if sqlite3 is available
  if ! command -v sqlite3 &> /dev/null; then
    echo "ERROR: sqlite3 command not found!" >&2
    echo ""
    return 1
  fi
  
  # Check if yesterday's database exists (night might span across midnight)
  if [ -n "$YESTERDAY" ]; then
    DB_YESTERDAY="${DATABASEDIR}/database_${YESTERDAY}.db"
  fi
  
  # Find the last night by querying databases
  # Strategy: Find the last continuous sequence of night_mode=1 entries
  
  # Get the last night_mode=1 timestamp (end of last night) from both databases
  local LAST_NIGHT_END=""
  if [ -f "$DB_TODAY" ]; then
    LAST_NIGHT_END=$(sqlite3 "$DB_TODAY" "SELECT MAX(timestamp) FROM image WHERE night_mode = 1;" 2>/dev/null)
  fi
  if [ -n "$DB_YESTERDAY" ] && [ -f "$DB_YESTERDAY" ]; then
    local YESTERDAY_END=$(sqlite3 "$DB_YESTERDAY" "SELECT MAX(timestamp) FROM image WHERE night_mode = 1;" 2>/dev/null)
    if [ -n "$YESTERDAY_END" ]; then
      if [ -z "$LAST_NIGHT_END" ] || [ "$YESTERDAY_END" -gt "$LAST_NIGHT_END" ]; then
        LAST_NIGHT_END="$YESTERDAY_END"
      fi
    fi
  fi
  
  if [ -z "$LAST_NIGHT_END" ]; then
    echo ""
    return 0
  fi
  
  # Find the start of the last night: find the last night_mode=0 before LAST_NIGHT_END
  # Then take all night_mode=1 entries after that point
  local LAST_DAY_BEFORE_NIGHT="0"
  if [ -f "$DB_TODAY" ]; then
    local TODAY_LAST_DAY=$(sqlite3 "$DB_TODAY" "SELECT COALESCE(MAX(timestamp), 0) FROM image WHERE night_mode = 0 AND timestamp < ${LAST_NIGHT_END};" 2>/dev/null)
    if [ -n "$TODAY_LAST_DAY" ] && [ "$TODAY_LAST_DAY" -gt "$LAST_DAY_BEFORE_NIGHT" ]; then
      LAST_DAY_BEFORE_NIGHT="$TODAY_LAST_DAY"
    fi
  fi
  if [ -n "$DB_YESTERDAY" ] && [ -f "$DB_YESTERDAY" ]; then
    local YESTERDAY_LAST_DAY=$(sqlite3 "$DB_YESTERDAY" "SELECT COALESCE(MAX(timestamp), 0) FROM image WHERE night_mode = 0 AND timestamp < ${LAST_NIGHT_END};" 2>/dev/null)
    if [ -n "$YESTERDAY_LAST_DAY" ] && [ "$YESTERDAY_LAST_DAY" -gt "$LAST_DAY_BEFORE_NIGHT" ]; then
      LAST_DAY_BEFORE_NIGHT="$YESTERDAY_LAST_DAY"
    fi
  fi
  
  # If no day entry found before the night, take the first night_mode=1 entry as start
  if [ "$LAST_DAY_BEFORE_NIGHT" = "0" ]; then
    if [ -f "$DB_TODAY" ]; then
      LAST_DAY_BEFORE_NIGHT=$(sqlite3 "$DB_TODAY" "SELECT MIN(timestamp) FROM image WHERE night_mode = 1;" 2>/dev/null)
    fi
    if [ -z "$LAST_DAY_BEFORE_NIGHT" ] || [ "$LAST_DAY_BEFORE_NIGHT" = "" ]; then
      if [ -n "$DB_YESTERDAY" ] && [ -f "$DB_YESTERDAY" ]; then
        LAST_DAY_BEFORE_NIGHT=$(sqlite3 "$DB_YESTERDAY" "SELECT MIN(timestamp) FROM image WHERE night_mode = 1;" 2>/dev/null)
      fi
    fi
    # If still no start found, use 0 to get all night images
    if [ -z "$LAST_DAY_BEFORE_NIGHT" ] || [ "$LAST_DAY_BEFORE_NIGHT" = "" ]; then
      LAST_DAY_BEFORE_NIGHT="0"
    fi
  fi
  
  # Now get all night_mode=1 timestamps between LAST_DAY_BEFORE_NIGHT and LAST_NIGHT_END from both databases
  if [ -f "$DB_TODAY" ]; then
    while IFS= read -r timestamp; do
      if [ -n "$timestamp" ] && [ "$timestamp" -gt "$LAST_DAY_BEFORE_NIGHT" ] && [ "$timestamp" -le "$LAST_NIGHT_END" ]; then
        # Convert Unix timestamp to YYYYMMDDHHMMSS format
        local file_date=$(date -u -d "@${timestamp}" +%Y%m%d 2>/dev/null || date -u -r "${timestamp}" +%Y%m%d 2>/dev/null)
        local file_time=$(date -u -d "@${timestamp}" +%H%M%S 2>/dev/null || date -u -r "${timestamp}" +%H%M%S 2>/dev/null)
        
        if [ -n "$file_date" ] && [ -n "$file_time" ]; then
          local image_file="${IMAGEDIR}/${file_date}/image-${file_date}${file_time}.jpg"
          
          # Check if file exists
          if [ -f "$image_file" ]; then
            FILES_LIST="${FILES_LIST}${FILES_LIST:+ }${image_file}"
          fi
        fi
      fi
    done < <(sqlite3 "$DB_TODAY" "SELECT timestamp FROM image WHERE night_mode = 1 AND timestamp > ${LAST_DAY_BEFORE_NIGHT} AND timestamp <= ${LAST_NIGHT_END} ORDER BY timestamp ASC;" 2>/dev/null)
  fi
  
  if [ -n "$DB_YESTERDAY" ] && [ -f "$DB_YESTERDAY" ]; then
    while IFS= read -r timestamp; do
      if [ -n "$timestamp" ] && [ "$timestamp" -gt "$LAST_DAY_BEFORE_NIGHT" ] && [ "$timestamp" -le "$LAST_NIGHT_END" ]; then
        # Convert Unix timestamp to YYYYMMDDHHMMSS format
        local file_date=$(date -u -d "@${timestamp}" +%Y%m%d 2>/dev/null || date -u -r "${timestamp}" +%Y%m%d 2>/dev/null)
        local file_time=$(date -u -d "@${timestamp}" +%H%M%S 2>/dev/null || date -u -r "${timestamp}" +%H%M%S 2>/dev/null)
        
        if [ -n "$file_date" ] && [ -n "$file_time" ]; then
          local image_file="${IMAGEDIR}/${file_date}/image-${file_date}${file_time}.jpg"
          
          # Check if file exists and not already in list
          if [ -f "$image_file" ] && [[ ! "$FILES_LIST" =~ "$image_file" ]]; then
            FILES_LIST="${FILES_LIST}${FILES_LIST:+ }${image_file}"
          fi
        fi
      fi
    done < <(sqlite3 "$DB_YESTERDAY" "SELECT timestamp FROM image WHERE night_mode = 1 AND timestamp > ${LAST_DAY_BEFORE_NIGHT} AND timestamp <= ${LAST_NIGHT_END} ORDER BY timestamp ASC;" 2>/dev/null)
  fi
  
  # Return the list
  echo "$FILES_LIST"
}

generate_timelapse() {
  local MODES="max motion"
  local DATESTAMP=$(date -u +%Y%m%d)
  
  # Get night images from database
  FILES=$(get_night_images)
  
  if [ -z "$FILES" ]; then
    echo "WARNING: No night images found in database!"
    return 1
  fi
  
  # Normal
  echo "Generating normal timelapse for $1"
  echo "Found $(echo $FILES | wc -w) night images"
  mkdir -p "$VIDEODIR"
  $BINDIR/allsky360-timelapse -o ${VIDEODIR}/normal_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -s ${VIDEODIR}/normal_${DATESTAMP}.png -v $FILES

  # All other modes
  for MODE in $MODES; do
    echo "Generating $MODE timelapse for $1"
    echo "Found $(echo $FILES | wc -w) night images"
    mkdir -p "$VIDEODIR"
    $BINDIR/allsky360-timelapse -o ${VIDEODIR}/${MODE}_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -s ${VIDEODIR}/${MODE}_${DATESTAMP}.png -m ${MODE} -v $FILES
  done
}

# Generate the timelapse
generate_timelapse $LATEST_IMAGE

