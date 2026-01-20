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
LOG="/var/log/allsky360-endofnight.log"

CURRENT_USER=$(whoami)
# or
# CURRENT_USER=$(id -un)

USER_HOME=$HOME
# or
# USER_HOME=$(getent passwd "$CURRENT_USER" | cut -d: -f6)

# ========== CONFIG ==========

BINDIR="/opt/allsky360/bin"
IMAGEDIR="${USER_HOME}/images"
DATABASEDIR="${USER_HOME}/database"
VIDEODIR="${USER_HOME}/videos"
WIDTH=640
HEIGHT=640
FPS=25

# =============================================
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [END_OF_NIGHT] $*" >> "$LOG"
}

# Get the current date and time
DATE=$(date +%Y%m%d)
TIME=$(date +%H%M%S)
YESTERDAY=$(date -d "yesterday" +%Y%m%d 2>/dev/null || date -v -1d +%Y%m%d 2>/dev/null || echo "")

# Get the latest image
LATEST_IMAGE=$(ls -t ${IMAGEDIR}/*.jpg 2>/dev/null | head -n 1)

# Function to get night images from database
get_night_images() {
  local FILES_LIST_YESTERDAY=""
  local FILES_LIST_TODAY=""
  local DB_TODAY="${DATABASEDIR}/database_${DATE}.db"
  local DB_YESTERDAY=""
  
  # Check if sqlite3 is available
  if ! command -v sqlite3 &> /dev/null; then
    log "ERROR: sqlite3 command not found!"
    return 1
  fi
  
  # Check if yesterday's database exists (night might span across midnight)
  if [ -n "$YESTERDAY" ]; then
    DB_YESTERDAY="${DATABASEDIR}/database_${YESTERDAY}.db"
  fi

  # echo "DB_TODAY: $DB_TODAY"
  # echo "DB_YESTERDAY: $DB_YESTERDAY"

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
    #echo "No night images found in database!" >&2
    return 0
  fi

  # LAST_NIGHT_END_STR=$(date -d "@${LAST_NIGHT_END}" "+%Y.%m.%d %H:%M:%S" 2>/dev/null)
  # echo "LAST_NIGHT_END: $LAST_NIGHT_END"
  # echo "LAST_NIGHT_END_STR: $LAST_NIGHT_END_STR"

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

  # LAST_DAY_BEFORE_NIGHT_STR=$(date -d "@${LAST_DAY_BEFORE_NIGHT}" "+%Y.%m.%d %H:%M:%S" 2>/dev/null)
  #echo "LAST_DAY_BEFORE_NIGHT: $LAST_DAY_BEFORE_NIGHT"
  # echo "LAST_DAY_BEFORE_NIGHT_STR: $LAST_DAY_BEFORE_NIGHT_STR"

  if [ -f "$DB_YESTERDAY" ]; then
    TIMESTAMP_LIST=$(sqlite3 "$DB_YESTERDAY" "SELECT timestamp FROM image WHERE night_mode = 1 AND timestamp > ${LAST_DAY_BEFORE_NIGHT} AND timestamp <= ${LAST_NIGHT_END} ORDER BY timestamp ASC;" 2>/dev/null)
    if [ -n "$TIMESTAMP_LIST" ]; then
      #echo "TIMESTAMP_LIST: $TIMESTAMP_LIST"
      #echo "--------------------------------"
    for timestamp in $TIMESTAMP_LIST; do
      local file_date=$(date -u -d "@${timestamp}" +%Y%m%d 2>/dev/null || date -u -r "${timestamp}" +%Y%m%d 2>/dev/null)
      local file_time=$(date -u -d "@${timestamp}" +%H%M%S 2>/dev/null || date -u -r "${timestamp}" +%H%M%S 2>/dev/null)
      local image_file="${IMAGEDIR}/${file_date}/image-${file_date}${file_time}.jpg"
      FILES_LIST_YESTERDAY="${FILES_LIST_YESTERDAY}${FILES_LIST_YESTERDAY:+ }${image_file}"
    done
  fi
fi

#echo "FILES_LIST_YESTERDAY: $FILES_LIST_YESTERDAY" >&2
#echo "Yesterday images --------------------------------" >&2

  # Now get all night_mode=1 timestamps between LAST_DAY_BEFORE_NIGHT and LAST_NIGHT_END from both databases
  if [ -f "$DB_TODAY" ]; then
    TIMESTAMP_LIST=$(sqlite3 "$DB_TODAY" "SELECT timestamp FROM image WHERE night_mode = 1 AND timestamp > ${LAST_DAY_BEFORE_NIGHT} AND timestamp <= ${LAST_NIGHT_END} ORDER BY timestamp ASC;" 2>/dev/null)
    if [ -n "$TIMESTAMP_LIST" ]; then
      #echo "TIMESTAMP_LIST: $TIMESTAMP_LIST"
      #echo "--------------------------------"
      for timestamp in $TIMESTAMP_LIST; do
        local file_date=$(date -u -d "@${timestamp}" +%Y%m%d 2>/dev/null || date -u -r "${timestamp}" +%Y%m%d 2>/dev/null)
          local file_time=$(date -u -d "@${timestamp}" +%H%M%S 2>/dev/null || date -u -r "${timestamp}" +%H%M%S 2>/dev/null)
        local image_file="${IMAGEDIR}/${file_date}/image-${file_date}${file_time}.jpg"
        FILES_LIST_TODAY="${FILES_LIST_TODAY}${FILES_LIST_TODAY:+ }${image_file}"
      done
    fi
  fi

#echo "FILES_LIST_TODAY: $FILES_LIST_TODAY" >&2
#echo "Today images --------------------------------" >&2

  # Return the list
  echo "$FILES_LIST_YESTERDAY $FILES_LIST_TODAY"
}

generate_timelapse() {
  local MODE="night"
  local DATESTAMP=$(date +%Y%m%d)
  
  # Get night images from database
  FILES=$(get_night_images)
  
  if [ -z "$FILES" ]; then
    log "WARNING: No night images found in database!"
    return 1
  fi
  
  log "Generating timelapse"
  log "Found $(echo $FILES | wc -w) night images"

  # echo "$BINDIR/allsky360-timelapse -o ${VIDEODIR}/${MODE}_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -v $FILES"

  log "Generating normal timelapse"
  $BINDIR/allsky360-timelapse -o ${VIDEODIR}/${MODE}_normal_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -s ${VIDEODIR}/${MODE}_normal_${DATESTAMP}.png -v $FILES
  log "Generating motion timelapse"
  $BINDIR/allsky360-timelapse -o ${VIDEODIR}/${MODE}_motion_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -m motion -s ${VIDEODIR}/${MODE}_motion_${DATESTAMP}.png -v $FILES
  log "Generating sigma timelapse"
  $BINDIR/allsky360-timelapse -o ${VIDEODIR}/${MODE}_sigma_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -m sigma -s ${VIDEODIR}/${MODE}_sigma_${DATESTAMP}.png -v $FILES
  log "Generating min timelapse"
  $BINDIR/allsky360-timelapse -o ${VIDEODIR}/${MODE}_min_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -m min -s ${VIDEODIR}/${MODE}_min_${DATESTAMP}.png -v $FILES
  log "Generating max timelapse"
  $BINDIR/allsky360-timelapse -o ${VIDEODIR}/${MODE}_max_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -m max -s ${VIDEODIR}/${MODE}_max_${DATESTAMP}.png -v $FILES
  log "Generating diff timelapse"
  $BINDIR/allsky360-timelapse -o ${VIDEODIR}/${MODE}_diff_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -m diff -s ${VIDEODIR}/${MODE}_diff_${DATESTAMP}.png -v $FILES
  log "Generating avg timelapse"
  $BINDIR/allsky360-timelapse -o ${VIDEODIR}/${MODE}_avg_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -m avg -s ${VIDEODIR}/${MODE}_avg_${DATESTAMP}.png -v $FILES

  log "Timelapse creation finished ------------------"
}

# Generate the timelapse
generate_timelapse $LATEST_IMAGE

