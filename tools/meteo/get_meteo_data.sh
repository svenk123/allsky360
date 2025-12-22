#!/bin/bash

INTERVAL=900
CONFIG=/home/allskyuser/config.ini
DATABASEDIR=/opt/allsky360/database
OUTFILE=/opt/allsky360/allsky-api/public/latest_meteo_data.json

while true; do
    ./tools/meteo/bin/allsky360-meteo -c ${CONFIG} -f ${OUTFILE} -d ${DATABASEDIR}
    sleep $INTERVAL
done

exit 0

