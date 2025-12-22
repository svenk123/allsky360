#!/bin/bash

INTERVAL=300
CONFIG=/home/allskyuser/config.ini
DATABASEDIR=/opt/allsky360/database
OUTFILE=/opt/allsky360/allsky-api/public/latest_aurora_data.json

while true; do
    ./tools/aurora/bin/allsky360-aurora -c ${CONFIG} -f ${OUTFILE} -d ${DATABASEDIR} --kp --mag --plasma
    sleep $INTERVAL
done

exit 0

