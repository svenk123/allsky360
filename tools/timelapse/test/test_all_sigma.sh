#!/bin/sh

BASEDIR=/home/user1/timelapse_cuda_nv
BINDIR=$BASEDIR/bin
VIDEODIR=$BASEDIR/videos
TESTDIR=$BASEDIR/test
WIDTH=800
HEIGHT=800
FPS=5
DATESTAMP=20250629

cd $BASEDIR

mkdir -p $VIDEODIR

FILES=`cat $TESTDIR/${DATESTAMP}.txt`

#MODES="avg max min sum sigma diff motion"

MODES="motion"
#THRESHOLDS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0"
THRESHOLDS="75.0 90.0 100.0 120.0"

cd $VIDEODIR
for MODE in $MODES; do
    for THRESHOLD in $THRESHOLDS; do
	echo "$BINDIR/timelapse_cuda_nvenc -o ${MODE}_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -s ${MODE}_${DATESTAMP}.png -m ${MODE} -t ${THRESHOLD} -v"
	echo " "

	$BINDIR/timelapse_cuda_nvenc -o ${MODE}_${THRESHOLD}_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -s ${MODE}_${THRESHOLD}_${DATESTAMP}.png -m ${MODE} -t ${THRESHOLD} -v $FILES

	echo "Video creation (mode: $MODE, threshold: ${THRESHOLD}.0) finisched ------------------"
    done
done

cd -

exit 0
