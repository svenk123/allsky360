#!/bin/sh

BASEDIR=/home/user1/timelapse_cuda_nv
BINDIR=$BASEDIR/bin
VIDEODIR=$BASEDIR/videos
WIDTH=800
HEIGHT=800
FPS=5
DATESTAMP=20250630

cd $BASEDIR

mkdir -p $VIDEODIR

FILES=`cat $VIDEODIR/t.txt`

#MODES="avg max min sum sigma diff motion"

MODES="sigma"

cd $VIDEODIR
for MODE in $MODES; do
    echo "$BINDIR/timelapse_cuda_nvenc -o ${MODE}_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -s ${MODE}_${DATESTAMP}.png -m ${MODE} -v"
    echo " "

    $BINDIR/timelapse_cuda_nvenc -o ${MODE}_${DATESTAMP}.mp4 -w $WIDTH -h $HEIGHT -p $FPS -s ${MODE}_${DATESTAMP}.png -m ${MODE} -v $FILES

    echo "Video creation (mode: $MODE) finisched ------------------"
done

cd -

exit 0
