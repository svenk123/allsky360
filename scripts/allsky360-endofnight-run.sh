#!/bin/sh
# Usage: allsky360-endofnight-run

# Build --setenv=... args for every ALLSKY_* variable
SETENV_ARGS=""
for name in $(env | sed -n 's/^\(ALLSKY360_[A-Za-z0-9_]*\)=.*/\1/p'); do
  # Note: This assumes values don't contain newlines. Avoid secrets or escape properly if needed.
  eval "val=\${$name}"
  SETENV_ARGS="$SETENV_ARGS --setenv=$name=$val"
done

exec /usr/bin/systemd-run \
  --unit=ai-processing-%j \
  --description="Allsky360 End of Night" \
  --collect \
  --property=User=allskyuser \
  --property=Group=allskyuser \
  --property=WorkingDirectory=/opt/allsky360 \
  --property=UMask=002 \
  --property=SetLoginEnvironment=yes \
  $SETENV_ARGS \
  /opt/allsky360/scripts/endofnight.sh
