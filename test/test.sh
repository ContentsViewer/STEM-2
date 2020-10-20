#!/bin/sh
# ros2 topic echo --csv /general_sensor_data > ii1 &
# ros2 topic echo --csv /balloon_sensor_data > ii2 &
# wait

grep 'name:' | ( while read a; do set $a && ros2 topic echo --csv $2 > $(echo $2 | sed -e 's/\//-/g') & done; wait )