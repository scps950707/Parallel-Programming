#!/bin/sh
if [ $# -eq '0' ]; then
    echo "./exe [cores] [exec times]"
else
    for i in `seq 1 $2`:
    do
        t=$(expr $1 - 1)
        taskset -c 0-$t ./bin/cg | grep -E "VERIFICATION|Initialization|Execution"
        echo "--------------------------------------------------"
    done
fi
