#!/bin/sh
TIMECMD="time -f real:%E\nuser:%U\nsystem:%S\nCPU:%P"
$TIMECMD ./pi 1 10000000
echo "---------------------"
$TIMECMD ./pi 2 10000000
echo "---------------------"
$TIMECMD ./pi 4 10000000
echo "---------------------"
$TIMECMD ./pi 8 10000000
echo "---------------------"
$TIMECMD ./pi 1 100000000
echo "---------------------"
$TIMECMD ./pi 2 100000000
echo "---------------------"
$TIMECMD ./pi 4 100000000
echo "---------------------"
$TIMECMD ./pi 8 100000000
echo "---------------------"
