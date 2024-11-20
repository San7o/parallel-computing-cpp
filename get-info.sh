#!/bin/sh

# Use this script to get information
# about the system

OUTPUT_FILE="info.txt"
OUTPUT=${PWD}/$OUTPUT_FILE

touch $OUTPUT

echo "===== /proc/meminfo =====" >> $OUTPUT
cat /proc/meminfo >> $OUTPUT

echo "===== /proc/cpuinfo =====" >> $OUTPUT
cat /proc/cpuinfo >> $OUTPUT

echo "===== lscpu =====" >> $OUTPUT
lscpu >> $OUTPUT

echo "==== uname =====" >> $OUTPUT
uname -a >> $OUTPUT 

echo "Done"
