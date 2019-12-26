#!/bin/bash
echo "remove output.csv"
rm -rf output.csv
for c1 in 32 64 128 256 512 1024 2048
do
for c2 in 32 64 128 256 512 1024 2048
do
for g in 1 2 4 8 16 32 64 128 256 512 1024 2048
do
if [ $g -gt $c1 -o $g -gt $c2 ]; then
continue
fi
echo "running group="$g "c1="$c1 "c2="$c2
CUDA_VISIBLE_DEVICES=3 \
python measure_time.py --g=$g  --c1=$c1 --c2=$c2
done
done
done

