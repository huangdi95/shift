#!/bin/bash
gpu_id=3
gpu_type="V100-16GB-"$gpu_id
echo "remove "$gpu_type".csv"
rm -rf $gpu_type.csv
for c1 in 8192 4096 2048 1024 512 256 128 64 32
do
for c2 in 8192 4096 2048 1024 512 256 128 64 32
do
for g in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
do
if [ $g -gt $c1 -o $g -gt $c2 ]; then
continue
fi
echo "running group="$g "c1="$c1 "c2="$c2
CUDA_VISIBLE_DEVICES=$gpu_id \
python measure_time.py --group=$g  --c1=$c1 --c2=$c2 \
2>&1 | tee run.log
done
done
done
