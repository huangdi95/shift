#!/bin/bash
for g in 1 2 4 8 16 32 64 128
#for g in 32 64
do
echo "profiling group="$g
CUDA_VISIBLE_DEVICES=3 \
#nvprof --export-profile ./torchprof/"g"$g \
#python measure_time.py --g=$g
nvprof --profile-from-start off -o ./torchprof/nvtx/"g"$g \
python measure_time.py --g=$g
done
