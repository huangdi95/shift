#!/bin/bash
groups=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
channel_in=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
channel_out=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
height=(7 14 28 56 112 224 448 896)
width=(7 14 28 56 112 224 448 896)
for gpu_id in 1
do
    gpu_type="V100-32GB-"$gpu_id
    csv="csvs/"$gpu_type"-single.csv"
    echo "remove "$csv
    rm -rf $csv
    for h in ${height[*]} 
    do
    for w in ${width[*]} 
    do
    for c1 in ${channel_in[*]} 
    do
    for c2 in ${channel_out[*]} 
    do
    for g in ${groups[*]}
    do
        if [ $g -gt $c1 -o $g -gt $c2 ]; then
            continue
        fi
        echo "running group="$g "h="$h "w="$w "c1="$c1 "c2="$c2
        CUDA_VISIBLE_DEVICES=$gpu_id \
        python measure_time.py --group=$g --h=$h --w=$w --c1=$c1 --c2=$c2 --filename=$csv --gpu-type=$gpu_type \
        2>&1 | tee $gpu_type.log
    done
    done
    done
    done
    done
done
