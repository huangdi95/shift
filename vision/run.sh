#!/bin/bash
groups=(1 2 4 8 16 32 64 128 256 512 1024 2048)
for model in 'shufflenet_v2_group'
do
    for gpu_id in 1
    do
        gpu_type="V100-32GB-"$gpu_id
        csv="csvs/"$gpu_type"-"$model".csv"
        echo "remove "$csv
        rm -rf $csv
        for g in ${groups[*]} 
        do
            echo "running group="$g
            CUDA_VISIBLE_DEVICES=$gpu_id \
            python model_time.py \
                --group=$g \
                --filename=$csv \
                --gpu-type=$gpu_type \
                --model=$model \
                --max-group=${groups[-1]} \
             2>&1 | tee logs/$gpu_type"-"$model".log"
        done
    done
done
