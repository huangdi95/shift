#!/bin/bash
CUDA_VISIBLE_DEVICES=3 \
nvprof --analysis-metrics -o ./metrics/$1 \
python measure_time.py
