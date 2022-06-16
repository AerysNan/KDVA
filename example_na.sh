#!/bin/bash
len=$(expr \( $# - 1 \) / 2)
AMLT_OUTPUT_DIRS=(${@:2:$len})
DATASETS=(${@:$(expr $len + 2)})
for i in $(seq 0 1 $(expr $len - 1))
do
  CUDA_VISIBLE_DEVICES=$1 AMLT_OUTPUT_DIR=${AMLT_OUTPUT_DIRS[$i]} python tools/full_pipeline.py \
    --datasets ${DATASETS[$i]} \
    --anno-template {}.g.json \
    --n-window 30 \
    --train-rate 1 \
    --val-rate 0.1 \
    --cfg configs/custom/ssd.py \
    --eval-template {}.eval.pkl \
    --n-process 8 \
    --framerates 5 10 15 20 25 30
done