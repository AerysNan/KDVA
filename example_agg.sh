#!/bin/bash
DATASETS=${@:3}

CUDA_VISIBLE_DEVICES=$1 AMLT_OUTPUT_DIR=$2 python tools/full_pipeline.py \
  --datasets $DATASETS \
  --anno-template {}.g.json \
  --n-window 30 \
  --train-rate 1 \
  --val-rate 0.1 \
  --agg-name agg \
  --aggregation 3 \
  --cfg configs/custom/ssd_head.py \
  --backbone-cfg configs/custom/ssd.py \
  --eval-template {}.eval.pkl \
  --n-process 4 \
  --framerates 5 10 15 20 25 30