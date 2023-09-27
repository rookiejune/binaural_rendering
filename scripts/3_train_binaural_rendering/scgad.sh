#!/bin/bash
WORKSPACE=/mnt/local1_ssd/zhuyin/Workspace/binaural_rendering
INDEXES_CONFIG_YAML="configs/1.yaml"

CUDA_VISIBLE_DEVICES='0' python ../../train_binaural_rendering.py \
    --workspace=$WORKSPACE \
    --config_yaml=$INDEXES_CONFIG_YAML
