#!/bin/bash
# WORKSPACE=/mnt/local1_ssd/zhuyin/Workspace/binaural_rendering
INDEXES_CONFIG_YAML="configs/btpab.yaml"

python ../../create_indexes.py \
    --config_yaml=$INDEXES_CONFIG_YAML
