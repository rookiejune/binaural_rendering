#!/bash
# DATASET_DIR=/hy-tmp/musdb18
# WORKSPACE=/hy-tmp/workspace/sslm

DATASET_DIR=/mnt/local1_ssd/zhuyin/Dataset/btpab22
WORKSPACE=/mnt/local1_ssd/zhuyin/Workspace/binaural_rendering

python ../../pack_btpab_to_hdf5s.py \
    --dataset_dir=${DATASET_DIR}/train \
    --hdf5s_dir=${WORKSPACE}/hdf5s/btpab/train \
    --sample_rate=44100 \

# python ../../pack_btpab_to_hdf5s.py \
#     --dataset_dir=${DATASET_DIR}/test \
#     --hdf5s_dir=${WORKSPACE}/hdf5s/btpab/test \
#     --sample_rate=44100 \
