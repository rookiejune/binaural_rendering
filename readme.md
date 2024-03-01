# Neural Binaural Rendering
## What it is
Binaural rendering provides vivid sound experience for headphone users by transforming surround sounds to binaural audios.
Previously, it only can be done by measuring Head-Related Transfer Functions (HRTFs), which is costly.
Although there are many public HRTF datasets,
their application scope is not extensive enough.
Therefore, we proposed a framework that is independent of HRTFs and has comparable performance.

This work is helpful for users who has personalized needs and cannot afford the cost of measuring HRTFs.

More details can be found in our paper: [End-to-End Paired Ambisonic-Binaural Audio Rendering](https://www.ieee-jas.net/en/article/id/7f409b94-2b8f-4143-99d2-e5ae6fdede8e).

## How to start

### Prepare dataset
We released ByteDance Paired Ambisonics and Binaurals (BTPAB) at https://zenodo.org/record/7212795.
BTPAB is recorded by Neumann KU100 dummy head in a living band and has total 49-minutes audios.
All audios are of 48000 Hz sample rate, and the ambisonics are of A-format AmbiX.

Due to the nature of binaural rendering, we highly recommend users to record their own dataset if possible.
The experiments show that our method has pleasing performance even with 30-minutes audios.


### Train
Run the following two commands to preprocess the dataset.
```[bash]
bash scripts/1_pack_audios_to_hdf5s/1.sh
bash scripts/2_create_indexes/1.sh
```
Note that the variable `DATASET_DIR` is the path to the dataset and `WORKSPACE` is the path to workspace (contains hdf5 files, logs, models, etc).

Then run the following command to train the model.
```[bash]
bash scripts/3_train_binaural_rendering/1.sh
```

## Contact with us

Email to yinzhu20@fudan.edu.cn for details of the paper.