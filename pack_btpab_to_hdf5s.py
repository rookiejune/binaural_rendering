# For ByTedance Paired Ambisonic and Binaural (BTPAB) dataset
import argparse
from pathlib import Path
import time

import h5py
import numpy as np

from deepaudio.data_module.utils import float32_to_int16, load_audio


def pack_dir_to_hdf5(
    dataset_dir,
    hdf5_dir,
    sample_rate,
    ) -> None:
    r"""Pack (resampled) audio files into hdf5 files to speed up loading.

    Args:
        audio_dir,
        hdf5_dir,
        source_type,
        sample_rate,
    Returns:
        None
    """

    # arguments & parameters
    dataset_dir = Path(dataset_dir)
    hdf5_dir = Path(hdf5_dir)
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    ambisonic_dir: Path = dataset_dir/"ambisonic"
    binaural_dir: Path = dataset_dir/"binaural"

    ambisonic_paths = sorted([p for p in ambisonic_dir.iterdir()])
    binaural_paths = sorted([p for p in binaural_dir.iterdir()])

    start_time = time.time()
    for ambisonic_path, binaural_path in zip(ambisonic_paths, binaural_paths):
        assert ambisonic_path.stem[6:] == binaural_path.stem[9:], \
            "Ambisonic path {} not aligns with binaural path {}".format(ambisonic_path, binaural_path)
        hdf5_path = hdf5_dir/"{}.h5".format(ambisonic_path.stem[6:])

        write_audio_to_hdf5(
            ambisonic_path,
            binaural_path,
            hdf5_path,
            sample_rate,
        )

    print("Pack hdf5 time: {:.3f} s".format(time.time() - start_time))
    print("HDF5 structure: ")
    print("- Attrs: audio_name, sample_rate")
    print("- Dataset: ambisonic, binaural")


def write_audio_to_hdf5(
    ambisonic_path,
    binaural_path,
    hdf5_path,
    sample_rate,
    ):
    r"""Write single audio into hdf5 file."""
    with h5py.File(hdf5_path, "w") as hf:
        hf.attrs.create("audio_name", data=ambisonic_path.stem, dtype="S100")
        hf.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)

        ambisonic = load_audio(
            audio_path=ambisonic_path, mono=False, sample_rate=sample_rate)
        # audio: (channels_num, audio_samples)
        hf.create_dataset(
            name="ambisonic", data=float32_to_int16(ambisonic), dtype=np.int16
        )

        binaural = load_audio(
            audio_path=binaural_path, mono=False, sample_rate=sample_rate)
        hf.create_dataset(
            name="binaural", data=float32_to_int16(binaural), dtype=np.int16
        )

    print('Write hdf5: {}'.format(hdf5_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--hdf5s_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sample_rate", type=int, required=True, help="Sample rate."
    )

    args = parser.parse_args()

    pack_dir_to_hdf5(
        args.dataset_dir,
        args.hdf5s_dir,
        args.sample_rate,
    )
