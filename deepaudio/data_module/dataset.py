from typing import Dict, List

import h5py
import numpy as np

from .utils import int16_to_float32


class Dataset:
    def __init__(
        self,
        source_types: List[str],
        segment_samples: int,
    ):
        r"""Used for getting data according to a meta.

        Args:
            input_source_types: list of str, e.g., ['vocals', 'accompaniment']
            target_source_types: list of str, e.g., ['vocals']
            input_channels: int
            augmentor: Augmentor
            segment_samples: int
        """
        self.segment_samples = segment_samples
        self.source_types = source_types

    def __getitem__(self, meta: Dict) -> Dict:
        r"""Return data according to a meta. E.g., an input meta looks like: {
            'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
            'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}.
        }

        Then, vocals segments of song_A and song_B will be mixed (mix-audio augmentation).
        Accompaniment segments of song_C and song_B will be mixed (mix-audio augmentation).
        Finally, mixture is created by summing vocals and accompaniment.

        Args:
            meta: dict, e.g., {
                'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
                'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}
            }

        Returns:
            data_dict: dict, e.g., {
                'vocals': (channels, segments_num),
                'accompaniment': (channels, segments_num),
                'mixture': (channels, segments_num),
            }
        """
        data_dict = {}

        for source_type in self.source_types:
            # E.g., ['vocals', 'accompaniment']

            waveforms = []  # Audio segments to be mix-audio augmented.

            for m in meta[source_type]:
                # E.g., {
                #     'hdf5_path': '.../song_A.h5',
                #     'key_in_hdf5': 'vocals',
                #     'begin_sample': '13406400',
                #     'end_sample': 13538700,
                # }
                hdf5_path = m['hdf5_path']
                key_in_hdf5 = m['key_in_hdf5']
                bgn_sample = m['begin_sample']
                end_sample = bgn_sample + self.segment_samples

                with h5py.File(hdf5_path, 'r') as hf:
                    waveform = int16_to_float32(
                        hf[key_in_hdf5][:, bgn_sample:end_sample]
                    )
                    # (input_channels, segments_num)
                waveforms.append(waveform)
            # E.g., waveforms: [(input_channels, audio_samples), (input_channels, audio_samples)]

            # mix-audio augmentation
            data_dict[source_type] = np.sum(waveforms, axis=0)
            # data_dict[source_type]: (input_channels, audio_samples)

        # data_dict looks like: {
        #     'vocals': (input_channels, audio_samples),
        #     'accompaniment': (input_channels, audio_samples)
        # }
        return data_dict
