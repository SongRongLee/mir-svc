import torch
import random
import h5py
from torch.utils.data import Dataset
from pathlib import Path

from ..util.singer_mapping import SingerMapping
from ..util.const import SAMPLE_RATE, MEL_HOP_LENGTH


class TrainingDataset(Dataset):
    def __init__(self, data_dir, singer_mapping=None, epoch_len=1000):
        self.epoch_len = epoch_len
        self.data_dir = Path(data_dir)
        self.singer_paths = list(self.data_dir.iterdir())
        self.singer_data_paths = {singer_path.stem: list(singer_path.iterdir()) for singer_path in self.singer_paths}
        if singer_mapping is not None:
            self.singer_mapping = singer_mapping
        else:
            self.singer_mapping = SingerMapping(data_dir=self.data_dir)
        self.singer_ids = self.singer_mapping.get_singer_ids()
        print('Training data dataset initialized from {}.'.format(self.data_dir))

    def _get_random_singer_id(self):
        return random.choice(self.singer_ids)

    def _get_random_data_path(self, singer_id):
        return random.choice(self.singer_data_paths[singer_id])

    def _get_random_segment(self, data_path, segment_duration=3.0):
        with h5py.File(data_path, 'r') as f:
            segment_mfcc_length = int(segment_duration * SAMPLE_RATE / MEL_HOP_LENGTH + 1)
            mfcc_length = f['mfcc'].shape[1]

            # Check data length
            if mfcc_length < segment_mfcc_length:
                raise ValueError('Input mfcc should have legnth >= {} but got {}'.format(segment_mfcc_length, mfcc_length))

            # Segment mfcc
            mfcc_start_idx = random.randint(0, mfcc_length-segment_mfcc_length)
            segmented_mfcc = f['mfcc'][:, mfcc_start_idx:mfcc_start_idx+segment_mfcc_length]
            return segmented_mfcc

    def __getitem__(self, idx):
        # From a random singer's random audio data extract a random segment (default 1.0 second)
        singer_id = self._get_random_singer_id()
        data_path = self._get_random_data_path(singer_id)
        mfcc = self._get_random_segment(data_path)
        singer_idx = self.singer_mapping.singer_id_to_idx(singer_id)

        return (
            torch.FloatTensor(mfcc),
            torch.LongTensor([singer_idx])[0],
        )

    def __len__(self):
        return self.epoch_len
