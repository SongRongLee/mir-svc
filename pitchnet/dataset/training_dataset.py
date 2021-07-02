import torch
import random
import h5py
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

from ..util.singer_mapping import SingerMapping
from ..util.const import SAMPLE_RATE, PITCH_HOP_LENGTH
from ..util.audio_process import cal_pitch_range_stats


class TrainingDataset(Dataset):
    def __init__(self, data_dir, singer_mapping=None, epoch_len=300000):
        self.epoch_len = epoch_len
        self.data_dir = Path(data_dir)
        self.singer_paths = list(self.data_dir.iterdir())
        self.singer_data_paths = {singer_path.stem: list(singer_path.iterdir()) for singer_path in self.singer_paths}
        if singer_mapping is not None:
            self.singer_mapping = singer_mapping
        else:
            self.singer_mapping = SingerMapping(data_dir=self.data_dir)
        self.singer_ids = self.singer_mapping.get_singer_ids()
        print('Training data dataset initialized from {}'.format(self.data_dir))

    def _get_random_singer_id(self):
        return random.choice(self.singer_ids)

    def _get_random_data_path(self, singer_id):
        return random.choice(self.singer_data_paths[singer_id])

    def _get_random_segment(self, data_path, segment_duration=1.0):
        with h5py.File(data_path, 'r') as f:
            segment_sample_length = int(segment_duration * SAMPLE_RATE)
            segment_pitch_length = int(segment_duration * SAMPLE_RATE / PITCH_HOP_LENGTH + 1)
            sample_length = f['samples'].shape[0]
            pitch_length = f['pitch'].shape[0]

            # Check data length
            if sample_length < segment_sample_length:
                raise ValueError('Input samples should have legnth >= {} but got {}'.format(segment_sample_length, sample_length))
            if pitch_length < segment_pitch_length:
                raise ValueError('Input pitch should have legnth >= {} but got {}'.format(segment_pitch_length, pitch_length))

            # Segment samples
            sample_start_idx = random.randint(0, sample_length-segment_sample_length)
            segmented_samples = f['samples'][sample_start_idx:sample_start_idx+segment_sample_length]

            # Segment pitch
            pitch_start_idx = sample_start_idx // PITCH_HOP_LENGTH
            segmented_pitch = f['pitch'][pitch_start_idx:pitch_start_idx+segment_pitch_length]
            segmented_pitch_mask = f['pitch_mask'][pitch_start_idx:pitch_start_idx+segment_pitch_length]

            return segmented_samples, segmented_pitch, segmented_pitch_mask

    def __getitem__(self, idx):
        # From a random singer's random audio data extract a random segment (default 1.0 second)
        singer_id = self._get_random_singer_id()
        data_path = self._get_random_data_path(singer_id)
        samples, pitch, pitch_mask = self._get_random_segment(data_path)
        singer_idx = self.singer_mapping.singer_id_to_idx(singer_id)

        return (
            torch.FloatTensor(samples),
            torch.FloatTensor(pitch),
            torch.FloatTensor(pitch_mask),
            torch.LongTensor([singer_idx])[0],
        )

    def __len__(self):
        return self.epoch_len

    def get_pitch_ranges(self):
        """Get pitch statistics for each singer."""
        pitch_ranges = {}
        for singer_id in self.singer_ids:
            data_paths = self.singer_data_paths[singer_id]
            pitches = []
            for data_path in data_paths:
                with h5py.File(data_path, 'r') as f:
                    pitches.append((f['pitch'][()]))
            pitches = np.concatenate(pitches, axis=0)
            pitch_ranges[singer_id] = cal_pitch_range_stats(pitches)

        return pitch_ranges
