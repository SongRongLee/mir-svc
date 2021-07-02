import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from pitchnet.util.audio_process import (  # nopep8
    preprocess_audio_file,
    shift_pitch_range,
)


def ncc_score(original_path, converted_path, target_pitch_range=None):
    """Calculate normalized cross-correlation score."""
    origin_pitch = preprocess_audio_file(original_path)['pitch']
    if target_pitch_range is not None:
        origin_pitch = shift_pitch_range(origin_pitch, target_pitch_range)
    converted_pitch = preprocess_audio_file(converted_path)['pitch']
    a = (origin_pitch - np.mean(origin_pitch)) / np.std(origin_pitch)
    b = (converted_pitch - np.mean(converted_pitch)) / np.std(converted_pitch)
    ncc = np.correlate(a, b) / len(origin_pitch)
    return ncc
