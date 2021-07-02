import librosa

from .const import *


def preprocess_audio_file(file_path, sr=SAMPLE_RATE):
    """
    Preprocess a given audio file.
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=MEL_HOP_LENGTH)

    return {
        'mfcc': mfcc,
    }
