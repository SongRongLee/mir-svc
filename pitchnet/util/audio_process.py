import warnings
import math
import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

from .const import *


def preprocess_audio_file(file_path, sr=SAMPLE_RATE, pitch_shift_semi=None):
    """
    Preprocess a given audio file.
    - pitch_shift_semi: The semitone to perform pitch shift on the raw audio
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    # Perform pitch shifting on raw audio
    if pitch_shift_semi is not None:
        print('Shifting pitch with semitone: {:.4f}'.format(pitch_shift_semi))
        y = pitch_shift(y, sr, pitch_shift_semi)

    mu_law_compressed = mu_law_encode(y)
    extracted_pitch, pitch_mask = extract_pitch(y)

    return {
        'samples': mu_law_compressed,
        'pitch': extracted_pitch,
        'pitch_mask': pitch_mask,
    }


def postprocess_audio(y, sr):
    """
    Perform post-processing on a given audio signal.
    """
    y = librosa.resample(y, sr, OUTPUT_SAMPLE_RATE)
    return y, OUTPUT_SAMPLE_RATE


def augment_audio_file(file_path, sr=SAMPLE_RATE, aug_type='pitchnet'):
    """
    Augment a given audio file.
        - aug_type: 
            - 'pitchnet' for x4  pitchnet augmentation
            - 'pitch_aug' for pitch augmentation
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)
    if aug_type == 'pitchnet':
        back_y = np.flip(y)
        phase_y = y * -1.0
        back_phase_y = back_y * -1.0
        return {
            'sr': sr,
            'data': {
                'original': y,
                'aug_back': back_y,
                'aug_phase': phase_y,
                'aug_back_phase': back_phase_y,
            }
        }
    elif aug_type == 'pitch_aug':
        y_pm6 = pitch_shift(y, sr, -6)
        y_pm3 = pitch_shift(y, sr, -3)
        y_pa3 = pitch_shift(y, sr, 3)
        y_pa6 = pitch_shift(y, sr, 6)
        return {
            'sr': sr,
            'data': {
                'original': y,
                'aug_pm6': y_pm6,
                'aug_pm3': y_pm3,
                'aug_pa3': y_pa3,
                'aug_pa6': y_pa6,
            }
        }


def pitch_shift(y, sr, semitone):
    """
    Perform pitch shifting for a given audio signal by semitone.
    """
    sound = parselmouth.Sound(y, sr)
    manipulation = call(sound, 'To Manipulation', 0.01, PITCH_FMIN, PITCH_FMAX)
    pitch_tier = call(manipulation, 'Extract pitch tier')
    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, 2 ** (semitone / 12))
    call([pitch_tier, manipulation], "Replace pitch tier")
    sound_shifted = call(manipulation, "Get resynthesis (overlap-add)")
    return sound_shifted.values[0]


def shift_pitch_range(pitch, target_pitch_range, print_info=False):
    """
    Shift given pitch seqeunce to target pitch range.
    """
    src_pitch_range = cal_pitch_range_stats(pitch)
    shift_factor = target_pitch_range['pitch_q50'] / src_pitch_range['pitch_q50']
    if print_info:
        print('Source pitch info:')
        print(src_pitch_range)
        print('Target pitch info:')
        print(target_pitch_range)
        print('Shifting pitch with factor: {:.4f}'.format(shift_factor))
    shifted_pitch = pitch * shift_factor
    return shifted_pitch


def mu_law_encode(y, mu=MU):
    """
    Encode floating values ranging from -1.0 ~ 1.0 to integer 0 ~ mu.
    """
    mu_law_compressed = librosa.mu_compress(y, mu=mu, quantize=True) + ((mu+1) // 2)
    mu_law_compressed = mu_law_compressed.astype('int16')
    return mu_law_compressed


def mu_law_decode(y, mu=MU):
    """
    Decode integer values between 0 ~ mu to values bewteen -1.0 ~ 1.0.
    """
    mu_law_expanded = librosa.mu_expand(y - ((mu+1) // 2), mu=mu, quantize=True)
    return mu_law_expanded


def extract_pitch(y, sr=SAMPLE_RATE, hop_length=PITCH_HOP_LENGTH, fmin=PITCH_FMIN, fmax=PITCH_FMAX):
    """
    Conduct pitch tracking. Return normalized pitch (0.0 ~ 1.0) and pitch mask where 1.0 corresponds to
    valid pitches and 0.0 corresponds to invalid pitches. 
    """
    pitches, _ = librosa.piptrack(y, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
    pitches[pitches == 0.0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pitches = np.nanmin(pitches, axis=0)
    pitches[np.isnan(pitches)] = 0.0  # All-nan columns result in nan pitch. Use 0.0 for these values.
    pitch_mask = np.where(pitches > 0.0, 1.0, 0.0)
    normalized_pitches = np.clip((pitches - PITCH_FMIN) / (PITCH_FMAX - PITCH_FMIN), 0.0, 1.0)
    return normalized_pitches, pitch_mask


def cal_pitch_range_stats(pitches):
    """
    Calculate pitch statistics for a given pitch sequence.
    """
    # Filter out 0.0 pitch
    pitches = np.where(pitches == 0.0, np.nan, pitches)

    return {
        'pitch_q25': np.nanquantile(pitches, 0.25),
        'pitch_q50': np.nanquantile(pitches, 0.50),
        'pitch_q75': np.nanquantile(pitches, 0.75),
        'pitch_avg': np.nanmean(pitches)
    }


def cal_shift_pitch_semi(pitch, target_pitch_range, print_info=False):
    """
    Calculate pitch shift semi tone using source pitches and target pitch range
    """
    src_pitch_range = cal_pitch_range_stats(pitch)
    shift_factor = target_pitch_range['pitch_q50'] / src_pitch_range['pitch_q50']
    if print_info:
        print('Source pitch info:')
        print(src_pitch_range)
        print('Target pitch info:')
        print(target_pitch_range)
        print('Shifting pitch with factor: {:.4f}'.format(shift_factor))
    return math.log2(shift_factor) * 12
