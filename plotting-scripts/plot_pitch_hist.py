import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from pitchnet.util.audio_process import preprocess_audio_file  # nopep8
from pitchnet.util.const import PITCH_FMIN, PITCH_FMAX  # nopep8


def is_male(singer_id):
    if singer_id in ['JTAN', 'SAMF', 'VKOW', 'JLEE', 'ZHIY', 'KENN']:
        return True
    else:
        return False


def parallel_work(args):
    audio_path = args['audio_path']
    singer_id = args['singer_id']

    # Calculate pitch
    processed_data = preprocess_audio_file(audio_path)
    pitch = processed_data['pitch'] * (PITCH_FMAX - PITCH_FMIN) + PITCH_FMIN
    return {
        'pitch': pitch,
        'is_male': is_male(singer_id)
    }


def main(args):
    result_dir = Path('./plotting-scripts/plotting-results/')
    result_dir.mkdir(parents=True, exist_ok=True)
    image_path = result_dir / 'pitch_hist.png'
    pitches_cache_path = result_dir / 'pitch_cache.npy'
    raw_dir = Path(args.raw_dir)
    parallel_work_args = []

    if pitches_cache_path.is_file():
        pitches = np.load(pitches_cache_path, allow_pickle=True)
    else:
        # Iterate each singer
        for singer_path in raw_dir.iterdir():
            singer_id = singer_path.stem
            # Iterate each file
            for audio_path in singer_path.iterdir():
                parallel_work_args.append({
                    'audio_path': audio_path,
                    'singer_id': singer_id
                })

        with mp.Pool(mp.cpu_count()) as pool:
            # Parallel work for each audio file
            results = list(tqdm(pool.imap(parallel_work, parallel_work_args), total=len(parallel_work_args)))

        male_pitches = []
        female_pitches = []
        for result in results:
            if result['is_male']:
                male_pitches.append(result['pitch'])
            else:
                female_pitches.append(result['pitch'])
        pitches = np.array([np.concatenate(male_pitches, axis=0), np.concatenate(female_pitches, axis=0)])
        with open(pitches_cache_path, 'wb') as f:
            np.save(f, pitches)

    # Plot
    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.hist(pitches[0], bins=100, range=(100, 600), label='Male', color='#1f77b4')
    plt.hist(pitches[1], bins=100, range=(100, 600), label='Female', color='#ff7f0ee0')
    plt.legend()
    plt.title('Pitch histogram')
    plt.xlabel('Pitch (Hz)')
    plt.ylabel('Count')

    plt.savefig(image_path)

    print('Plotting done. Image saved to {}'.format(image_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir')

    args = parser.parse_args()

    main(args)
