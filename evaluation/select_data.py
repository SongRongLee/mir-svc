import argparse
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import multiprocessing as mp
import random
import librosa


def parallel_work(args):
    audio_path = args['audio_path']
    output_singer_dir = args['output_singer_dir']
    seg_len = args['seg_len']

    # Get random seg_len segment
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    sample_length = y.shape[0]
    segment_sample_length = int(seg_len * sr)
    # Check data length
    if sample_length < segment_sample_length:
        raise ValueError('Input samples should have legnth >= {} but got {}'.format(segment_sample_length, sample_length))
    sample_start_idx = random.randint(0, sample_length-segment_sample_length)
    segmented_samples = y[sample_start_idx:sample_start_idx+segment_sample_length]

    # Write file
    sf.write(
        output_singer_dir / '{}.wav'.format(audio_path.stem),
        segmented_samples,
        sr,
    )


def main(args):
    print('Start data selection...')

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    parallel_work_args = []

    # Iterate each singer
    for singer_path in raw_dir.iterdir():
        singer_id = singer_path.stem
        # Iterate each file
        for audio_path in singer_path.iterdir():
            output_singer_dir = output_dir / singer_id
            output_singer_dir.mkdir(parents=True, exist_ok=True)
            parallel_work_args.append({
                'audio_path': audio_path,
                'output_singer_dir': output_singer_dir,
                'seg_len': args.seg_len,
            })

    with mp.Pool(mp.cpu_count()) as pool:
        # Parallel work for each audio file
        list(tqdm(pool.imap(parallel_work, parallel_work_args), total=len(parallel_work_args)))

    print('Data selection done. Files written to {}'.format(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--seg-len', default=3, type=int)

    args = parser.parse_args()

    main(args)
