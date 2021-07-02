import argparse
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import multiprocessing as mp

from pitchnet.util.audio_process import augment_audio_file


def parallel_work(args):
    audio_path = args['audio_path']
    output_singer_dir = args['output_singer_dir']
    aug_type = args['aug_type']

    processed_data = augment_audio_file(audio_path, aug_type=aug_type)
    sr = processed_data['sr']
    for aug_name, aug_data in processed_data['data'].items():
        sf.write(
            output_singer_dir / '{}_{}.wav'.format(audio_path.stem, aug_name),
            aug_data,
            sr,
        )


def main(args):
    print('Start data augmentation...')

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    aug_type = args.aug_type
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
                'aug_type': aug_type
            })

    with mp.Pool(mp.cpu_count()) as pool:
        # Parallel work for each audio file
        list(tqdm(pool.imap(parallel_work, parallel_work_args), total=len(parallel_work_args)))

    print('Data augmentation done. Files written to {}'.format(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--aug-type',
                        choices=['pitchnet', 'pitch_aug'],
                        default='pitchnet')

    args = parser.parse_args()

    main(args)
