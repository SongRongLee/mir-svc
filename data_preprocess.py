import argparse
import h5py
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path


def pitchnet_parallel_work(args):
    from pitchnet.util.audio_process import preprocess_audio_file
    audio_path = args['audio_path']
    output_data_path = args['output_data_path']
    singer_id = args['singer_id']

    processed_data = preprocess_audio_file(audio_path)
    with h5py.File(output_data_path, 'w') as f:
        f.attrs['singer_id'] = singer_id
        f.create_dataset('samples', data=processed_data['samples'], dtype='int16')
        f.create_dataset('pitch', data=processed_data['pitch'], dtype='float32')
        f.create_dataset('pitch_mask', data=processed_data['pitch_mask'], dtype='float32')


def singer_classifier_parallel_work(args):
    from singer_classifier.util.audio_process import preprocess_audio_file
    audio_path = args['audio_path']
    output_data_path = args['output_data_path']
    singer_id = args['singer_id']

    processed_data = preprocess_audio_file(audio_path)
    with h5py.File(output_data_path, 'w') as f:
        f.attrs['singer_id'] = singer_id
        f.create_dataset('mfcc', data=processed_data['mfcc'], dtype='float32')


def main(args):
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    model = args.model
    parallel_work_args = []

    # Iterate each singer
    print('Start data preprocess for {}'.format(model))
    for singer_path in raw_dir.iterdir():
        singer_id = singer_path.stem
        # Iterate each file
        for audio_path in singer_path.iterdir():
            output_singer_dir = output_dir / singer_id
            output_data_path = output_singer_dir / (audio_path.stem + '.h5')
            output_singer_dir.mkdir(parents=True, exist_ok=True)
            parallel_work_args.append({
                'audio_path': audio_path,
                'output_data_path': output_data_path,
                'singer_id': singer_id
            })

    if model == 'singer_classifier':
        parallel_work = singer_classifier_parallel_work
    else:
        parallel_work = pitchnet_parallel_work

    with mp.Pool(mp.cpu_count()) as pool:
        # Parallel work for each audio file
        list(tqdm(pool.imap(parallel_work, parallel_work_args), total=len(parallel_work_args)))

    print('Data preprocess done. Files written to {}'.format(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir')
    parser.add_argument('output_dir')
    parser.add_argument(
        '--model',
        choices=['pitchnet', 'singer_classifier'],
        default='pitchnet'
    )

    args = parser.parse_args()

    main(args)
