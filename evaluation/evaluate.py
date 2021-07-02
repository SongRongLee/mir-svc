import os
import sys
import argparse
import copy
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from config import conversion_mappings  # nopep8
from util import ncc_score  # nopep8
from predict_singer import predict_singers  # nopep8
from pitchnet import PitchNetConvertor  # nopep8
from pitchnet.util.audio_process import (  # nopep8
    preprocess_audio_file,
    postprocess_audio,
    shift_pitch_range,
)


def parallel_preprocess(args):
    audio_path = args['audio_path']
    singer_id = args['singer_id']

    processed_data = preprocess_audio_file(audio_path)
    processed_data['audio_path'] = audio_path
    processed_data['singer_id'] = singer_id
    return processed_data


def parallel_eval(args):
    original_path = args['original_path']
    converted_path = args['converted_path']
    target_singer_id = args['target_singer_id']
    predicted_id = args['predicted_id']
    target_pitch_range = args['target_pitch_range']

    ncc = ncc_score(original_path, converted_path, target_pitch_range)
    sca = 1.0 if target_singer_id == predicted_id else 0.0
    return {
        'ncc': ncc,
        'sca': sca,
    }


def main(args):
    print('Start evaluation...')

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)
    sc_model_path = Path(args.sc_model_path)
    conversion_mapping = conversion_mappings[args.mapping]
    pitch_shift = args.pitch_shift
    two_phase = args.two_phase
    train_data_dir = args.train_data_dir
    parallel_preprocess_args = []
    convertor = PitchNetConvertor(model_path)

    # Preprocess audio file
    print('Preprocessing audio files...')
    for singer_path in raw_dir.iterdir():
        singer_id = singer_path.stem
        # Iterate each file
        for audio_path in singer_path.iterdir():
            parallel_preprocess_args.append({
                'audio_path': audio_path,
                'singer_id': singer_id
            })

    if two_phase:
        processed_datas = copy.deepcopy(parallel_preprocess_args)
    else:
        with mp.Pool(mp.cpu_count()) as pool:
            # Parallel work for each audio file
            processed_datas = list(tqdm(pool.imap(parallel_preprocess, parallel_preprocess_args), total=len(parallel_preprocess_args)))

    # Make conversion pairs
    input_datas = []
    target_singer_ids = []
    conversion_records = []
    for processed_data in processed_datas:
        for target_singer in conversion_mapping[processed_data['singer_id']]:
            target_pitch_range = None
            if not two_phase:
                if pitch_shift:
                    target_pitch_range = convertor.pitch_ranges[target_singer]
                    processed_data['pitch'] = shift_pitch_range(processed_data['pitch'], target_pitch_range)
                input_datas.append(processed_data)
            target_singer_ids.append(target_singer)
            conversion_records.append({
                'audio_path': processed_data['audio_path'],
                'singer_id': processed_data['singer_id'],
                'target_singer_id': target_singer,
                'target_pitch_range': target_pitch_range,
            })

    # Perform conversion
    if two_phase:
        y_list = []
        for conversion_record in conversion_records:
            y, sr = convertor.two_phase_convert(
                src_file=conversion_record['audio_path'],
                singer_id=conversion_record['target_singer_id'],
                train_data_dir=train_data_dir
            )
            y_list.append(y)
    else:
        y_list, sr = convertor.bulk_convert_data(
            input_datas=input_datas,
            target_singer_ids=target_singer_ids
        )

    # Save the converted audio
    for conversion_record, y in zip(conversion_records, y_list):
        # Post processing
        y_processed, new_sr = postprocess_audio(y, sr)

        # Save the converted audio
        target_dir = output_dir / conversion_record['singer_id']
        target_dir.mkdir(parents=True, exist_ok=True)
        converted_filename = target_dir / '{}_{}.wav'.format(conversion_record['audio_path'].stem, conversion_record['target_singer_id'])
        conversion_record['converted_path'] = converted_filename
        sf.write(converted_filename, y_processed, new_sr)

    # Perform singer classification
    predicted_ids = predict_singers(
        [conversion_record['converted_path'] for conversion_record in conversion_records],
        sc_model_path)
    for conversion_record, predicted_id in zip(conversion_records, predicted_ids):
        conversion_record['predicted_id'] = predicted_id

    # Calculate evaluation score
    print('Evaluating...')
    parallel_eval_args = []
    for conversion_record in conversion_records:
        parallel_eval_args.append({
            'original_path': conversion_record['audio_path'],
            'converted_path': conversion_record['converted_path'],
            'target_singer_id': conversion_record['target_singer_id'],
            'predicted_id': conversion_record['predicted_id'],
            'target_pitch_range': conversion_record['target_pitch_range'],
        })

    with mp.Pool(mp.cpu_count()) as pool:
        # Parallel work for each audio file
        eval_results = list(tqdm(pool.imap(parallel_eval, parallel_eval_args), total=len(parallel_eval_args)))

    ncc_con = []
    ncc_rec = []
    sca_con = []
    sca_rec = []
    for conversion_record, eval_result in zip(conversion_records, eval_results):
        if conversion_record['singer_id'] == conversion_record['target_singer_id']:
            ncc_rec.append(eval_result['ncc'])
            sca_rec.append(eval_result['sca'])
        else:
            ncc_con.append(eval_result['ncc'])
            sca_con.append(eval_result['sca'])
    avg_ncc_con = np.mean(ncc_con)
    avg_ncc_rec = np.mean(ncc_rec)
    avg_sca_con = np.mean(sca_con)
    avg_sca_rec = np.mean(sca_rec)

    # Print results
    print('Conversion NCC:      {:.4f}'.format(avg_ncc_con))
    print('Reconstruction NCC:  {:.4f}'.format(avg_ncc_rec))
    print('Conversion SCA:      {:.4f}'.format(avg_sca_con))
    print('Reconstruction SCA:  {:.4f}'.format(avg_sca_rec))
    print('Evaluation done. Converted files are saved to {}'.format(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir')
    parser.add_argument('output_dir')
    parser.add_argument('model_path')
    parser.add_argument('sc_model_path')
    parser.add_argument('mapping', choices=[
        'all',
        'male',
        'female-male',
        'female',
        'ext',
        'mos',
    ])
    parser.add_argument('--pitch-shift', action='store_true', help='Enable pitch shifting')
    parser.add_argument('--two-phase', action='store_true', help='Enable two-phase conversion')
    parser.add_argument('--train-data-dir', help='Required for two-phase conversion')

    args = parser.parse_args()

    main(args)
