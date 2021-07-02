import argparse
from pathlib import Path
import soundfile as sf

from pitchnet import PitchNetConvertor
from pitchnet.util.audio_process import postprocess_audio


def main(args):
    src_file = Path(args.src_file)
    target_dir = Path(args.target_dir)

    # Conversion
    convertor = PitchNetConvertor(args.model_path)
    if args.two_phase:
        y, sr = convertor.two_phase_convert(
            src_file=src_file,
            singer_id=args.singer_id,
            train_data_dir=args.train_data_dir,
            pitch_shift=args.pitch_shift
        )
    else:
        y, sr = convertor.convert(
            src_file=src_file,
            singer_id=args.singer_id,
            pitch_shift=args.pitch_shift
        )

    # Post processing
    y, sr = postprocess_audio(y, sr)

    # Save the converted audio
    target_dir.mkdir(parents=True, exist_ok=True)
    converted_filename = target_dir / '{}_{}{}{}.wav'.format(
        src_file.stem,
        args.singer_id,
        '_PS_{}'.format(args.pitch_shift) if args.pitch_shift is not None else '',
        '_TP' if args.two_phase else ''
    )
    sf.write(converted_filename, y, sr)
    print('Conversion done. File saved to {}'.format(converted_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_file')
    parser.add_argument('target_dir')
    parser.add_argument('singer_id')
    parser.add_argument('model_path')
    parser.add_argument('--pitch-shift')
    parser.add_argument('--two-phase', action='store_true', help='Enable two-phase conversion')
    parser.add_argument('--train-data-dir', help='Required for two-phase conversion')

    args = parser.parse_args()

    main(args)
