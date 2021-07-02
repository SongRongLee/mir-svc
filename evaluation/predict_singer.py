import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))
from singer_classifier import SingerPredictor  # nopep8


def predict_singers(audio_paths, model_path):
    predictor = SingerPredictor(model_path)
    return predictor.predict(audio_paths)


def main(args):
    audio_paths = [Path(audio_path) for audio_path in args.audio_paths]
    model_path = Path(args.model_path)

    predicted_ids = predict_singers(audio_paths, model_path)
    print('Predicted singers: {}'.format(predicted_ids))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--audio-paths', nargs='+')

    args = parser.parse_args()

    main(args)
