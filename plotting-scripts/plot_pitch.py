import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from pitchnet.util.audio_process import preprocess_audio_file  # nopep8
from pitchnet.util.const import PITCH_FMIN, PITCH_FMAX  # nopep8


def main(args):
    result_dir = Path('./plotting-scripts/plotting-results/')
    result_dir.mkdir(parents=True, exist_ok=True)
    image_path = result_dir / 'pitch.png'
    src_file = Path(args.src_file)

    # Calculate pitch
    processed_data = preprocess_audio_file(src_file)
    pitch = processed_data['pitch'] * (PITCH_FMAX - PITCH_FMIN) + PITCH_FMIN

    # Plot
    plt.plot(pitch, label=src_file.name)
    plt.title('Pitch vs. Time')
    plt.xlabel('Time step')
    plt.ylabel('Pitch')
    plt.legend()

    plt.savefig(image_path)

    print('Plotting done. Image saved to {}'.format(image_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src_file')

    args = parser.parse_args()

    main(args)
