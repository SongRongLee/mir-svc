import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def main(args):
    result_dir = Path('./plotting-scripts/plotting-results/')
    result_dir.mkdir(parents=True, exist_ok=True)
    image_path = result_dir / 'spec.png'
    src_file = Path(args.src_file)

    # Process audio file
    y, sr = librosa.load(src_file)
    D = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plot
    plt.figure()
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    plt.savefig(image_path)

    print('Plotting done. Image saved to {}'.format(image_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src_file')

    args = parser.parse_args()

    main(args)
