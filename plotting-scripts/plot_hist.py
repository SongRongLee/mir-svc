import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import librosa


def main(args):
    result_dir = Path('./plotting-scripts/plotting-results/')
    result_dir.mkdir(parents=True, exist_ok=True)
    image_path = result_dir / 'hist.png'
    raw_dir = Path(args.raw_dir)

    durations = []
    # Iterate each singer
    for singer_path in raw_dir.iterdir():
        singer_id = singer_path.stem
        # Iterate each file
        for audio_path in singer_path.iterdir():
            durations.append(librosa.get_duration(filename=audio_path))

    # Print stats
    print('Duration total: {:.2f}'.format(np.sum(durations)))
    print('Duration average: {:.2f}'.format(np.average(durations)))

    # Plot
    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.hist(durations)
    plt.title('Data duration histogram')
    plt.xlabel('Duration (sec)')
    plt.ylabel('Count')

    plt.savefig(image_path)

    print('Plotting done. Image saved to {}'.format(image_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir')

    args = parser.parse_args()

    main(args)
