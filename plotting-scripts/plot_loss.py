import torch
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from plotting_util import moving_avg


def main(args):
    checkpoint = torch.load(args.checkpoint_path)
    losses = checkpoint['losses']
    result_dir = Path('./plotting-scripts/plotting-results/')
    result_dir.mkdir(parents=True, exist_ok=True)
    image_path = result_dir / 'losses.png'

    # Form data matrix
    losses_list = []
    for loss in losses:
        losses_list.append([loss['loss_s'],
                            loss['loss_p'],
                            loss['loss_adv'],
                            loss['loss_rec'],
                            loss['loss_total']])
    losses_matrix = np.array(losses_list)

    # Apply moving average
    losses_matrix = moving_avg(losses_matrix, args.window_size)

    steps = [loss['current_step'] for loss in losses]
    # Singer classification loss
    if 's' in args.loss_types:
        plt.plot(steps, losses_matrix[:, 0], label='loss_s')
    # Pitch regression loss
    if 'p' in args.loss_types:
        plt.plot(steps, losses_matrix[:, 1], label='loss_p')
    # Adversarial loss
    if 'adv' in args.loss_types:
        plt.plot(steps, losses_matrix[:, 2], label='loss_adv')
    # Reconstruction loss
    if 'rec' in args.loss_types:
        plt.plot(steps, losses_matrix[:, 3], label='loss_rec')
    # Total loss
    if 'total' in args.loss_types:
        plt.plot(steps, losses_matrix[:, 4], label='loss_total')

    # Common
    plt.title('Loss vs. Step')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_path)

    print('Plotting done. Image saved to {}'.format(image_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path')
    parser.add_argument('--window-size', default=1000, type=int)
    parser.add_argument(
        '--loss-types',
        choices=['s', 'p', 'adv', 'rec', 'total'],
        nargs="+",
        default=['s', 'p', 'adv', 'rec', 'total']
    )

    args = parser.parse_args()

    main(args)
