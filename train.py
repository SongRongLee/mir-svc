import argparse


def main(args):
    if args.model == 'pitchnet':
        from pitchnet import PitchNetConvertor
        convertor = PitchNetConvertor(args.model_path, args.train_data_dir, args.model_dir)
        convertor.fit(
            batch_size=args.batch_size,
            total_steps=args.total_steps,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            save_model_every=args.save_model_every,
            bt=args.bt,
            bt_start=args.bt_start,
            bt_every=args.bt_every,
            bt_new=args.bt_new,
            g_batch_size=args.g_batch_size,
        )
    elif args.model == 'singer_classifier':
        from singer_classifier import SingerPredictor
        predictor = SingerPredictor(args.model_path, args.train_data_dir, args.model_dir)
        predictor.fit(
            batch_size=16,
            total_steps=50000,
            lr=1e-3,
            lr_step_size=1000,
            save_model_every=10000,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_dir')
    parser.add_argument('model_dir')
    parser.add_argument(
        '--model',
        choices=['pitchnet', 'singer_classifier'],
        default='pitchnet'
    )
    parser.add_argument('--model-path')
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--total-steps', default=300000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr-step-size', default=1000, type=int)
    parser.add_argument('--save-model-every', default=10000, type=int)
    parser.add_argument('--bt', action='store_true', help='Enable backtranslation')
    parser.add_argument(
        '--bt-start',
        default=200000,
        type=int,
        help='When to start phase two backtranslation training'
    )
    parser.add_argument(
        '--bt-every',
        default=2000,
        type=int,
        help='Number of steps to trigger backtranslation'
    )
    parser.add_argument(
        '--bt-new',
        default=96,
        type=int,
        help='Number of new segments generated from backtranslation'
    )
    parser.add_argument(
        '--g-batch-size',
        default=96,
        type=int,
        help='Batch size during backtranslation generation'
    )

    args = parser.parse_args()

    main(args)
