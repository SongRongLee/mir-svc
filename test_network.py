import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def main(args):
    if args.target_net == 'pitchnet.net.pitchnet':
        from pitchnet.net.pitchnet import PitchNet
        print('---------PitchNet---------')
        samples = torch.randint(0, 256, [2, 16000]).float().cuda()
        pitch = torch.rand([2, 161]).cuda()
        pitch_mask = torch.randint(0, 2, [2, 161]).float().cuda()
        singer_idx = torch.LongTensor([4, 5]).cuda()

        pitchnet = nn.DataParallel(PitchNet(singer_num=12)).cuda()
        output = pitchnet(samples, pitch, pitch_mask, singer_idx, mode='full')
        print(output)
        output = pitchnet(samples, pitch, pitch_mask, singer_idx, mode='enc')
        print(output)
        output = pitchnet(samples, pitch, pitch_mask, singer_idx, mode='adv')
        print(output)
        output = pitchnet(samples, pitch, pitch_mask, singer_idx, mode='ae')
        print(output)

    elif args.target_net == 'pitchnet.net.encoder':
        from pitchnet.net.encoder import Encoder
        print('---------Encoder---------')
        encoder = Encoder().cuda()
        summary(encoder, (16000,))

    elif args.target_net == 'pitchnet.net.wavenet_decoder':
        from pitchnet.net.wavenet_decoder import WaveNet
        print('---------WaveNet---------')
        wavenet = WaveNet().cuda()
        summary(wavenet, [(16000,), (129, 16000)])

    elif args.target_net == 'pitchnet.net.singer_classifier':
        from pitchnet.net.singer_classifier import SingerClassifier
        print('---------SingerClassifier---------')
        singer_classifier = SingerClassifier(singer_num=12).cuda()
        summary(singer_classifier, (64, 20))

    elif args.target_net == 'pitchnet.net.pitch_regressor':
        from pitchnet.net.pitch_regressor import PitchRegressor
        print('---------PitchRegressor---------')
        pitch_regressor = PitchRegressor().cuda()
        summary(pitch_regressor, (64, 20))

    elif args.target_net == 'singer_classifier.net.singer_classifier':
        from singer_classifier.net.singer_classifier import SingerClassifier
        print('---------SingerClassifier---------')
        singer_classifier = SingerClassifier(singer_num=12).cuda()
        summary(singer_classifier, (20, 301))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('target_net', choices=[
        'pitchnet.net.pitchnet',
        'pitchnet.net.encoder',
        'pitchnet.net.wavenet_decoder',
        'pitchnet.net.singer_classifier',
        'pitchnet.net.pitch_regressor',
        'singer_classifier.net.singer_classifier',
    ])

    args = parser.parse_args()

    main(args)
