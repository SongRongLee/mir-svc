import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.tensor_util import *
from .encoder import Encoder
from .wavenet_decoder import WaveNet
from .singer_classifier import SingerClassifier
from .pitch_regressor import PitchRegressor


class PitchNet(nn.Module):
    def __init__(self, singer_num, latent_d=64, singer_d=64):
        super(PitchNet, self).__init__()
        self.singer_embedding = nn.Embedding(singer_num, singer_d)
        self.encoder = Encoder(latent_d=latent_d)
        self.decoder = WaveNet(latent_d=latent_d+singer_d+1)
        self.singer_classifier = SingerClassifier(singer_num=singer_num, input_channels=latent_d)
        self.pitch_regressor = PitchRegressor(input_channels=latent_d)

        self.lambda_coef = 0.01
        self.mu_coef = 0.1

    def cal_singer_loss(self, input, target):
        # print(input.shape, target.shape)
        # [2, 12], [2]
        return F.cross_entropy(input, target)

    def cal_pitch_loss(self, input, target, mask):
        input = linear_upsample(input, target.shape[1]).squeeze(1)
        # print(input.shape, target.shape)
        # [2, 161], [2, 161]
        mask_sum = torch.sum(mask)
        if mask_sum == 0.0:
            return F.mse_loss(input*mask, target*mask, reduction='sum')
        return F.mse_loss(input*mask, target*mask, reduction='sum') / mask_sum

    def cal_rec_loss(self, input, target):
        target = target.long()
        # print(input.shape, target.shape)
        # [2, 256, 16000], [2, 16000]
        return F.cross_entropy(input, target)

    def forward(self, samples, pitch, pitch_mask, singer_idx, mode, **args):
        """
        mode:
            - 'full': Forward whole model
            - 'adv': Forward encoder and adversarial networks
            - 'enc': Forward encoder only
            - 'ae': Forward encoder and decoder
        args
            - emb: For 'enc' mode. Set to True to provide custom singer embedding
            - singer_v: For 'enc' mode. Provided singer embedding
            - bt: For 'ae' mode. Set to True to provide teacher forcing samples
            - teacher_samples: For 'ae' mode. Provided teacher forcing samples
        Return:
            - 'loss_s': Singer classification loss
            - 'loss_p': Pitch regression loss
            - 'loss_adv': Adversarial loss
            - 'loss_rec': Reconstruction loss
            - 'loss_total': Total loss
            - 'enc_out': Encoder output
            - 'dec_out': Decoder output
            - 'decoder_c': Condition vectors for decoder
        """
        if mode == 'enc':
            if 'emb' in args and args['emb']:
                singer_v = args['singer_v'].unsqueeze(2)
            else:
                singer_v = self.singer_embedding(singer_idx).unsqueeze(2)

            # Forward encoder
            enc_out = self.encoder(samples)

            # Build condition vector
            enc_upsampled = nearest_upsample(enc_out, samples.shape[1])
            singer_v_upsampled = nearest_upsample(singer_v, samples.shape[1])
            pitch_upsampled = linear_upsample(pitch, samples.shape[1])
            decoder_c = torch.cat((enc_upsampled, singer_v_upsampled, pitch_upsampled), dim=1)

            return {
                'enc_out': enc_out,
                'decoder_c': decoder_c,
            }
        elif mode == 'adv':
            # Forward encoder and adversarial networks
            with torch.no_grad():
                enc_out = self.encoder(samples)
            s_out = self.singer_classifier(enc_out)
            p_out = self.pitch_regressor(enc_out)

            # Calculate loss
            loss_s = self.cal_singer_loss(s_out, singer_idx).unsqueeze(0)
            loss_p = self.cal_pitch_loss(p_out, pitch, pitch_mask).unsqueeze(0)
            loss_adv = self.lambda_coef*loss_s + self.mu_coef*loss_p

            return {
                'loss_s': loss_s,
                'loss_p': loss_p,
                'loss_adv': loss_adv,
                'enc_out': enc_out,
            }
        elif mode == 'ae':
            # Forward encoder
            singer_v = self.singer_embedding(singer_idx).unsqueeze(2)
            enc_out = self.encoder(samples)

            # Build condition vector
            enc_upsampled = nearest_upsample(enc_out, samples.shape[1])
            singer_v_upsampled = nearest_upsample(singer_v, samples.shape[1])
            pitch_upsampled = linear_upsample(pitch, samples.shape[1])
            decoder_c = torch.cat((enc_upsampled, singer_v_upsampled, pitch_upsampled), dim=1)

            # Forward decoder
            if 'bt' in args and args['bt']:
                teacher_samples = args['teacher_samples']
            else:
                teacher_samples = samples
            dec_out = self.decoder(teacher_samples, decoder_c)

            # Calculate loss
            loss_rec = self.cal_rec_loss(dec_out, teacher_samples).unsqueeze(0)

            return {
                'loss_rec': loss_rec,
            }
        else:
            # Forward encoder and adversarial networks
            singer_v = self.singer_embedding(singer_idx).unsqueeze(2)
            enc_out = self.encoder(samples)
            s_out = self.singer_classifier(enc_out)
            p_out = self.pitch_regressor(enc_out)
            # print(enc_out.shape, s_out.shape, p_out.shape, singer_v.shape)
            # [2, 64, 20], [2, 12], [2, 1, 20], [2, 64, 1]

            # Build condition vector
            enc_upsampled = nearest_upsample(enc_out, samples.shape[1])
            singer_v_upsampled = nearest_upsample(singer_v, samples.shape[1])
            pitch_upsampled = linear_upsample(pitch, samples.shape[1])
            decoder_c = torch.cat((enc_upsampled, singer_v_upsampled, pitch_upsampled), dim=1)
            # print(enc_upsampled.shape, singer_v_upsampled.shape, pitch_upsampled.shape, decoder_c.shape)
            # [2, 64, 16000], [2, 64, 16000], [2, 1, 16000], [2, 129, 16000]

            # Forward decoder
            dec_out = self.decoder(samples, decoder_c)
            # print(dec_out.shape)
            # [2, 256, 16000]

            # Calculate loss
            loss_s = self.cal_singer_loss(s_out, singer_idx).unsqueeze(0)
            loss_p = self.cal_pitch_loss(p_out, pitch, pitch_mask).unsqueeze(0)
            loss_adv = self.lambda_coef*loss_s + self.mu_coef*loss_p
            loss_rec = self.cal_rec_loss(dec_out, samples).unsqueeze(0)
            loss_total = loss_rec - loss_adv

            return {
                'loss_s': loss_s,
                'loss_p': loss_p,
                'loss_adv': loss_adv,
                'loss_rec': loss_rec,
                'loss_total': loss_total,
                'enc_out': enc_out,
                'dec_out': dec_out,
            }
