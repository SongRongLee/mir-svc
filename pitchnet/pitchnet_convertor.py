import time
import copy
import random
from pathlib import Path
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .net import PitchNet
from .util.singer_mapping import SingerMapping
from .util.const import SAMPLE_RATE
from .util.audio_process import (
    preprocess_audio_file,
    mu_law_decode,
    shift_pitch_range,
    cal_shift_pitch_semi
)
from .util import rgetattr
from .dataset import TrainingDataset
from .wavenet_generator import WaveNetGenerator


class PitchNetConvertor:
    def __init__(self, model_path=None, train_data_dir=None, model_dir=None):
        # Load checkpoint and singer mapping
        self.model_path = model_path
        self.train_data_dir = train_data_dir
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.loaded_checkpoint = None
        self.pitch_ranges = None
        if self.model_path is not None:
            self.loaded_checkpoint = torch.load(self.model_path)

            # singer_ids loading or injection
            if 'singer_ids' in self.loaded_checkpoint:
                self.singer_mapping = SingerMapping(singer_ids=self.loaded_checkpoint['singer_ids'])
            else:
                # Inject singer_ids info into checkpoint
                self.singer_mapping = SingerMapping(data_dir=self.train_data_dir)
                self.loaded_checkpoint['singer_ids'] = self.singer_mapping.get_singer_ids()
                torch.save(self.loaded_checkpoint, self.model_path)
                print('singer_ids injected.')

            # pitch_ranges loading or injection
            if 'pitch_ranges' in self.loaded_checkpoint:
                self.pitch_ranges = self.loaded_checkpoint['pitch_ranges']
            else:
                # Inject pitch_ranges info into checkpoint
                self.training_dataset = TrainingDataset(self.train_data_dir, self.singer_mapping)
                self.pitch_ranges = self.training_dataset.get_pitch_ranges()
                self.loaded_checkpoint['pitch_ranges'] = self.pitch_ranges
                torch.save(self.loaded_checkpoint, self.model_path)
                print('pitch_ranges injected.')
        elif self.train_data_dir is not None and self.model_dir is not None:
            self.singer_mapping = SingerMapping(data_dir=self.train_data_dir)
        else:
            raise ValueError('Either model_path or (train_data_dir, model_dir) should be provided to the convertor.')
        self.singer_ids = self.singer_mapping.get_singer_ids()
        self.singer_idxs = self.singer_mapping.get_singer_idxs()

        # Initialize model
        self.pitchnet = nn.DataParallel(PitchNet(singer_num=len(self.singer_ids))).cuda()

        # Load model if checkpoint is given
        if self.loaded_checkpoint is not None:
            self.pitchnet.load_state_dict(self.loaded_checkpoint['pitchnet_state_dict'])
            print('Model read from {}.'.format(self.model_path))

        print('Convertor initialized.')

    def fit(self, **training_args):
        """
        training_args:
            - batch_size
            - total_steps
            - lr
            - lr_step_size
            - save_model_every
            - bt
            - bt_start
            - bt_every
            - bt_new
            - g_batch_size
        """
        # Set training params
        self.batch_size = training_args['batch_size']
        self.total_steps = training_args['total_steps']
        self.lr = training_args['lr']
        self.lr_step_size = training_args['lr_step_size']
        self.bt = training_args['bt']
        self.bt_start = training_args['bt_start']
        self.bt_every = training_args['bt_every']
        self.bt_new = training_args['bt_new']
        self.bt_steps = self.bt_new // self.batch_size
        self.g_batch_size = training_args['g_batch_size']

        # Loop units
        self.print_every = 100
        self.save_model_every = training_args['save_model_every']

        # Initial states
        self.current_step = 1
        self.losses = []

        # Setup optimizers and schedulers
        self.ae_optimizer = optim.Adam(
            list(self.pitchnet.module.singer_embedding.parameters()) +
            list(self.pitchnet.module.encoder.parameters()) +
            list(self.pitchnet.module.decoder.parameters()),
            lr=self.lr)
        self.singer_optimizer = optim.Adam(
            self.pitchnet.module.singer_classifier.parameters(),
            lr=1e-3)
        self.pitch_optimizer = optim.Adam(
            self.pitchnet.module.pitch_regressor.parameters(),
            lr=1e-3)
        self.ae_scheduler = optim.lr_scheduler.StepLR(self.ae_optimizer, step_size=self.lr_step_size, gamma=0.98)

        # Parse checkpoint
        if self.loaded_checkpoint is not None:
            self.ae_optimizer.load_state_dict(self.loaded_checkpoint['ae_optimizer_state_dict'])
            self.singer_optimizer.load_state_dict(self.loaded_checkpoint['singer_optimizer_state_dict'])
            self.pitch_optimizer.load_state_dict(self.loaded_checkpoint['pitch_optimizer_state_dict'])
            self.current_step = self.loaded_checkpoint['current_step']+1
            self.ae_scheduler = optim.lr_scheduler.StepLR(
                self.ae_optimizer,
                step_size=self.lr_step_size,
                gamma=0.98,
                last_epoch=self.loaded_checkpoint['current_step']-1
            )
            self.losses = self.loaded_checkpoint['losses']

        # Read datasets
        print('Creating datasets...')
        self.training_dataset = TrainingDataset(self.train_data_dir, self.singer_mapping)

        # Parse pitch ranges from the dataset
        if self.pitch_ranges is None:
            self.pitch_ranges = self.training_dataset.get_pitch_ranges()

        # Setup dataloader and initial variables
        self.train_loader = DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        start_time = time.time()
        loss_s, loss_p, loss_adv, loss_rec, loss_total = torch.zeros(5)
        bt_generated = False
        bt_data = None

        # Start training
        print('Training...')
        # torch.autograd.set_detect_anomaly(True)
        while self.current_step <= self.total_steps:
            for batch_idx, batch_data in enumerate(self.train_loader):
                if (self.bt and self.current_step > self.bt_start and
                        (self.current_step-1) % self.bt_every < self.bt_steps):
                    # Backtranslation training
                    if not bt_generated:
                        bt_data = self.generate_bt_data()
                        bt_generated = True
                        print('Backtranslation data generated. Start backtranslation for {} steps.'.format(self.bt_steps))

                    self.pitchnet.train()

                    # Parse data
                    bt_idx = (self.current_step-1) % self.bt_every
                    teacher_samples = bt_data[bt_idx][0].cuda()
                    input_samples = bt_data[bt_idx][6].cuda()
                    input_pitch = bt_data[bt_idx][1].cuda()
                    input_singer_idx = bt_data[bt_idx][3].cuda()

                    # Optimize autoenccoder
                    self.ae_optimizer.zero_grad()
                    model_output = self.pitchnet(
                        input_samples,
                        input_pitch,
                        None,
                        input_singer_idx,
                        mode='ae',
                        bt=True,
                        teacher_samples=teacher_samples
                    )
                    loss_rec = model_output['loss_rec'].mean()

                    loss_rec.backward()
                    self.ae_optimizer.step()
                    self.ae_scheduler.step()
                else:
                    # Non-backtranslation training
                    bt_generated = False
                    self.pitchnet.train()

                    # Parse data
                    input_samples = batch_data[0].cuda()
                    input_pitch = batch_data[1].cuda()
                    input_pitch_mask = batch_data[2].cuda()
                    input_singer_idx = batch_data[3].cuda()

                    # Optimize adversarial networks
                    self.ae_optimizer.zero_grad()
                    self.singer_optimizer.zero_grad()
                    self.pitch_optimizer.zero_grad()
                    model_output = self.pitchnet(input_samples, input_pitch, input_pitch_mask, input_singer_idx, mode='adv')
                    loss_adv = model_output['loss_adv'].mean()

                    loss_adv.backward()
                    self.singer_optimizer.step()
                    self.pitch_optimizer.step()

                    # Optimize whole network
                    self.ae_optimizer.zero_grad()
                    self.singer_optimizer.zero_grad()
                    self.pitch_optimizer.zero_grad()
                    model_output = self.pitchnet(input_samples, input_pitch, input_pitch_mask, input_singer_idx, mode='full')
                    loss_s = model_output['loss_s'].mean()
                    loss_p = model_output['loss_p'].mean()
                    loss_adv = model_output['loss_adv'].mean()
                    loss_rec = model_output['loss_rec'].mean()
                    loss_total = model_output['loss_total'].mean()

                    loss_total.backward()
                    self.ae_optimizer.step()
                    self.ae_scheduler.step()

                # Save losses
                self.losses.append({
                    'current_step': self.current_step,
                    'loss_s': loss_s.item(),
                    'loss_p': loss_p.item(),
                    'loss_adv': loss_adv.item(),
                    'loss_rec': loss_rec.item(),
                    'loss_total': loss_total.item(),
                })

                # Save checkpoint
                if self.current_step % self.save_model_every == 0:
                    save_dict = {
                        'pitchnet_state_dict': self.pitchnet.state_dict(),
                        'ae_optimizer_state_dict': self.ae_optimizer.state_dict(),
                        'singer_optimizer_state_dict': self.singer_optimizer.state_dict(),
                        'pitch_optimizer_state_dict': self.pitch_optimizer.state_dict(),
                        'current_step': self.current_step,
                        'losses': self.losses,
                        'singer_ids': self.singer_ids,
                        'pitch_ranges': self.pitch_ranges,
                    }
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = self.model_dir / 's_{}.tar'.format(self.current_step)
                    torch.save(save_dict, checkpoint_path)

                # Show training message
                if self.current_step % self.print_every == 0:
                    print(
                        (
                            '| Step [{:6d}/{:6d}] '
                            'loss_s {:.4f} loss_p {:.4f} loss_adv {:.4f} loss_rec {:.4f} loss_total {:.4f} '
                            'Time {:.1f}'
                        ).format(
                            self.current_step,
                            self.total_steps,
                            self.losses[-1]['loss_s'],
                            self.losses[-1]['loss_p'],
                            self.losses[-1]['loss_adv'],
                            self.losses[-1]['loss_rec'],
                            self.losses[-1]['loss_total'],
                            (time.time()-start_time) / 60,
                        )
                    )

                self.current_step += 1
                if self.current_step > self.total_steps:
                    break

        print('Training done in {:.1f} minutes.'.format((time.time()-start_time) / 60))

    def generate_bt_data(self):
        """
        Generate a temporary dataset for backtranslation.
        The dataset is of shape (batch_number, 7), corresponds to the following
        0: samples
        1: pitch
        2: pitch_mask
        3: singer_idx_A
        4: singer_idx_B
        5: singer_v_C
        6: samples_C
        """
        g_batch_number = self.bt_new // self.g_batch_size

        # Setup models
        inference_pitchnet = PitchNet(singer_num=len(self.singer_ids)).cuda()
        inference_pitchnet.load_state_dict(self.pitchnet.module.state_dict())  # Get rid of DataParallel by cloning
        singer_embedding = inference_pitchnet.singer_embedding
        decoder = copy.deepcopy(self.pitchnet.module.decoder)  # Clone a decoder for generator
        generator = WaveNetGenerator(decoder, batch_size=self.g_batch_size)

        # Load random data
        g_loader = DataLoader(
            self.training_dataset,
            batch_size=self.g_batch_size,
            num_workers=4,
            pin_memory=False,
        )
        loaded_data = []
        for b_idx, batch_data in enumerate(g_loader):
            if b_idx >= g_batch_number:
                break
            # Get random singer B different from the data singer A
            singer_idx_B_batch_data = None
            for data_idx in range(self.g_batch_size):
                singer_idx_A = batch_data[3][data_idx].item()
                excluded_list = [singer_idx for singer_idx in self.singer_idxs if singer_idx != singer_idx_A]
                singer_idx_B = random.choice(excluded_list)
                singer_idx_B_data = torch.tensor(singer_idx_B).unsqueeze(0)
                if singer_idx_B_batch_data is None:
                    singer_idx_B_batch_data = singer_idx_B_data
                else:
                    singer_idx_B_batch_data = torch.cat((singer_idx_B_batch_data, singer_idx_B_data))
            batch_data.append(singer_idx_B_batch_data)
            batch_data_cp = copy.deepcopy(batch_data)
            del batch_data
            loaded_data.append(batch_data_cp)

        # Generate fake singer embedding for each data
        singer_embedding.eval()
        with torch.no_grad():
            for b_idx, batch_data in enumerate(loaded_data):
                # Get embedding vectors for A and B
                input_singer_idx = batch_data[3].cuda()
                singer_v_A = singer_embedding(input_singer_idx)
                input_singer_idx = batch_data[4].cuda()
                singer_v_B = singer_embedding(input_singer_idx)

                # Calculate embedding for fake singer C
                merge_weight = torch.rand((self.g_batch_size, 1)).cuda()
                singer_v_C = singer_v_A * merge_weight + singer_v_B * (1.0 - merge_weight)
                batch_data.append(singer_v_C)

        # Generate new samples for singer C
        inference_pitchnet.eval()
        with torch.no_grad():
            for b_idx, batch_data in enumerate(loaded_data):
                input_samples = batch_data[0].cuda()
                input_pitch = batch_data[1].cuda()
                singer_v_C = batch_data[5].cuda()
                model_output = inference_pitchnet(
                    input_samples,
                    input_pitch,
                    None,
                    None,
                    mode='enc',
                    emb=True,
                    singer_v=singer_v_C
                )
                decoder_c = model_output['decoder_c']
                audio_data = generator.generate(decoder_c)
                batch_data.append(audio_data)

        # Re-arrange the data to apply the training batch size
        data_instances = []
        for b_idx, batch_data in enumerate(loaded_data):
            for data_idx in range(self.g_batch_size):
                # Move to cpu to prevent CUDA multiprocessing error
                data_instances.append([data_column[data_idx].cpu() for data_column in batch_data])
        bt_loader = DataLoader(
            data_instances,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=False,
            drop_last=False,
        )
        bt_data = []
        for batch_data in bt_loader:
            batch_data_cp = copy.deepcopy(batch_data)
            del batch_data
            bt_data.append(batch_data_cp)

        return bt_data

    def convert(self, src_file, singer_id, pitch_shift=None):
        """
        Perform singing voice conversion given source audio file and target singer id.
            - src_file: Path to the source audio file
            - singer_id: Singer id (name) of the target singer
            - pitch_shift: Whether or not to do pitch shifting
        """
        singer_idx = self.singer_mapping.singer_id_to_idx(singer_id)

        # Preprocess audio file
        processed_data = preprocess_audio_file(src_file)
        input_samples = torch.FloatTensor([processed_data['samples']]).cuda()
        if pitch_shift is not None:
            if pitch_shift == 'auto':
                pitch = shift_pitch_range(processed_data['pitch'], self.pitch_ranges[singer_id], print_info=True)
                input_pitch = torch.FloatTensor([pitch]).cuda()
            else:
                print('Shifting pitch with factor: {:.4f}'.format(float(pitch_shift)))
                input_pitch = torch.FloatTensor([processed_data['pitch']]).cuda() * float(pitch_shift)
        else:
            input_pitch = torch.FloatTensor([processed_data['pitch']]).cuda()
        input_singer_idx = torch.LongTensor([singer_idx]).cuda()
        batch_size = input_samples.shape[0]

        # Setup models
        inference_pitchnet = PitchNet(singer_num=len(self.singer_ids)).cuda()
        inference_pitchnet.load_state_dict(self.pitchnet.module.state_dict())  # Get rid of DataParallel by cloning
        decoder = copy.deepcopy(self.pitchnet.module.decoder)  # Clone a decoder for generator
        generator = WaveNetGenerator(decoder, batch_size=batch_size)

        # Start generate
        inference_pitchnet.eval()
        with torch.no_grad():
            model_output = inference_pitchnet(input_samples, input_pitch, None, input_singer_idx, mode='enc')
            decoder_c = model_output['decoder_c']
            audio_data = generator.generate(decoder_c).reshape(-1).cpu().numpy()

        # Convert back to audio value
        y = mu_law_decode(audio_data)

        return y, SAMPLE_RATE

    def bulk_convert_data(self, input_datas, target_singer_ids):
        """
        Perform singing voice conversion given pre-processed input data and target singer ids.
        This function is used for evaluation purpose and requires the source data have the same length.
            - input_datas: List of processed audio data
            - target_singer_ids: Singer ids (names) of the corresponding target singers
        """
        target_singer_idxs = [self.singer_mapping.singer_id_to_idx(target_singer_id) for target_singer_id in target_singer_ids]
        batch_size = len(input_datas)

        # Form batch data
        input_samples = []
        input_pitch = []
        input_singer_idx = torch.LongTensor(target_singer_idxs).cuda()
        for input_data in input_datas:
            input_samples.append(input_data['samples'])
            input_pitch.append(input_data['pitch'])
        input_samples = torch.FloatTensor(input_samples).cuda()
        input_pitch = torch.FloatTensor(input_pitch).cuda()

        # Setup models
        inference_pitchnet = PitchNet(singer_num=len(self.singer_ids)).cuda()
        inference_pitchnet.load_state_dict(self.pitchnet.module.state_dict())  # Get rid of DataParallel by cloning
        decoder = copy.deepcopy(self.pitchnet.module.decoder)  # Clone a decoder for generator
        generator = WaveNetGenerator(decoder, batch_size=batch_size)

        # Start generate
        inference_pitchnet.eval()
        with torch.no_grad():
            model_output = inference_pitchnet(input_samples, input_pitch, None, input_singer_idx, mode='enc')
            decoder_c = model_output['decoder_c']
            audio_datas = generator.generate(decoder_c).cpu().numpy()

        # Convert back to audio value
        y_list = [mu_law_decode(audio_data) for audio_data in audio_datas]

        return y_list, SAMPLE_RATE

    def two_phase_convert(self, src_file, singer_id, train_data_dir, pitch_shift=None):
        """
        Perform two-phase singing voice conversion given source audio file, target singer id and training data.
            - src_file: Path to the source audio file
            - singer_id: Singer id (name) of the target singer
            - train_data_dir: Original training data directory
            - pitch_shift: Whether or not to do pitch shifting
        """
        print('Two-phase conversion started.')

        tmp_singer_id = 'NEWS'
        tmp_model_dir = Path('./tmp/model/two-phase/')
        tmp_data_dir = Path('./tmp/data/')
        tmp_input_path = tmp_data_dir / tmp_singer_id / 'input.h5'

        # Preprocess audio file and save as .h5
        processed_data = preprocess_audio_file(src_file)
        if pitch_shift is not None:
            pitch_shift_semi = cal_shift_pitch_semi(processed_data['pitch'], self.pitch_ranges[singer_id], print_info=True) / 3  # Reduce pitch shift
            processed_data = preprocess_audio_file(src_file, pitch_shift_semi=pitch_shift_semi)

        tmp_input_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(tmp_input_path, 'w') as f:
            f.attrs['singer_id'] = tmp_singer_id
            f.create_dataset('samples', data=processed_data['samples'], dtype='int16')
            f.create_dataset('pitch', data=processed_data['pitch'], dtype='float32')
            f.create_dataset('pitch_mask', data=processed_data['pitch_mask'], dtype='float32')
        print('Processed input file saved to {}'.format(tmp_input_path))

        # Setup intermidiate model
        tp_singer_ids = self.singer_ids + [tmp_singer_id]
        tp_singer_mapping = SingerMapping(singer_ids=tp_singer_ids)
        tp_pitchnet = nn.DataParallel(PitchNet(singer_num=len(tp_singer_ids))).cuda()

        manual_keys = [
            'module.singer_embedding.weight',
            'module.singer_classifier.fc.weight',
            'module.singer_classifier.fc.bias'
        ]
        filtered_dict = copy.deepcopy(self.loaded_checkpoint['pitchnet_state_dict'])
        for manual_key in manual_keys:
            del filtered_dict[manual_key]
        tp_pitchnet.load_state_dict(filtered_dict, strict=False)

        with torch.no_grad():
            for manual_key in manual_keys:
                rgetattr(tp_pitchnet, manual_key)[:len(self.singer_ids)].copy_(
                    self.loaded_checkpoint['pitchnet_state_dict'][manual_key]
                )
        print('Two-phase model created from {}'.format(self.model_path))

        # ---- First phase: Training ----
        print('---- First phase: Training ----')
        # Set training params
        batch_size = 4
        total_steps = 302000
        lr = 1e-3

        # Loop units
        print_every = 1000

        # Initial states
        current_step = self.loaded_checkpoint['current_step'] + 1
        losses = []

        # Setup optimizers and schedulers
        ae_optimizer = optim.Adam(
            list(tp_pitchnet.module.singer_embedding.parameters()) +
            list(tp_pitchnet.module.encoder.parameters()) +
            list(tp_pitchnet.module.decoder.parameters()),
            lr=lr * (0.98 ** 300))
        singer_optimizer = optim.Adam(
            tp_pitchnet.module.singer_classifier.parameters(),
            lr=lr)
        pitch_optimizer = optim.Adam(
            tp_pitchnet.module.pitch_regressor.parameters(),
            lr=lr)
        losses = self.loaded_checkpoint['losses']

        # Read datasets
        print('Creating datasets...')
        origin_dataset = TrainingDataset(train_data_dir, self.singer_mapping)
        news_dataset = TrainingDataset(tmp_data_dir, tp_singer_mapping)
        news_dataset.singer_ids = [tmp_singer_id]  # Force dataset to only retrieve the new singer

        # Setup dataloader and initial variables
        origin_train_loader = iter(DataLoader(
            origin_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        ))
        news_train_loader = iter(DataLoader(
            news_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        ))

        start_time = time.time()
        loss_s, loss_p, loss_adv, loss_rec, loss_total = torch.zeros(5)

        # Start training
        print('Training...')
        while current_step <= total_steps:
            news_batch_data = next(news_train_loader)
            batch_data = next(origin_train_loader)
            for type_data_idx, type_data in enumerate(batch_data):
                # Replace the last entry of the batch data with news data
                type_data[3] = news_batch_data[type_data_idx][0]

            # Non-backtranslation training
            tp_pitchnet.train()

            # Parse data
            input_samples = batch_data[0].cuda()
            input_pitch = batch_data[1].cuda()
            input_pitch_mask = batch_data[2].cuda()
            input_singer_idx = batch_data[3].cuda()

            # Optimize adversarial networks
            ae_optimizer.zero_grad()
            singer_optimizer.zero_grad()
            pitch_optimizer.zero_grad()
            model_output = tp_pitchnet(input_samples, input_pitch, input_pitch_mask, input_singer_idx, mode='adv')
            loss_adv = model_output['loss_adv'].mean()

            loss_adv.backward()
            singer_optimizer.step()
            pitch_optimizer.step()

            # Optimize whole network
            ae_optimizer.zero_grad()
            singer_optimizer.zero_grad()
            pitch_optimizer.zero_grad()
            model_output = tp_pitchnet(input_samples, input_pitch, input_pitch_mask, input_singer_idx, mode='full')
            loss_s = model_output['loss_s'].mean()
            loss_p = model_output['loss_p'].mean()
            loss_adv = model_output['loss_adv'].mean()
            loss_rec = model_output['loss_rec'].mean()
            loss_total = model_output['loss_total'].mean()

            loss_total.backward()
            ae_optimizer.step()

            # Save losses
            losses.append({
                'current_step': current_step,
                'loss_s': loss_s.item(),
                'loss_p': loss_p.item(),
                'loss_adv': loss_adv.item(),
                'loss_rec': loss_rec.item(),
                'loss_total': loss_total.item(),
            })

            # Show training message
            if current_step % print_every == 0:
                print(
                    (
                        '| Step [{:6d}/{:6d}] '
                        'loss_s {:.4f} loss_p {:.4f} loss_adv {:.4f} loss_rec {:.4f} loss_total {:.4f} '
                        'Time {:.1f}'
                    ).format(
                        current_step,
                        total_steps,
                        losses[-1]['loss_s'],
                        losses[-1]['loss_p'],
                        losses[-1]['loss_adv'],
                        losses[-1]['loss_rec'],
                        losses[-1]['loss_total'],
                        (time.time()-start_time) / 60,
                    )
                )
            current_step += 1

        # Save checkpoint
        save_dict = {
            'pitchnet_state_dict': tp_pitchnet.state_dict(),
            'ae_optimizer_state_dict': ae_optimizer.state_dict(),
            'singer_optimizer_state_dict': singer_optimizer.state_dict(),
            'pitch_optimizer_state_dict': pitch_optimizer.state_dict(),
            'current_step': current_step-1,
            'losses': losses,
            'singer_ids': tp_singer_ids,
            'pitch_ranges': self.pitch_ranges,
        }
        tmp_model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = tmp_model_dir / 's_{}.tar'.format(current_step-1)
        torch.save(save_dict, checkpoint_path)

        print('Training done in {:.1f} minutes.'.format((time.time()-start_time) / 60))

        # ---- Second phase: Inference ----
        print('---- Second phase: Inference ----')
        # Setup input tensors
        singer_idx = tp_singer_mapping.singer_id_to_idx(singer_id)
        input_samples = torch.FloatTensor([processed_data['samples']]).cuda()
        input_pitch = torch.FloatTensor([processed_data['pitch']]).cuda()
        input_singer_idx = torch.LongTensor([singer_idx]).cuda()
        inference_batch_size = input_samples.shape[0]

        # Setup models
        inference_pitchnet = PitchNet(singer_num=len(tp_singer_ids)).cuda()
        inference_pitchnet.load_state_dict(tp_pitchnet.module.state_dict())  # Get rid of DataParallel by cloning
        decoder = copy.deepcopy(tp_pitchnet.module.decoder)  # Clone a decoder for generator
        generator = WaveNetGenerator(decoder, batch_size=inference_batch_size)

        # Start generate
        inference_pitchnet.eval()
        with torch.no_grad():
            model_output = inference_pitchnet(input_samples, input_pitch, None, input_singer_idx, mode='enc')
            decoder_c = model_output['decoder_c']
            audio_data = generator.generate(decoder_c).reshape(-1).cpu().numpy()

        # Convert back to audio value
        y = mu_law_decode(audio_data)

        return y, SAMPLE_RATE

    def print_checkpoint_info(self):
        """Print out saved information in the loaded checkpoint."""
        # Print singer_ids
        print('singer_ids:')
        print(self.singer_ids)

        # Print pitch_ranges
        print('pitch_ranges:')
        for singer_id, pitch_range in self.pitch_ranges.items():
            print(singer_id)
            for stat_type, value in pitch_range.items():
                print('{}: {:.4f}'.format(stat_type, value), end=' ')
            print('')
