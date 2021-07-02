import time
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


from .net import SingerClassifier
from .util.singer_mapping import SingerMapping
from .util.audio_process import preprocess_audio_file
from .dataset import TrainingDataset


def _parallel_preprocess_work(audio_path):
    processed_data = preprocess_audio_file(audio_path)
    return processed_data['mfcc']


class SingerPredictor:
    def __init__(self, model_path=None, train_data_dir=None, model_dir=None):
        # Load checkpoint and singer mapping
        self.model_path = model_path
        self.train_data_dir = train_data_dir
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.loaded_checkpoint = None
        if self.model_path is not None:
            self.loaded_checkpoint = torch.load(self.model_path)
            self.singer_mapping = SingerMapping(singer_ids=self.loaded_checkpoint['singer_ids'])
        elif self.train_data_dir is not None and self.model_dir is not None:
            self.singer_mapping = SingerMapping(data_dir=self.train_data_dir)
        else:
            raise ValueError('Either model_path or (train_data_dir, model_dir) should be provided to the convertor.')
        self.singer_ids = self.singer_mapping.get_singer_ids()
        self.singer_idxs = self.singer_mapping.get_singer_idxs()

        # Initialize model
        self.singer_classifier = nn.DataParallel(SingerClassifier(singer_num=len(self.singer_ids))).cuda()

        # Load model if checkpoint is given
        if self.loaded_checkpoint is not None:
            self.singer_classifier.load_state_dict(self.loaded_checkpoint['singer_classifier_state_dict'])
            print('Model read from {}.'.format(self.model_path))

        print('Predictor initialized.')

    def fit(self, **training_args):
        """
        training_args:
            - batch_size
            - total_steps
            - lr
            - lr_step_size
            - save_model_every
        """
        # Set training params
        self.batch_size = training_args['batch_size']
        self.total_steps = training_args['total_steps']
        self.lr = training_args['lr']
        self.lr_step_size = training_args['lr_step_size']

        # Loop units
        self.print_every = 100
        self.save_model_every = training_args['save_model_every']

        # Initial states
        self.current_step = 1
        self.losses = []

        # Setup optimizers and schedulers
        self.optimizer = optim.Adam(self.singer_classifier.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size, gamma=0.95)

        # Parse checkpoint
        if self.loaded_checkpoint is not None:
            self.optimizer.load_state_dict(self.loaded_checkpoint['optimizer_state_dict'])
            self.current_step = self.loaded_checkpoint['current_step']+1
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.lr_step_size,
                gamma=0.95,
                last_epoch=self.loaded_checkpoint['current_step']-1
            )
            self.losses = self.loaded_checkpoint['losses']

        # Read datasets
        print('Creating datasets...')
        self.training_dataset = TrainingDataset(self.train_data_dir, self.singer_mapping)

        # Setup dataloader and initial variables
        self.train_loader = DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        start_time = time.time()

        # Start training
        print('Training...')
        while self.current_step <= self.total_steps:
            for batch_idx, batch_data in enumerate(self.train_loader):
                self.singer_classifier.train()

                # Parse data
                input_mfcc = batch_data[0].cuda()
                target_singer_idx = batch_data[1].cuda()

                # Optimize whole network
                self.optimizer.zero_grad()
                model_output = self.singer_classifier(input_mfcc)
                loss_s = F.cross_entropy(model_output, target_singer_idx)

                loss_s.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Save losses
                self.losses.append({
                    'current_step': self.current_step,
                    'loss_s': loss_s.item(),
                })

                # Save checkpoint
                if self.current_step % self.save_model_every == 0:
                    save_dict = {
                        'singer_classifier_state_dict': self.singer_classifier.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'current_step': self.current_step,
                        'losses': self.losses,
                        'singer_ids': self.singer_ids,
                    }
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = self.model_dir / 's_{}.tar'.format(self.current_step)
                    torch.save(save_dict, checkpoint_path)

                # Show training message
                if self.current_step % self.print_every == 0:
                    print('| Step [{:6d}/{:6d}] loss_s {:.4f} Time {:.1f}'.format(
                        self.current_step,
                        self.total_steps,
                        self.losses[-1]['loss_s'],
                        (time.time()-start_time) / 60,
                    )
                    )

                self.current_step += 1
                if self.current_step > self.total_steps:
                    break

        print('Training done in {:.1f} minutes.'.format((time.time()-start_time) / 60))

    def predict(self, audio_paths):
        """
        Perform singer classification given source audio files.
        Return a list of predicted singer_id
            - audio_paths: Paths to the source audio files
        """
        print('Preprocessing audio files...')
        with mp.get_context("spawn").Pool(mp.cpu_count()) as pool:
            # Parallel work for each audio file
            input_mfcc = list(tqdm(pool.imap(_parallel_preprocess_work, audio_paths), total=len(audio_paths)))
        input_mfcc = torch.FloatTensor(input_mfcc).cuda()

        print('Forwarding model...')
        self.singer_classifier.eval()
        with torch.no_grad():
            model_output = self.singer_classifier(input_mfcc)
            _, predicted_indices = torch.max(model_output, dim=1)
            predicted_ids = [self.singer_mapping.singer_idx_to_id(predicted_idx.item())
                             for predicted_idx in predicted_indices]

        return predicted_ids
