# Unsupervised WaveNet-based Singing Voice Conversion Using Pitch Augmentation and Two-phase Approach
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)  
This repository implements the singing voice conversion method described in [Pitchnet: Unsupervised Singing Voice Conversion with Pitch Adversarial Network](https://arxiv.org/abs/1912.01852) along with multiple improvements regarding its conversion quality using PyTorch. Detailed surveys and experiments have been published as a master thesis, you can view it [here](TODO).

You can find demo audio files and comaprisons to the original PitchNet on our [demo website](https://songronglee.github.io/singing-voice-conversion/).


## Table of contents
- [Dataset](https://github.com/SongRongLee/singing-voice-conversion#dataset)
- [Environment setup](https://github.com/SongRongLee/singing-voice-conversion#environment-setup)
- [Scripts usage](https://github.com/SongRongLee/singing-voice-conversion#scripts-usage)
  - [Data augmentation](https://github.com/SongRongLee/singing-voice-conversion#data-augmentation)
  - [Data preprocessing](https://github.com/SongRongLee/singing-voice-conversion#data-preprocessing)
  - [Start training](https://github.com/SongRongLee/singing-voice-conversion#start-training)
  - [Converting an audio file](https://github.com/SongRongLee/singing-voice-conversion#converting-an-audio-file)
  - [Ploting results](https://github.com/SongRongLee/singing-voice-conversion#ploting-results)
  - [Network summary & testing](https://github.com/SongRongLee/singing-voice-conversion#network-summary--testing)
  - [Evaluation](https://github.com/SongRongLee/singing-voice-conversion#evaluation)
- [License](https://github.com/SongRongLee/singing-voice-conversion#license)
- [Citation](https://github.com/SongRongLee/singing-voice-conversion#citation)


## Dataset
We use [NUS-48E](https://smcnus.comp.nus.edu.sg/nus-48e-sung-and-spoken-lyrics-corpus/) dataset throughout the whole project. You can download it and perform data preprocessing and augmentation below.


## Environment setup
Create a conda environment using `environment.yml`:  
`conda env create -f environment.yml`


## Scripts usage
**Notice: Make sure you are under the project root when executing these scripts!** 

### Data augmentation
This script will read through the given `$raw_dir` and generate folders with the same structure to `$output_dir`, containing augmented audio files next to the original ones.  

`python data_augmentation.py $raw_dir $output_dir --aug-type $aug_type`  
- `raw_dir`: Path to the raw data directory with the following structure:
```
-> $raw_dir/
├── ADIZ
│   ├── 01.wav
│   ├── 09.wav
│   ├── 13.wav
│   └── 18.wav
├── JLEE
│   ├── 05.wav
│   ├── 08.wav
│   ├── 11.wav
│   └── 15.wav
...
```  
- `output_dir`: Path to the directory to save the augmented and original files. The resulting structure will look like this:
```
-> $output_dir/
├── ADIZ
│   ├── 01_original.wav
│   ├── 01_aug_back.wav
│   ├── 01_aug_phase.wav
│   ├── 01_aug_back_phase.wav
│   ├── 09_original.wav
│   ├── 09_aug_back.wav
│   ├── 09_aug_phase.wav
│   ├── 09_aug_back_phase.wav
...
...
```  
- `aug_type`: Type of augmentation

### Data preprocessing
This script will read through the given `$raw_dir` and generate folders with the same structure to `$output_dir`, with each audio file processed as a `*.h5` data file ready to be read by dataset classes.  

`python data_preprocess.py $raw_dir $output_dir --model $model`  
- `raw_dir`: Path to the raw data directory
- `output_dir`: Path to the directory to save the processed files
- `model`: Target model type which we are doing data preprocessing for

### Start training
This script will train the model. If `--model-path` is given, the training will continue with that checkpoint. To see other training parameters, run the script with `-h`.  

`python train.py $train_data_dir $model_dir --model $model --model-path $model_path`  
- `train_data_dir`: Path to the proccesed data directory
- `model_dir`: Directory to save checkpoint models
- `model`: Target model type
- `model_path`: Path to pretrained model  

You can get our pretrained proposed model [here](https://drive.google.com/file/d/1f0x8M4QYoRsB5T8FeM0K1hXF3nTa7cRf/view?usp=sharing).

### Converting an audio file
This script will perform singing voice conversion on the given audio file. For two-phase conversion, the intermediate files will be saved to `.tmp/` directory.   
`python inference.py $src_file $target_dir $singer_id $model_path --pitch-shift $pitch_shift --two-phase --train-data-dir $train_data_dir`  
- `src_file`: Path to the source audio file
- `target_dir`: Path to save the converted audio file
- `singer_id`: Target singer ID (name)
- `model_path`: Model path
- `pitch_shift`: Factor of pitch shifting performed on conversion, or "auto" for automatic pitch range shifting
- `two_phase`: Whether or not to perform two-phase conversion
- `train_data_dir`: The original training data used for two-phase conversion

### Ploting results
#### Loss curves
This script will plot the training loss curves of a given checkpoint. The output image will be stored in `plotting-scripts/plotting-results/`.  
`python plotting-scripts/plot_loss.py $checkpoint_path --window-size $window_size --loss-types $loss_types`  
- `checkpoint_path`: Path to the target training checkpoint
- `window_size`: Window size for moving average
- `loss_types`: Target types of loss separated by spaces

#### Pitch curves
This script will plot the pitch extracted from the given audio file.  
`python plotting-scripts/plot_pitch.py $src_file`  
- `src_file`: Path to the source audio file

#### Duration histogram
This script will plot the audio duration histogram of the given dataset.  
`python plotting-scripts/plot_hist.py $raw_dir`  
- `raw_dir`: Path to the raw data directory

#### Pitch histogram
This script will plot the pitch histogram of the given dataset.  
`python plotting-scripts/plot_pitch_hist.py $raw_dir`  
- `raw_dir`: Path to the raw data directory

#### Audio Spectrogram
This script will plot the spectrogram of the given audio file.  
`python plotting-scripts/plot_spec.py $src_file`  
- `src_file`: Path to the source audio file

### Network summary & testing
This script will conduct simple unit tests and print out a model summary (if applicable). Run with `-h` option to see all available networks.  

`python test_network.py $target_net`

### Evaluation
#### Data selection
This script will select random N seconds segment for each raw audio file in the given data directory and output it as a mini dataset.  

`python evaluation/select_data.py $raw_dir $output_dir --seg-len $seg_len`
- `raw_dir`: Path to the raw data directory
- `output_dir`: Path to the directory to save the processed files
- `seg_len`: Length (seconds) for each segment

#### Evaluation script
This script will perform evaluation given evaluation data directory, output file directory, and the target model.

`python evaluation/evaluate.py $raw_dir $output_dir $model_path $sc_model_path $mapping --pitch-shift --two-phase --train-data-dir`
- `raw_dir`: Path to the evaluation data directory
- `output_dir`: Path to the directory to save converted audio files
- `model_path`: Path to the target model to evaluate
- `sc_model_path`: Path to the singer classifier model
- `mapping`: The mapping config of the conversion pairs
- `pitch_shift`: Whether or not to perform pitch shifting
- `two_phase`: Whether or not to perform two-phase conversion
- `train_data_dir`: The original training data used for two-phase conversion

You can get the singer classifier model we used in the evaluation [here](https://drive.google.com/file/d/1IoqfatYEL43pGFnAqSobl_XETQijFn3k/view?usp=sharing).


## License
[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc/4.0/)  

- This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).  
- We referenced [facebookresearch/music-translation](https://github.com/facebookresearch/music-translation), which has the same [license](https://github.com/facebookresearch/music-translation/blob/master/LICENSE), for WaveNet implementation and made modifications accordingly to fit our usages.  
- [pytorch-summary](https://github.com/sksq96/pytorch-summary) is used in this repo, which is licensed under a [MIT License](torchsummary/LICENSE)  


## Citation
TODO
