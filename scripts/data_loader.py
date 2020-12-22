from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import audio.audio_utils as audio
import audio.hparams as hparams
import random
import os
import librosa
     
class DataGenerator(Dataset):

    def __init__(self, pretrain_path, train_path, noise_path, sampling_rate, split):

        self.files = hparams.get_all_files(pretrain_path, train_path, split) 
        self.random_files = hparams.get_noise_list(noise_path)
        self.sampling_rate = sampling_rate
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        while(1):
            index = random.randint(0, len(self.files) - 1)
            fname = self.files[index]

            mel, stft, y = self.process_audio(fname)

            if mel is None or stft is None or y is None:
                continue

            inp_mel = torch.FloatTensor(np.array(mel)).unsqueeze(1)
            inp_stft = torch.FloatTensor(np.array(stft))
            gt_stft = torch.FloatTensor(np.array(y))

            return inp_mel, inp_stft, gt_stft 


    def process_audio(self, file):

        # Load the gt wav file
        try:
            gt_wav = audio.load_wav(file, self.sampling_rate)                   # m
        except:
            return None, None, None

        # Get the random file from VGGSound to mix with the ground truth file
        random_file = random.choice(self.random_files)

        # Load the random wav file
        try:
            random_wav = audio.load_wav(random_file, self.sampling_rate)        # n
        except:
            return None, None, None

        # Mix the noisy wav file with the clean GT file
        try:
            idx = random.randint(0, len(random_wav) - len(gt_wav) - 1)
            random_wav = random_wav[idx:idx + len(gt_wav)]
            snrs = [0, 5, 10]
            target_snr = random.choice(snrs)
            noisy_wav = self.add_noise(gt_wav, random_wav, target_snr)
        except:
            return None, None, None

        # Extract the corresponding audio segments of 1 second
        start_idx, gt_seg_wav, noisy_seg_wav = self.crop_audio_window(gt_wav, noisy_wav, random_wav)
        
        if start_idx is None or gt_seg_wav is None or noisy_seg_wav is None:
            return None, None, None


        # -----------------------------------STFTs--------------------------------------------- #
        # Get the STFT, normalize and concatenate the mag and phase of GT and noisy wavs
        gt_spec = self.get_spec(gt_seg_wav)                                     # Tx514

        noisy_spec = self.get_spec(noisy_seg_wav)                               # Tx514 
        # ------------------------------------------------------------------------------------- #


        # -----------------------------------Melspecs------------------------------------------ #                          
        noisy_mels = self.get_segmented_mels(start_idx, noisy_wav)              # Tx80x16
        if noisy_mels is None:
            return None, None, None
        # ------------------------------------------------------------------------------------- #
        
        # Input to the lipsync student model: Noisy melspectrogram
        inp_mel = np.array(noisy_mels)                                          # Tx80x16

        # Input to the denoising model: Noisy linear spectrogram
        inp_stft = np.array(noisy_spec)                                         # Tx514

        # GT to the denoising model: Clean linear spectrogram
        gt_stft = np.array(gt_spec)                                             # Tx514

        
        return inp_mel, inp_stft, gt_stft


    def crop_audio_window(self, gt_wav, noisy_wav, random_wav):

        if gt_wav.shape[0] - hparams.hparams.wav_step_size <= 1280: 
            return None, None, None

        # Get 1 second random segment from the wav
        start_idx = np.random.randint(low=1280, high=gt_wav.shape[0] - hparams.hparams.wav_step_size)
        end_idx = start_idx + hparams.hparams.wav_step_size
        gt_seg_wav = gt_wav[start_idx : end_idx]
        
        if len(gt_seg_wav) != hparams.hparams.wav_step_size: 
            return None, None, None

        noisy_seg_wav = noisy_wav[start_idx : end_idx]
        if len(noisy_seg_wav) != hparams.hparams.wav_step_size: 
            return None, None, None

        # Data augmentation
        aug_steps = np.random.randint(low=0, high=3200)
        aug_start_idx = np.random.randint(low=0, high=hparams.hparams.wav_step_size - aug_steps)
        aug_end_idx = aug_start_idx+aug_steps

        aug_types = ['zero_speech', 'reduce_speech', 'increase_noise']
        aug = random.choice(aug_types)

        if aug == 'zero_speech':    
            noisy_seg_wav[aug_start_idx:aug_end_idx] = 0.0
            
        elif aug == 'reduce_speech':
            noisy_seg_wav[aug_start_idx:aug_end_idx] = 0.1*gt_seg_wav[aug_start_idx:aug_end_idx]

        elif aug == 'increase_noise':
            random_seg_wav = random_wav[start_idx : end_idx]
            noisy_seg_wav[aug_start_idx:aug_end_idx] = gt_seg_wav[aug_start_idx:aug_end_idx] + (2*random_seg_wav[aug_start_idx:aug_end_idx])

        return start_idx, gt_seg_wav, noisy_seg_wav


    def crop_mels(self, start_idx, noisy_wav):
        
        end_idx = start_idx + 3200

        # Get the segmented wav (0.2 second)
        noisy_seg_wav = noisy_wav[start_idx : end_idx]
        if len(noisy_seg_wav) != 3200: 
            return None
        
        # Compute the melspectrogram using librosa
        spec = audio.melspectrogram(noisy_seg_wav, hparams.hparams).T              # 16x80
        spec = spec[:-1] 

        return spec


    def get_segmented_mels(self, start_idx, noisy_wav):

        mels = []
        if start_idx - 1280 < 0: 
            return None

        # Get the overlapping continuous segments of noisy mels
        for i in range(start_idx, start_idx + hparams.hparams.wav_step_size, 640): 
            m = self.crop_mels(i - 1280, noisy_wav)                             # Hard-coded to get 0.2sec segments (5 frames)
            if m is None or m.shape[0] != hparams.hparams.mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)                                                 # Tx80x16

        return mels


    def get_spec(self, wav):

        # Compute STFT using librosa
        stft = librosa.stft(y=wav, n_fft=hparams.hparams.n_fft_den, \
               hop_length=hparams.hparams.hop_size_den, win_length=hparams.hparams.win_size_den).T
        stft = stft[:-1]                                                        # Tx257

        # Decompose into magnitude and phase representations
        mag = np.abs(stft)
        mag = audio.db_from_amp(mag)
        phase = audio.angle(stft)

        # Normalize the magnitude and phase representations
        norm_mag = audio.normalize_mag(mag)
        norm_phase = audio.normalize_phase(phase)
            
        # Concatenate the magnitude and phase representations
        spec = np.concatenate((norm_mag, norm_phase), axis=1)               # Tx514
        
        return spec

    def add_noise(self, gt_wav, random_wav, desired_snr):

        samples = len(gt_wav)

        signal_power = np.sum(np.square(np.abs(gt_wav)))/samples
        noise_power = np.sum(np.square(np.abs(random_wav)))/samples

        k = (signal_power/(noise_power+1e-8)) * (10**(-desired_snr/10))

        scaled_random_wav = np.sqrt(k)*random_wav

        noisy_wav = gt_wav + scaled_random_wav

        return noisy_wav


def load_data(pretrain_path, train_path, noise_path, num_workers, batch_size=4, split='train', sampling_rate=16000, shuffle=False):
    
    dataset = DataGenerator(pretrain_path, train_path, noise_path, sampling_rate, split)

    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return data_loader