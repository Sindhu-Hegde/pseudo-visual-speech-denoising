# Import all the necessary libraries
import argparse
import os, sys, cv2, subprocess
from os.path import dirname, join, basename, isfile
import numpy as np
import random
from tqdm import tqdm

from models import *
from hparams import hparams, get_all_files, get_noise_list
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils

# Initialize the global variables
global_step = 0
global_epoch = 0

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
if use_cuda:
	cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")

recon_loss = nn.L1Loss()

# Dataloader class
class Dataset(object):
	def __init__(self, split):
		self.all_videos = get_all_files(args.data_root_lrs3_pretrain, args.data_root_lrs3_train, split) 
		self.all_noises = get_noise_list(args.noise_data_root)
		self.face = cv2.imread('checkpoints/taylor.jpg')
		self.face = cv2.resize(self.face, (hparams.img_size, hparams.img_size))

		wrong_window = self.prepare_window([self.face for _ in range(hparams.syncnet_T)])
		window = self.prepare_window([self.face for _ in range(hparams.syncnet_T)])
		window[:, :, window.shape[2]//2:] = 0
		self.face = np.concatenate([window, wrong_window], axis=0)

	def get_segmented_mels(self, wav):
		clean_mels = []
		noisy_mels = []
		assert hparams.syncnet_T == 5
		
		# Get the random start index
		start_idx = random.randint(0, len(wav) - hparams.syncnet_wav_step_size - 1) ## Hard-coded for 25fps, 16000 sample_rate
		end_idx = start_idx + hparams.syncnet_wav_step_size

		# Segment the clean wav based on start and end index 
		seg_wav = wav[start_idx : end_idx]

		# Check if the segmented wav corresponds to wav step size (here: 3200)
		if len(seg_wav) != hparams.syncnet_wav_step_size: 
			return None, [None, None]

		# Get the melspectrogram of the segmented clean wav
		spec = audio.melspectrogram(seg_wav).T                      # (T+1)x80
		spec = spec[:-1]                                            # Drop last time-step, Tx80

		# Choose the random wav from VGG sound data
		noisy_wav = audio.load_wav(random.choice(self.all_noises), sr=hparams.sample_rate)

		# Mix the random wav with the clean wav
		try:
			noisy_wav = wav + random.uniform(0.3, 1) * noisy_wav[:len(wav)]
		except: 
			return None, [None, None]

		# Get the 5 overlapping continuous segments of clean and noisy mels
		for i in range(start_idx, start_idx + 3200, 640):

			# Start index (2 steps behind)
			s = i - 1280
			if s < 0: 
				return None, [None, None]

			# End index
			e = s + hparams.syncnet_wav_step_size

			# Get the corresponding clean and noisy wav segments
			clean_seg_wav = wav[s:e]
			noisy_seg_wav = noisy_wav[s:e]

			# Check for the wav step size (here: 3200)
			if len(noisy_seg_wav) != hparams.syncnet_wav_step_size or len(clean_seg_wav) != hparams.syncnet_wav_step_size: 
				return None, [None, None]

			# Compute the melspectrogram for the clean wav segment
			clean_m = audio.melspectrogram(clean_seg_wav).T              
			clean_m = clean_m[:-1]                                      # Tx80

			# Compute the melspectrogram for the noisy wav segment
			noisy_m = audio.melspectrogram(noisy_seg_wav).T             
			noisy_m = noisy_m[:-1]                                      # Tx80

			# Check for the melspec dimensions
			if clean_m is None or clean_m.shape[0] != hparams.syncnet_mel_step_size:
				return None, [None, None]
			if noisy_m is None or noisy_m.shape[0] != hparams.syncnet_mel_step_size:
				return None, [None, None]

			clean_mels.append(clean_m.T)
			noisy_mels.append(noisy_m.T)

		# Convert to array
		clean_mels = np.asarray(clean_mels)                             # 5x80xT
		noisy_mels = np.asarray(noisy_mels)                             # 5x80xT     

		indiv_mels = [clean_mels, noisy_mels]

		return spec, indiv_mels

	def prepare_window(self, window):

		# 3 x T x H x W
		x = np.asarray(window) / 255.
		x = np.transpose(x, (3, 0, 1, 2))

		return x

	def __len__(self):
		return len(self.all_videos)

	def __getitem__(self, idx):
		while 1:
			idx = random.randint(0, len(self.all_videos) - 1)
			vidname = self.all_videos[idx]

			try:
				wavpath = os.path.join(vidname, "audio.wav")
				wav = audio.load_wav(wavpath, sr=hparams.sample_rate)
			except Exception as e:
				continue

			mel, indiv_mels = self.get_segmented_mels(wav.copy())
			clean_indiv_mels = indiv_mels[0]
			noisy_indiv_mels = indiv_mels[1]

			if clean_indiv_mels is None or noisy_indiv_mels is None: 
				continue

			mel = torch.FloatTensor(mel.T).unsqueeze(0)								# 1x80xT
			clean_indiv_mels = torch.FloatTensor(clean_indiv_mels).unsqueeze(1) 	# 5x1x80xT    
			noisy_indiv_mels = torch.FloatTensor(noisy_indiv_mels).unsqueeze(1) 	# 5x1x80xT    
			y = torch.FloatTensor(self.face)       									# 6x5x96x96                                 

			return mel, clean_indiv_mels, noisy_indiv_mels, y

def save_sample_images(g, gt, global_step, checkpoint_dir):
	g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
	gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

	folder = join(checkpoint_dir, "samples_step{:05d}".format(global_step))
	if not os.path.exists(folder): os.mkdir(folder)
	collage = np.concatenate((g, gt), axis=-2)
	for batch_idx, c in enumerate(collage):
		for t in range(len(c)):
			cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

	print("Saved samples:", folder)



def train(device, model, train_data_loader, test_data_loader, optimizer,
		  checkpoint_dir, checkpoint_interval, nepochs):
	
	global wav2lip_teacher
	global global_step, global_epoch
	
	resumed_step = global_step
	
	while global_epoch < nepochs:
	
		running_l1_loss = 0.0
		prog_bar = tqdm(enumerate(train_data_loader))		
	
		for step, (mel, clean_indiv_mels, noisy_indiv_mels, face) in prog_bar:
			
			model.train()
			optimizer.zero_grad()

			# Transform data to CUDA device
			mel = mel.to(device)										# Bx1x80x16
			clean_indiv_mels = clean_indiv_mels.to(device)				# Bx5x1x80x16
			noisy_indiv_mels = noisy_indiv_mels.to(device)				# Bx5x1x80x16
			face = face.to(device)										# Bx6x5x96x96

			# Get the GT lip-shape using the teacher wav2lip model
			with torch.no_grad():
				gt = wav2lip_teacher(clean_indiv_mels, face)			# Bx3x5x96x96

			# Generate the lips using the student lisync model (on noisy data)
			generated_face = model(noisy_indiv_mels)					# Bx3x5x48x96
			
			# Get the L1 reconstruction loss
			loss = recon_loss(generated_face, gt[:, :, :, gt.size(3)//2:])
			running_l1_loss += loss.item()

			# Backpropagate
			loss.backward()
			optimizer.step()

			# Logs
			global_step += 1
			cur_session_steps = global_step - resumed_step
			
			# Save the model
			if global_step == 1 or global_step % checkpoint_interval == 0:
				save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)

			# Validation loop
			if global_step == 1 or global_step % hparams.eval_interval == 0:
				with torch.no_grad():
					validate(test_data_loader, global_step, device, model, checkpoint_dir)
					
			# Display the training progress
			prog_bar.set_description('L1: {}'.format(running_l1_loss / (step + 1)))

		global_epoch += 1
		

def validate(test_data_loader, global_step, device, model, checkpoint_dir):
	
	print('Evaluating for {} steps'.format(len(test_data_loader)))
	
	recon_losses = []
	
	for step, (mel, clean_indiv_mels, noisy_indiv_mels, face) in enumerate((test_data_loader)):

		model.eval()

		# Transform data to CUDA device
		mel = mel.to(device)
		clean_indiv_mels = clean_indiv_mels.to(device)
		noisy_indiv_mels = noisy_indiv_mels.to(device)
		face = face.to(device)

		with torch.no_grad():
			gt = wav2lip_teacher(clean_indiv_mels, face)

		generated_face = model(noisy_indiv_mels)
		
		l1loss = recon_loss(generated_face, gt[:, :, :, gt.size(3)//2:])
		recon_losses.append(l1loss.item())

	# Compute the average of the validation loss
	averaged_recon_loss = sum(recon_losses) / len(recon_losses)
	print('L1: {}'.format(averaged_recon_loss))

	# Save the ground truth and the generated samples
	gt_lh = gt[:, :, :, gt.size(3)//2:, :]
	save_sample_images(generated_face, gt_lh, global_step, checkpoint_dir)

	return


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

	checkpoint_path = join(checkpoint_dir, "checkpoint_step{:05d}.pth".format(global_step))

	torch.save({
		"state_dict": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"global_step": step,
		"global_epoch": epoch,
	}, checkpoint_path)

	print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
	
	if use_cuda:
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

	return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):

	global global_step
	global global_epoch

	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}

	for k, v in s.items():
		if torch.cuda.device_count() > 1:
			if not k.startswith('module.'):
				new_s['module.'+k] = v
			else:
				new_s[k] = v
		else:
			new_s[k.replace('module.', '')] = v

	model.load_state_dict(new_s)

	if not reset_optimizer:
		optimizer_state = checkpoint["optimizer"]
		if optimizer_state is not None:
			print("Load optimizer state from {}".format(path))
			optimizer.load_state_dict(checkpoint["optimizer"])
	
	if overwrite_global_states:
		global_step = checkpoint["global_step"]
		global_epoch = checkpoint["global_epoch"]

	return model

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Code to train the lipsync student model')

	parser.add_argument("--data_root_lrs3_pretrain", help="Root folder of the preprocessed LRS3 pre-train dataset", required=True, type=str)
	parser.add_argument("--data_root_lrs3_train", help="Root folder of the preprocessed LRS3 train dataset", required=True, type=str)
	parser.add_argument("--noise_data_root", help="Root folder of the VGGSound dataset", required=True, type=str)

	parser.add_argument('--wav2lip_checkpoint_path', help='Load the pre-trained Wav2Lip model', required=True, type=str)

	parser.add_argument('--checkpoint_dir', help='Save checkpoints of the trained lipsync student model to this directory', required=True, type=str)
	parser.add_argument('--checkpoint_path', help='Resume the lipsync student model from this checkpoint', default=None, type=str)

	args = parser.parse_args()

	# Dataset and Dataloader setup
	train_dataset = Dataset('train')
	test_dataset = Dataset('val')

	train_data_loader = data_utils.DataLoader(
		train_dataset, batch_size=hparams.batch_size, shuffle=True,
		num_workers=hparams.num_workers)
	print("Total train batch: ", len(train_data_loader))

	test_data_loader = data_utils.DataLoader(
		test_dataset, batch_size=hparams.batch_size,
		num_workers=hparams.num_workers)

	# Teacher wav2lip model
	wav2lip_teacher = Wav2Lip_Teacher()
	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs for Wav2Lip teacher model!")
		wav2lip_teacher = nn.DataParallel(wav2lip_teacher)
	else:
		print("Using single GPU for Wav2Lip teacher model!")
	wav2lip_teacher.to(device)

	for p in wav2lip_teacher.parameters():
		p.requires_grad = False


	# Student lipsync model
	model = Lipsync_Student()
	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs for model!")
		model = nn.DataParallel(model)
	else:
		print("Using single GPU for model!")
	model.to(device)

	print('Total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

	optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
						   lr=hparams.initial_learning_rate)

	# Resume the student lipsync model for training if the path is provided
	if args.checkpoint_path is not None:
		load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

	# Load the teacher wav2lip model 
	load_checkpoint(args.wav2lip_checkpoint_path, wav2lip_teacher, None, reset_optimizer=True, overwrite_global_states=False)

	# Train!
	train(device, model, train_data_loader, test_data_loader, optimizer,
			  checkpoint_dir=args.checkpoint_dir,
			  checkpoint_interval=hparams.checkpoint_interval,
			  nepochs=hparams.nepochs)

	print("Finished")
	sys.exit(0)
