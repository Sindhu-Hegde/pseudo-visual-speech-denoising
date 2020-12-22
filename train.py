import numpy as np
import os
import argparse
from models import *
import audio.hparams as hparams 
from scripts.data_loader import *
from tqdm import tqdm
import librosa
import torch
import torch.optim as optim
import cv2
import subprocess

# Initialize the global variables
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")

# Function to generate the video from the audio and frames
def generate_video(frames, audio_file, output_file_name, fps=25):

	fname = 'output_lower.avi'
	video = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frames[0].shape[1], frames[0].shape[0]))
 
	for i in range(len(frames)):
		img = np.clip(np.round(frames[i]*255), 0, 255)
		video.write(np.uint8(img))
	
	video.release()

	no_sound_video = output_file_name + '_nosound.mp4'
	subprocess.call('ffmpeg -hide_banner -loglevel panic -i %s -c copy -an -strict -2 %s' % (fname, no_sound_video), shell=True)

	video_output = output_file_name + '.mp4'
	subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -i %s -strict -2 -q:v 1 %s' % 
					(audio_file, no_sound_video, video_output), shell=True)

	os.remove(fname)
	os.remove(no_sound_video)

# Function to reconstruct the wav from the magnitude and phase representations
def reconstruct_wav(stft):

	mag = stft[:257, :]
	phase = stft[257:, :]
	
	denorm_mag = audio.unnormalize_mag(mag)
	denorm_phase = audio.unnormalize_phase(phase)
	recon_mag = audio.amp_from_db(denorm_mag)
	complex_arr = audio.make_complex(recon_mag, denorm_phase)
	wav = librosa.istft(complex_arr, hop_length=hparams.hparams.hop_size_den, win_length=hparams.hparams.win_size_den)
	
	return wav
 
# Function to generate and save the sample audio/video files
def save_samples(gt_stft, inp_stft, output_stft, faces, epoch, checkpoint_dir):

	gt_stft = gt_stft.detach().cpu().numpy()
	inp_stft = inp_stft.detach().cpu().numpy()
	output_stft = output_stft.detach().cpu().numpy()
	faces = faces.permute(0,2,3,4,1)
	faces = faces.detach().cpu().numpy()

	folder = join(checkpoint_dir, "samples_step{:04d}".format(epoch))
	if not os.path.exists(folder): 
		os.mkdir(folder)

	for step in range((gt_stft.shape[0])): 

		# Save GT audio
		gt = gt_stft[step]
		gt_wav = reconstruct_wav(gt.T)		
		gt_aud_fname = os.path.join(folder, str(step)+'_gt.wav')
		librosa.output.write_wav(gt_aud_fname, gt_wav, 16000) 

		# Save input audio
		inp = inp_stft[step]
		inp_wav = reconstruct_wav(inp.T)		
		inp_aud_fname = os.path.join(folder, str(step)+'_inp.wav')
		librosa.output.write_wav(inp_aud_fname, inp_wav, 16000)            
		
		# Save generated audio
		generated = output_stft[step]
		generated_wav = reconstruct_wav(generated.T)
		generated_aud_fname = os.path.join(folder, str(step)+'_pred.wav')
		librosa.output.write_wav(generated_aud_fname, generated_wav, 16000)            

		# Save generated video
		generated_vid_fname = os.path.join(folder, str(step)+'_pred')
		generate_video(faces[step], generated_aud_fname, generated_vid_fname)     
	
	print("Saved samples:", folder)


def train(device, lipsync_student, model, train_loader, test_loader, optimizer, epoch_resume, total_epochs, checkpoint_dir, args):

	l1_loss = nn.L1Loss()
	
	for epoch in range(epoch_resume+1, total_epochs+1):

		print("Epoch %d" %epoch)
		lipsync_student.eval()

		total_loss = 0.0
		progress_bar = tqdm(enumerate(train_loader))

		for step, (inp_mel, inp_stft, gt_stft) in progress_bar:

			model.train()
			optimizer.zero_grad()					

			# Transform data to CUDA device
			inp_mel = inp_mel.to(device)										# BxTx1x80x16
			inp_stft = inp_stft.to(device)										# BxTx514
			gt_stft = gt_stft.to(device)										# BxTx514

			# Generate the faces using lipsync student model
			with torch.no_grad(): 
				faces = lipsync_student(inp_mel)								# Bx3xTx48x96

			# Generate the clean stft
			output_stft = model(inp_stft, faces)								# BxTx514

			# Compute the L1 reconstruction loss
			loss = l1_loss(output_stft, gt_stft)
			total_loss += loss.item()
			
			# Backpropagate
			loss.backward()
			optimizer.step()

			# Display the training progress
			progress_bar.set_description('Loss: {}'.format(total_loss / (step + 1))) 
			progress_bar.refresh()

		train_loss = total_loss / total_batch

		# Save the checkpoint
		if epoch % args.ckpt_freq == 0:

			# Save the model
			save_checkpoint(model, optimizer, train_loss, checkpoint_dir, epoch)

		# Validation loop
		if epoch % args.validation_interval == 0:
			with torch.no_grad():
				validate(device, lipsync_student, model, test_loader, epoch, checkpoint_dir)

	
def validate(device, lipsync_student, model, test_loader, epoch, checkpoint_dir):

	print('\nEvaluating for {} steps'.format(len(test_loader)))

	l1_loss = nn.L1Loss()

	losses = []

	for step, (inp_mel, inp_stft, gt_stft) in enumerate(test_loader):

		model.eval()

		# Transform data to CUDA device
		inp_mel = inp_mel.to(device)
		inp_stft = inp_stft.to(device)
		gt_stft = gt_stft.to(device)
		
		# Generate the faces using lipsync student model
		faces = lipsync_student(inp_mel)

		# Generate the clean stft
		output_stft = model(inp_stft, faces)

		# Compute the L1 reconstruction loss
		loss = l1_loss(output_stft, gt_stft)
		losses.append(loss.item())

	# Compute the average of the validation loss
	averaged_loss = sum(losses) / len(losses)
	print("Validation loss: ", averaged_loss)

	# Save the GT and the denoised files
	save_samples(gt_stft, inp_stft, output_stft, faces, epoch, checkpoint_dir)

	return
	
def save_checkpoint(model, optimizer, train_loss, checkpoint_dir, epoch):
	
	checkpoint_path = join(checkpoint_dir, "checkpoint_step{:04d}.pt".format(epoch))

	torch.save({
		"state_dict": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"loss": train_loss,
		"epoch": epoch,
	}, checkpoint_path)
	
	print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
	
	if use_cuda:
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

	return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False):

	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}

	for k, v in s.items():
		if hparams.hparams.n_gpu > 1:
			if not k.startswith('module.'):
				new_s['module.'+k] = v
			else:
				new_s[k] = v
		else:
			new_s[k.replace('module.', '')] = v

	model.load_state_dict(new_s)

	epoch_resume = 0
	if not reset_optimizer:
		optimizer_state = checkpoint["optimizer"]
		if optimizer_state is not None:
			print("Load optimizer state from {}".format(path))
			optimizer.load_state_dict(checkpoint["optimizer"])

		epoch_resume = checkpoint['epoch']
		loss = checkpoint['loss']

		print("Model resumed for training...")
		print("Epoch: ", epoch_resume)
		print("Loss: ", loss)
	
	return model, epoch_resume

if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--data_root_lrs3_pretrain", help="Root folder of the preprocessed LRS3 pre-train dataset", required=True, type=str)
	parser.add_argument("--data_root_lrs3_train", help="Root folder of the preprocessed LRS3 train dataset", required=True, type=str)
	parser.add_argument("--noise_data_root", help="Root folder of the VGGSound dataset", required=True, type=str)

	parser.add_argument('--lipsync_student_model_path', type=str, required=True, help='Path of the lipsync student model to generate frames')

	parser.add_argument('--checkpoint_dir', required=True, type=str, help='Folder to save the model')
	parser.add_argument('--checkpoint_path', default=None, type=str, help='Path of the saved model to resume training')

	parser.add_argument('--continue_epoch', default=True, help='Continue epoch number?')
	
	args = parser.parse_args()

	# Call the data generator to get the data
	train_loader = load_data(pretrain_path=args.data_root_lrs3_pretrain, train_path=args.data_root_lrs3_train, noise_path=args.noise_data_root, num_workers=hparams.hparams.num_workers, batch_size=hparams.hparams.batch_size, shuffle=True, split='train')

	total_batch = len(train_loader)
	print("Total train batch: ", total_batch)

	test_loader = load_data(pretrain_path=args.data_root_lrs3_pretrain, train_path=args.data_root_lrs3_train, noise_path=args.noise_data_root, num_workers=hparams.hparams.num_workers, batch_size=hparams.hparams.batch_size, shuffle=False, split='val')


	# Initialize lipsync student model
	lipsync_student = Lipsync_Student()
	if hparams.hparams.n_gpu > 1:
		print("Using", hparams.hparams.n_gpu, "GPUs for lipsync student model!")
		lipsync_student = nn.DataParallel(lipsync_student)
	else:
		print("Using single GPU for lipsync student model!")
	lipsync_student.to(device)

	# Load lipsync student model
	lipsync_student, _ = load_checkpoint(args.lipsync_student_model_path, lipsync_student, None, reset_optimizer=True)

	# Initialize the Denoising model 
	model = Model()
	if hparams.hparams.n_gpu > 1:
		print("Using", hparams.hparams.n_gpu, "GPUs for the denoising model!")
		model = nn.DataParallel(model)
	else:
		print("Using single GPU for the denoising model!")
	model.to(device)

	print('Total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

	# Set the learning rate
	if hparams.hparams.reduced_learning_rate is not None:
		lr = hparams.hparams.reduced_learning_rate
	else:
		lr = hparams.hparams.initial_learning_rate

	# Set the optimizer
	optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

	# Resume the denoising model for training if the path is provided
	epoch_resume=0
	if args.checkpoint_path is not None:
		model, epoch_resume = load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

	if args.continue_epoch==True:
		epoch = epoch_resume
	else:
		epoch = 0

	# Create the folder to save checkpoints
	checkpoint_dir = args.checkpoint_dir
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	# Train!
	train(device, lipsync_student, model, train_loader, test_loader, optimizer, epoch, hparams.hparams.nepochs, checkpoint_dir, args)

	print("Finished")
	