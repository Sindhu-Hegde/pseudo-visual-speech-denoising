import numpy as np
import cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
from models import *


def datagen(mels):

	mel_batch = []

	for i, m in enumerate(mels):

		mel_batch.append(m)

		if len(mel_batch) >= args.batch_size:

			mel_batch = np.asarray(mel_batch)
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield mel_batch
			mel_batch = []

	if len(mel_batch) > 0:
		
		mel_batch = np.asarray(mel_batch)
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield mel_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Lipsync_Student()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def main():
	
	fps = args.fps

	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp.wav')

		subprocess.call(command, shell=True)
		args.audio = 'temp.wav'

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1
 
	print("Length of mel chunks: {}".format(len(mel_chunks)))

	batch_size = args.batch_size
	gen = datagen(mel_chunks)

	# Create the folder to save the results
	if not os.path.exists(args.results_dir):
		os.makedirs(args.results_dir)

	vid_name = args.output_fname+'.avi'
	for i, mel_batch in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		
		if i == 0:
		
			model = load_model(args.checkpoint_path)
			print ("Model loaded")
			
			out = cv2.VideoWriter(os.path.join(args.results_dir, vid_name), cv2.VideoWriter_fourcc(*'DIVX'), fps, (args.img_size, args.img_size//2))

		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch)

		pred = (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

		for p in pred:

			out.write(p)

	out.release()

	out_name = args.output_fname+'_voice.mp4'
	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, os.path.join(args.results_dir, vid_name), os.path.join(args.results_dir, out_name))
	subprocess.call(command, shell=True)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Inference code to lip-sync any noisy videos using the student-lipsync models')

	parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)

	parser.add_argument('--face', type=str, default="checkpoints/taylor.jpg", help='Filepath of image that the student model is trained', required=False)
	parser.add_argument('--audio', type=str, help='Filepath of nosy video/audio file', required=True)

	parser.add_argument('--results_dir', type=str, help='Folder to save all results into', default='results/')

	parser.add_argument('--fps', type=float, help='FPS to generate the video', default=25., required=False)
	parser.add_argument('--batch_size', type=int, help='Batch size for the model', default=128)

	parser.add_argument('--output_fname', type=str, default='result', required=False, help='Name of the output file')

	args = parser.parse_args()
	args.img_size = 96

	main()