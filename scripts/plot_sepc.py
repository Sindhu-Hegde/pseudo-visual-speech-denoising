import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../audio/")
import hparams as hp
import audio_utils as audio
import librosa
import librosa.display

def plot(file, fname):

	wav = librosa.load(file, sr=16000)[0]
	stft = librosa.stft(y=wav, n_fft=hp.hparams.n_fft_den, hop_length=hp.hparams.hop_size_den, win_length=hp.hparams.win_size_den)
	print("STFT: ", stft.shape)

	# Display magnitude spectrogram
	D = np.abs(stft)
	librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),y_axis='log', x_axis='time')
	plt.title('Power spectrogram')
	plt.colorbar(format='%+2.0f dB')
	plt.tight_layout()
	plt.show()
	plt.savefig(fname+".jpg")
	plt.clf()


if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--gt_file', type=str, required=True, help='GT wav file')
	parser.add_argument('--noisy_file', type=str, required=True, help='Noisy wav file')
	parser.add_argument('--pred_file', type=str, required=True, help='Predicted wav file')
	args = parser.parse_args()


	plot(args.gt_file, 'gt')
	plot(args.noisy_file, 'noisy')
	plot(args.pred_file, 'pred')
