import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

class Conv1d(nn.Module):

	def __init__(self, cin, cout, kernel_size, stride=1, padding=1, residual=False, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.conv_block = nn.Sequential(
							nn.Conv1d(cin, cout, kernel_size, stride, padding),
							nn.BatchNorm1d(cout)
							)
		self.act = nn.ReLU()
		self.residual = residual

	def forward(self, x):
		out = self.conv_block(x)
		if self.residual:
			out += x
		return self.act(out)


class Conv2d(nn.Module):
	
	def __init__(self, cin, cout, kernel_size, stride=1, padding=1, residual=False, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.conv_block = nn.Sequential(
							nn.Conv2d(cin, cout, kernel_size, stride, padding),
							nn.BatchNorm2d(cout)
							)
		self.act = nn.ReLU()
		self.residual = residual

	def forward(self, x):
		out = self.conv_block(x)
		if self.residual:
			out += x
		return self.act(out)


class Conv2dTranspose(nn.Module):

	def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.conv_block = nn.Sequential(
							nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
							nn.BatchNorm2d(cout)
							)
		self.act = nn.ReLU()

	def forward(self, x):
		out = self.conv_block(x)
		return self.act(out)


class Lipsync_Student(nn.Module):
    def __init__(self):
        super(Lipsync_Student, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=3, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(Conv2dTranspose(512, 512, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12

            nn.Sequential(Conv2dTranspose(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            nn.Sequential(Conv2dTranspose(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            nn.Sequential(Conv2dTranspose(64, 32, kernel_size=3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),),]) # 48,96

        self.output_block = nn.Sequential(Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 

    def forward(self, audio_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(audio_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)

        x = self.audio_encoder(audio_sequences) # B, 512, 1, 1
        for f in self.face_decoder_blocks:
            x = f(x)

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs
        


class Model(nn.Module):

	def __init__(self):
		super(Model, self).__init__()


		self.audio_encoder = nn.Sequential(
			Conv1d(514, 600, kernel_size=3, stride=1),
			Conv1d(600, 600, kernel_size=3, stride=1, residual=True),
			Conv1d(600, 600, kernel_size=3, stride=1, residual=True),
			Conv1d(600, 600, kernel_size=3, stride=1, residual=True),
			Conv1d(600, 600, kernel_size=3, stride=1, residual=True),
			Conv1d(600, 600, kernel_size=3, stride=1, residual=True),
			Conv1d(600, 600, kernel_size=3, stride=1)
			) 


		self.face_encoder = nn.Sequential(
			Conv2d(3, 32, kernel_size=5, stride=(1,2), padding=2),             # Bx32x25x48x48
			Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),      

			Conv2d(32, 64, kernel_size=3, stride=(2,2), padding=1),            # Bx64x25x24x24
			Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

			Conv2d(64, 128, kernel_size=3, stride=(2,2), padding=1),           # Bx128x25x12x12
			Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

			Conv2d(128, 256, kernel_size=3, stride=(2,2), padding=1),          # Bx256x25x6x6
			Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

			Conv2d(256, 512, kernel_size=3, stride=(2,2), padding=1),          # Bx512x25x3x3
			Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
			
			Conv2d(512, 512, kernel_size=3, stride=(3,3), padding=1),          # Bx512x25x1x1
			Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
			)

		
		self.time_upsampler = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.Upsample(scale_factor=2, mode='nearest'),
			Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
		)

		self.decoder = nn.Sequential(
			Conv1d(1112, 1024, kernel_size=3, stride=1),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			nn.Conv1d(1024, 514, kernel_size=1, stride=1, padding=0)
		)


	def forward(self, stft_sequence, face_sequence):	

		# -----------------------------Face----------------------------------- #
		# print("Face input: ", face_sequence.size())						# Bx3xTx48x96

		B = face_sequence.size(0)
		face_sequence = torch.cat([face_sequence[:, :, i] for i in range(face_sequence.size(2))], dim=0)
		# print("Face sequence concatenated: ", face_sequence.size())		# (B*T)x3x48x96
		
		# Face encoder
		face_enc = self.face_encoder(face_sequence)							# (B*T)x512x1x1
		face_enc = torch.split(face_enc, B, dim=0) 					
		face_enc = torch.stack(face_enc, dim=2) 							# Bx512xTx1x1				

		face_enc = face_enc.view(-1, face_enc.size(1), face_enc.size(2))	# Bx512xT

		face_output = self.time_upsampler(face_enc)							# Bx512x(T*4)
		# -------------------------------------------------------------------- #

		# -------------------------- Audio ------------------------------- #
		
		# print("STFT input: ", stft_sequence.size())						# BxTx514

		stft_sequence_permuted = stft_sequence.permute(0, 2, 1)				# Bx514xT

		# Audio encoder
		audio_enc = self.audio_encoder(stft_sequence_permuted)				# Bx600xT

		# Concatenate face network output and audio encoder output
		concatenated = torch.cat([audio_enc, face_output], dim=1)			# Bx1112xT

		# Audio decoder
		dec = self.decoder(concatenated)									# Bx514xT

		# Mask
		mask = dec.permute(0, 2, 1)											# BxTx514

		# Add the mask with the input noisy spec
		output = mask + stft_sequence
		output = torch.sigmoid(output)										# BxTx514
		# -------------------------------------------------------------------- #

		return output



