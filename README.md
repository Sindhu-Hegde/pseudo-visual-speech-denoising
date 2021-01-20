

# Pseudo-Visual Speech Denoising

This code is for our paper titled: *Visual Speech Enhancement Without A Real Visual Stream* published at WACV 2021.<br />
**Authors**: Sindhu Hegde*, K R Prajwal*, Rudrabha Mukhopadhyay*, Vinay Namboodiri, C.V. Jawahar

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/visual-speech-enhancement-without-a-real/speech-denoising-on-lrs3-vggsound)](https://paperswithcode.com/sota/speech-denoising-on-lrs3-vggsound?p=visual-speech-enhancement-without-a-real)     [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/visual-speech-enhancement-without-a-real/speech-denoising-on-lrs2-vggsound)](https://paperswithcode.com/sota/speech-denoising-on-lrs2-vggsound?p=visual-speech-enhancement-without-a-real)

|   📝 Paper   |   📑 Project Page    |  🛠 Demo Video  |  🗃 Real-World Test Set |
|-----------|-------------------|---------------|------------------------|
|[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Hegde_Visual_Speech_Enhancement_Without_a_Real_Visual_Stream_WACV_2021_paper.pdf) | [Website](http://cvit.iiit.ac.in/research/projects/cvit-projects/visual-speech-enhancement-without-a-real-visual-stream/) |[Video](https://youtu.be/y_oP9t7WEn4) | [Real-World Test Set (coming soon)](https://github.com/Sindhu-Hegde/pseudo-visual-speech-denoising#)
<br />
<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1y9FfNJIl5dI6_Luz6a7I_RWXHY7ZCF8_">
</p> 

------
**Features**
--------
- Denoise any real-world audio/video and obtain the clean speech.
- Works in unconstrained settings for any speaker in any language.
- Inputs only audio but uses the benefits of lip movements by generating a synthetic visual stream. 
- Complete training code and inference codes available. 

----
Prerequisites
---
- `Python 3.7.4` (Code has been tested with this version)
- ffmpeg: `sudo apt-get install ffmpeg`
- Install necessary packages using `pip install -r requirements.txt`
- Face detection [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) should be downloaded to `face_detection/detection/sfd/s3fd.pth`
-----
Getting the weights
-----


| Model  | Description |  Link to the model | 
| :-------------: | :---------------: | :---------------: |
| Denoising model  | Weights of the denoising model (needed for inference) | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sindhu_hegde_research_iiit_ac_in/Ea3mavZJa75Iu1nKUTvO9TwBro1ByZPyqF2dXYqrHQLQtA?e=UNtj54) |---
| Lipsync student  | Weights of the student lipsync model to generate the visual stream for noisy audio inputs (needed for inference)| [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sindhu_hegde_research_iiit_ac_in/EUR-4Fbq_11Dm5xzE5BpG8YBNVHqRi4cn0fabni74Zlauw?e=zl0AxL) |
| Wav2Lip teacher  |Weights of the teacher lipsync model (only needed if you want to train the network from scratch) | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW)  |

---
Denoising any audio/video using the pre-trained model (Inference)
----
You can denoise any noisy audio/video and obtain the clean speech of the target speaker using:

    python inference.py --lipsync_student_model_path=<trained-student-model-ckpt-path> --checkpoint_path=<trained-denoising-model-ckpt-path> --input=<noisy-audio/video-file>

The result is saved (by default) in `results/result.mp4`. The result directory can be specified in arguments, similar to several other available options. The input file can be any audio file: `*.wav`, `*.mp3` or even a video file, from which the code will automatically extract the audio and generate the clean speech. Note that the noise should not be human speech, as this work only tackles the denoising task, not speaker separation.

#### Generating only the lip-movements for any given noisy audio/video
The synthetic visual stream (lip-movements) can be generated for any noisy audio/video using:

    cd lipsync
    python inference.py --checkpoint_path=<trained-lipsync-student-model-ckpt-path> --audio=<path-of-noisy-audio/video>

The result is saved (by default) in `results/result_voice.mp4`. The result directory can be specified in arguments, similar to several other available options. The input file can be any audio file: `*.wav`, `*.mp3` or even a video file, from which the code will automatically extract the audio and generate the visual stream.

# Training

We illustrate the training process using the [LRS3](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html) and [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) dataset. Adapting for other datasets would involve small modifications to the code.

### Preprocess the dataset
##### LRS3 train-val/pre-train dataset folder structure

```
data_root (we use both train-val and pre-train sets of LSR3 dataset in this work)
├── list of folders
│   ├── five-digit numbered video IDs ending with (.mp4)
```

##### Preprocess the dataset

    python preprocess.py --data_root=<dataset-path> --preprocessed_root=<path-to-save-the-preprocessed-data>

Additional options like `batch_size` and number of GPUs to use in parallel to use can also be set.

##### Preprocessed LRS3 folder structure

```
preprocessed_root (lrs3_preprocessed)
├── list of folders
|	├── Folders with five-digit numbered video IDs
|	│   ├── *.jpg (extracted face crops from each frame)
```

##### VGGSound folder structure

We use [VGGSound dataset](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) as noisy data which is mixed with the clean speech from LRS3 dataset. We download the audio files (`*.wav files`) from [here](https://www.robots.ox.ac.uk/~vgg/data/vggsound/). 

```
data_root (vgg_sound)
├── *.wav (audio files)
```

## Train!

There are two major steps: (i) Train the student-lipsync model, (ii) Train the Denoising model.

### Train the Student-Lipsync model
Navigate to the lipsync folder: `cd lipsync`

The lipsync model can be trained using:

    python train_student.py --data_root_lrs3_pretrain=<path-of-preprocessed-LRS3-pretrain-set> --data_root_lrs3_train=<path-of-preprocessed-LRS3-train-set> --noise_data_root=<path-of-VGGSound-dataset-to-mix-with-clean-speech> --wav2lip_checkpoint_path=<pretrained-wav2lip-teacher-model-ckpt-path> --checkpoint_dir=<path-to-save-the-trained-student-lipsync-model>

**Note:** The pre-trained Wav2Lip teacher model must be downloaded ([wav2lip weights](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW))  before training the student model.
  
### Train the Denoising model!
Navigate to the main directory: `cd ..`

The denoising model can be trained using:

    python train.py --data_root_lrs3_pretrain=<path-of-preprocessed-LRS3-pretrain-set> --data_root_lrs3_train=<path-of-preprocessed-LRS3-train-set> --noise_data_root=<path-of-VGGSound-dataset-to-mix-with-clean-speech> --lipsync_student_model_path=<trained-student-lipsync-model-ckpt-path> --checkpoint_dir=<path-to-save-the-trained-student-lipsync-model>
    
The model can be resumed for training as well. Look at `python train.py --help` for more details. Also, additional less commonly-used hyper-parameters can be set at the bottom of the `audio/hparams.py` file.

----
Evaluation
---
To be updated soon!

---
Licence and Citation
---
The software is licensed under the MIT License. Please cite the following paper if you have used this code:

```
@InProceedings{Hegde_2021_WACV,
    author    = {Hegde, Sindhu B. and Prajwal, K.R. and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
    title     = {Visual Speech Enhancement Without a Real Visual Stream},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {1926-1935}
}
```

---
Acknowledgements
---
Parts of the lipsync code has been modified using our [Wav2Lip repository](https://github.com/Rudrabha/Wav2Lip). The audio functions and parameters are taken from this [TTS repository](https://github.com/r9y9/deepvoice3_pytorch). We thank the authors for this wonderful code. The code for Face Detection has been taken from the [face_alignment](https://github.com/1adrianb/face-alignment) repository. We thank the authors for releasing their code and models.
