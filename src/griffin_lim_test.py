import numpy as np
from synthesizer import audio
from synthesizer.hparams import hparams

# mel_prediction = np.load(r"D:\liuhaozhe\voice-cloning\src\saved_models\default\mel-spectrograms\mel-prediction-step-184228_sample_1.npy", allow_pickle=False)
mel_prediction = np.load(r"spec.npy", allow_pickle=False)
wav = audio.inv_mel_spectrogram(mel_prediction, hparams)
wav_fpath = "predicted_audio.wav"
audio.save_wav(wav, str(wav_fpath), sr=hparams.sample_rate)