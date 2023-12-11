from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage

import numpy as np
import scipy
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model

windowLength = 256
overlap      = round(0.25 * windowLength) # overlap of 75%
ffTLength    = windowLength
inputFs      = 48e3
fs           = 8e3
numFeatures  = ffTLength//2 + 1
numSegments  = 8

class FeatureExtractor:
    def __init__(self, audio, *, windowLength, overlap, sample_rate):
        self.audio = audio
        self.ffT_length = windowLength
        self.window_length = windowLength
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.window = scipy.signal.windows.hamming(self.window_length, sym=False)

    def get_stft_spectrogram(self):
        return librosa.stft(self.audio, n_fft=self.ffT_length, win_length=self.window_length, hop_length=self.overlap,
                            window=self.window, center=True)

    def get_audio_from_stft_spectrogram(self, stft_features):
        return librosa.istft(stft_features, win_length=self.window_length, hop_length=self.overlap,
                             window=self.window, center=True)

    def get_mel_spectrogram(self):
        return librosa.feature.melspectrogram(self.audio, sr=self.sample_rate, power=2.0, pad_mode='reflect',
                                              n_fft=self.ffT_length, hop_length=self.overlap, center=True)

    def get_audio_from_mel_spectrogram(self, M):
        return librosa.feature.inverse.mel_to_audio(M, sr=self.sample_rate, n_fft=self.ffT_length,
                                                    hop_length=self.overlap,
                                                    win_length=self.window_length, window=self.window,
                                                    center=True, pad_mode='reflect', power=2.0, n_iter=32, length=None)

def index(request):
    if request.method == 'POST':
        # Assuming you have a form with 'audio_file' as the file input name
        audio_file = request.FILES.get('audio_file')

        file_path = default_storage.save(r"C:\Users\shrey\Downloads\myproject\myproject\audio.wav", audio_file)
        print(file_path)

        # Perform the inference on the audio file and get the denoised audio
        perform_inference(file_path)

        return render(request, 'results.html')

    return render(request, 'index.html')


def read_audio(filepath, sample_rate, normalize=True):
    audio, sr = librosa.load(filepath, sr=sample_rate)
    if normalize is True:
        div_fac = 1 / np.max(np.abs(audio)) / 3.0
        audio = audio * div_fac
        # audio = librosa.util.normalize(audio)
    return audio, sr

def prepare_input_features(stft_features):
    # Phase Aware Scaling: To avoid extreme differences (more than
    # 45 degree) between the noisy and clean phase, the clean spectral magnitude was encoded as similar to [21]:
    noisySTFT = np.concatenate([stft_features[:,0:numSegments-1], stft_features], axis=1)
    stftSegments = np.zeros((numFeatures, numSegments , noisySTFT.shape[1] - numSegments + 1))

    for index in range(noisySTFT.shape[1] - numSegments + 1):
        stftSegments[:,:,index] = noisySTFT[:,index:index + numSegments]
    return stftSegments

def revert_features_to_audio(features, noiseAudioFeatureExtractor, phase, cleanMean=None, cleanStd=None):
    # scale the outpus back to the original range
    if cleanMean and cleanStd:
        features = cleanStd * features + cleanMean

    phase = np.transpose(phase, (1, 0))
    features = np.squeeze(features)

    # features = librosa.db_to_power(features)
    features = features * np.exp(1j * phase)  # that fixes the abs() ope previously done

    features = np.transpose(features, (1, 0))
    return noiseAudioFeatureExtractor.get_audio_from_stft_spectrogram(features)

def add_noise_to_clean_audio(clean_audio, noise_signal):
    if len(clean_audio) >= len(noise_signal):
        # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
        while len(clean_audio) >= len(noise_signal):
            noise_signal = np.append(noise_signal, noise_signal)

    ## Extract a noise segment from a random location in the noise file
    ind = np.random.randint(0, noise_signal.size - clean_audio.size)

    noiseSegment = noise_signal[ind: ind + clean_audio.size]

    speech_power = np.sum(clean_audio ** 2)
    noise_power = np.sum(noiseSegment ** 2)
    noisyAudio = clean_audio + np.sqrt(speech_power / noise_power) * noiseSegment
    return noisyAudio

def perform_inference(file_path):
    noisy_audio, sr = read_audio(file_path, fs) 
    print(noisy_audio)
    noiseAudioFeatureExtractor = FeatureExtractor(noisy_audio, windowLength=windowLength, overlap=overlap, sample_rate=sr)
    noise_stft_features = noiseAudioFeatureExtractor.get_stft_spectrogram()

    # Paper: Besides, spectral phase was not used in the training phase.
    # At reconstruction, noisy spectral phase was used instead to
    # perform in- verse STFT and recover human speech.
    noisyPhase = np.angle(noise_stft_features)
    print(noisyPhase.shape)
    noise_stft_features = np.abs(noise_stft_features)

    mean = np.mean(noise_stft_features)
    std = np.std(noise_stft_features)
    noise_stft_features = (noise_stft_features - mean) / std

    predictors = prepare_input_features(noise_stft_features)

    predictors = np.reshape(predictors, (predictors.shape[0], predictors.shape[1], 1, predictors.shape[2]))
    predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)
    print('predictors.shape:', predictors.shape)

    inception_resnet_model = load_model('inception_resnet_model.h5')
    inception_resnet_model_pred = inception_resnet_model.predict(predictors)
    print(inception_resnet_model_pred.shape)

    denoisedAudioFullyConvolutional = revert_features_to_audio(inception_resnet_model_pred, noiseAudioFeatureExtractor, noisyPhase, mean, std)
    print("Min:", np.min(denoisedAudioFullyConvolutional),"Max:",np.max(denoisedAudioFullyConvolutional))

    print(denoisedAudioFullyConvolutional)
    # Assuming denoised_audio is your NumPy array containing the denoised audio
    sr = int(sr)
    print(sr)
    audio_data = (denoisedAudioFullyConvolutional * 32767).astype(np.int16)
    sf.write(r"C:\Users\shrey\Downloads\myproject\myproject\myapp\static\output.wav", audio_data, sr, 'PCM_24')
    
