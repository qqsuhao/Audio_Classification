# -*- coding:utf8 -*-
# @TIME     : 2019/4/4 14:13
# @Author   : SuHao
# @File     : harmonic_test.py

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load(r'H:\毕业设计\boy_and_girl\class1\arctic_a0001.wav', duration=15)
D = librosa.stft(y)
H, P = librosa.decompose.hpss(D)


plt.figure()
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D)), y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Full power spectrogram')
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(H)), y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Harmonic power spectrogram')
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(P)), y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Percussive power spectrogram')
plt.tight_layout()
plt.show()