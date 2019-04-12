# -*- coding:utf8 -*-
# @TIME     : 2019/4/11 18:30
# @Author   : SuHao
# @File     : hilberfilter.py


import scipy.signal as signal
import numpy as np
import librosa as lib
import matplotlib.pyplot as plt
import time

ex = '..\\..\\cello_and_viola\\viola\\Viola.arco.ff.sulA.A4.stereo.aiff'
time_series, fs = lib.load(ex, sr=None, mono=True, res_type='kaiser_best')


#duration = 2.0
#fs = 400.0
#samples = int(fs*duration)
#t = np.arange(samples) / fs
#time_series = signal.chirp(t, 20.0, t[-1], 100.0)
#time_series *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )





window = signal.get_window(('kaiser', 1), 201, fftbins=False)
co = signal.firwin(numtaps=201,
                   cutoff=[0.001*fs/2/np.pi, (np.pi-0.001)*fs/2/np.pi],
                   width=None,
                   window='hann',
                   pass_zero=False,
                   scale=True,
                   nyq=None,
                   fs=fs)
w, h = signal.freqz(b=co, a=1, worN=2048, whole=False, plot=None, fs=2*np.pi)

#fig, ax1 = plt.subplots()
#ax1.set_title('Digital filter frequency response')
#ax1.plot(w, 20 * np.log10(abs(h)), 'b')
#ax1.set_ylabel('Amplitude [dB]', color='b')
#ax1.set_xlabel('Frequency [rad/sample]')
#
#ax2 = ax1.twinx()
#angles = np.unwrap(np.angle(h))
#ax2.plot(w, angles, 'g')
#ax2.set_ylabel('Angle (radians)', color='g')
#ax2.grid()
#ax2.axis('tight')
#plt.show()
#
#
#plt.figure()
#plt.plot(window)
#
#plt.figure()
#plt.plot(co, '*')


start = time.time()
out = signal.filtfilt(b=co, a=1, x=time_series, padlen=100)
envolope = np.sqrt(out**2 + time_series**2)
end = time.time()
a = end-start
print(a)


#sos = signal.tf2sos(b=co, a=1, pairing='nearest')
#out = signal.sosfilt(sos, time_series)
#out = signal.filtfilt(b=co, a=1, x=time_series, padlen=100)
#out = signal.convolve(co, time_series)
envolope = np.sqrt(out**2 + time_series**2)
plt.figure()
plt.plot(envolope)
plt.plot(time_series)
plt.show()

start = time.time()
env = np.abs(signal.hilbert(time_series))
end = time.time()
a=end-start
print(a)


plt.figure()
plt.plot(env)
plt.plot(time_series)
plt.show()


