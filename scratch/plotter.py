import matplotlib.pyplot as plt
from data import generate_freq_measurements
import numpy as np
import scipy.io as sio
import os, pickle

with open(os.path.join('results', '20201008_arun_interference_testonfirsttwoseqs_0', 'test', 'results.pkl' ), 'rb') as f:
    dataset_interference =  pickle.load(f)

with open(os.path.join('results', '20201008_arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps_50', 'test', 'results.pkl' ), 'rb') as f:
    dataset_randommissing =  pickle.load(f)

with open(os.path.join('results', '20201008_arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_50', 'test', 'results.pkl' ), 'rb') as f:
    dataset_gapmissing =  pickle.load(f)

plt.subplot(331)
data = np.squeeze(dataset_interference['measurements']).T[:,:1800]
data = data/np.max(np.abs(data))
data = data + 1e-32
data_logcompressed = 20*np.log10(np.abs(data))
plt.imshow(data_logcompressed, vmin=-40, vmax=0)
plt.subplot(332)
data = np.squeeze(dataset_interference['signals']).T[:,:1800]
data = data/np.max(np.abs(data))
data = data + 1e-32
data_logcompressed = 20*np.log10(np.abs(data))
plt.imshow(data_logcompressed, vmin=-40, vmax=0)
plt.subplot(333)
data = np.squeeze(dataset_interference['outputs']).T[:,:1800]
data = data/np.max(np.abs(data))
data = data + 1e-32
data_logcompressed = 20*np.log10(np.abs(data))
plt.imshow(data_logcompressed, vmin=-40, vmax=0)
plt.subplot(334)
data = np.squeeze(dataset_randommissing['measurements']).T[:,:1800]
data = data/np.max(np.abs(data))
data = data + 1e-32
data_logcompressed = 20*np.log10(np.abs(data))
plt.imshow(data_logcompressed, vmin=-40, vmax=0)
plt.subplot(335)
data = np.squeeze(dataset_randommissing['signals']).T[:,:1800]
data = data/np.max(np.abs(data))
data = data + 1e-32
data_logcompressed = 20*np.log10(np.abs(data))
plt.imshow(data_logcompressed, vmin=-40, vmax=0)
plt.subplot(336)
data = np.squeeze(dataset_randommissing['outputs']).T[:,:1800]
data = data/np.max(np.abs(data))
data = data + 1e-32
data_logcompressed = 20*np.log10(np.abs(data))
plt.imshow(data_logcompressed, vmin=-40, vmax=0)
plt.subplot(337)
data = np.squeeze(dataset_gapmissing['measurements']).T[:,:1800]
data = data/np.max(np.abs(data))
data = data + 1e-32
data_logcompressed = 20*np.log10(np.abs(data))
plt.imshow(data_logcompressed, vmin=-40, vmax=0)
plt.subplot(338)
data = np.squeeze(dataset_gapmissing['signals']).T[:,:1800]
data = data/np.max(np.abs(data))
data = data + 1e-32
data_logcompressed = 20*np.log10(np.abs(data))
plt.imshow(data_logcompressed, vmin=-40, vmax=0)
plt.subplot(339)
data = np.squeeze(dataset_gapmissing['outputs']).T[:,:1800]
data = data/np.max(np.abs(data))
data = data + 1e-32
data_logcompressed = 20*np.log10(np.abs(data))
plt.imshow(data_logcompressed, vmin=-40, vmax=0)
plt.savefig('temp.png')
plt.close()


# signals, measurements = dataset['signals'], dataset['measurements']
# chosen_idx = 250000

# plt.subplot(611)
# plt.plot(np.squeeze(signals[chosen_idx]))
# plt.subplot(612)
# plt.plot(np.abs(np.fft.rfft(np.squeeze(signals[chosen_idx]))))
# plt.subplot(613)
# plt.plot(np.squeeze(measurements[chosen_idx]))
# plt.subplot(614)
# plt.plot(np.abs(np.fft.rfft(np.squeeze(measurements[chosen_idx]))))
# plt.savefig('temp.png')
# plt.close()
# temp_1 = np.abs(np.fft.rfft(np.squeeze(signals[chosen_idx])))
# temp_2 = np.abs(np.fft.rfft(np.squeeze(measurements[chosen_idx])))
# abs(temp_2[11:47])<1e-6
# sum(abs(temp_2[11:47])<1e-6)
# missing_rate = 0.1
# energy_band=(380e6, 1300e6)
# measurements_2 = generate_freq_measurements(signals, missing_rate, energy_band)
# plt.subplot(615)
# plt.plot(np.squeeze(measurements_2[chosen_idx]))
# plt.subplot(616)
# plt.plot(np.abs(np.fft.rfft(np.squeeze(measurements_2[chosen_idx]))))
# plt.savefig('temp.png')
# plt.close()
