import os, pickle
import pdb
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
import random
from data import SARDataGenerator, generate_freq_measurements
# from sarnet_config import SARConfig as conf
# from sar_utilities import snr, snr_l1
from utils import snr_akshay
from tqdm import tqdm
import h5py
from normalize_easy import normc

mode = 'train'
if mode == 'train':
    try:
        dataset = sio.loadmat(os.path.join('data', 'mat_data', 'train_simulated_extended.mat'))
    except NotImplementedError:
        dataset = {}
        with h5py.File(os.path.join('data', 'mat_data', 'train_simulated_extended.mat'), 'r') as f:
            for k, v in f.items():
                dataset[k] = np.array(v).T # Need the transpose because for some reason it swaps shape...


signals_loaded = dataset['signals']
# IMPORTANT: Always normalize!!!! Better safe than sorry.
signals_loaded = normc(signals_loaded)
missing_rate = 0.5

# pdb.set_trace()

signals = signals_loaded
signals = signals.T[:,np.newaxis,:]
measurements = generate_freq_measurements(signals, missing_rate=missing_rate)

save_dict = {}
save_dict['signals'], save_dict['measurements'] = signals, measurements
save_dict['missing_rate'] = missing_rate

pdb.set_trace()
with open(os.path.join('data', f'{mode}_set_arun_extended.pkl'), 'wb') as f:
    pickle.dump(save_dict,f)
