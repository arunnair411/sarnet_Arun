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

mode = 'train_fwd_exact'
if mode == 'train_fwd_exact':
    dataset = sio.loadmat(os.path.join('data', 'mat_data', 'forward-looking', 'exact_Arun_training_20201101.mat'))
elif mode == 'train_fwd_generated':
    dataset = sio.loadmat(os.path.join('data', 'mat_data', 'forward-looking', 'generated_Arun_training_20201101.mat'))
elif mode == 'val_fwd':
    dataset = sio.loadmat(os.path.join('data', 'mat_data', 'forward-looking', 'exact_Arun_validation_20201101.mat'))
elif mode == 'test_fwd':
    dataset = sio.loadmat(os.path.join('data', 'mat_data', 'forward-looking', 'exact_Arun_testing_20201101.mat'))


clean_dict = sio.loadmat(os.path.join('data', 'mat_data', 'forward-looking', 'clean_dict.mat'))['clean_dict']

# random.seed(seed)
# np.random.seed(seed)
signals_loaded = dataset['signals']
num_omp_coefs = 200
missing_rate = 0.5
energy_band=(380e6, 1300e6)

# omp = OrthogonalMatchingPursuit(n_nonzero_coefs=num_omp_coefs)
# omp.fit(clean_dict, signals_loaded[:,0,:].T)
# preds = clean_dict.dot(omp.coef_.T).T[:,np.newaxis,:]
signals = signals_loaded
signals = signals.T[:,np.newaxis,:]
# pdb.set_trace()
measurements = generate_freq_measurements(signals, missing_rate, energy_band)

save_dict = {}
save_dict['signals'], save_dict['measurements'] = signals, measurements
save_dict['missing_rate'] = missing_rate

pdb.set_trace()
with open(os.path.join('data', f'{mode}_set_arun.pkl'), 'wb') as f:
    pickle.dump(save_dict,f)
