import os, pickle
import pdb
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
import random
from data import SARDataGenerator, generate_freq_measurements, generate_freq_measurements_randomsubsetmissing
# from sarnet_config import SARConfig as conf
# from sar_utilities import snr, snr_l1
from utils import snr_akshay
from tqdm import tqdm

# ## CASE 1 - block missing
# mode = 'test_real_onlyfirsttwoseqs'
# if mode == 'train':
#     input_filename = 'train_set_arun.pkl'
# elif mode == 'val':
#     input_filename = 'val_set_arun.pkl'
# elif mode == 'test_real_onlyfirsttwoseqs':
#     input_filename = 'test_set_real_onlyfirsttwoseqs.pkl'

# with open(os.path.join('data', input_filename), 'rb') as f:
#     dataset = pickle.load(f)

# missing_rate_list = [0.5, 0.6, 0.7, 0.8,  0.9]
# # missing_rate_list = [0.9]
# signals_list = []
# measurements_list = []
# signals = dataset['signals']
# for curr_missing_rate in missing_rate_list:
#     measurements = generate_freq_measurements(signals, curr_missing_rate)
#     signals_list.append(signals)
#     measurements_list.append(measurements)
# signals = np.concatenate(signals_list, axis=0)
# measurements = np.concatenate(measurements_list, axis=0)

# save_dict = {}
# save_dict['signals'], save_dict['measurements'] = signals, measurements

# # pdb.set_trace()

# if len(missing_rate_list) != 1:
#     with open(os.path.join('data', f'{mode}_set_arun_multiplemissingrates.pkl'), 'wb') as f:
#         pickle.dump(save_dict,f)
# else:
#     # pdb.set_trace()
#     with open(os.path.join('data', f'{mode}_set_arun_multiplemissingrates_{int(missing_rate_list[0]*100)}.pkl'), 'wb') as f:
#         pickle.dump(save_dict,f)

## CASE 2 - random missing
# mode = 'val'
mode = 'test_real_onlyfirsttwoseqs'
if mode == 'train':
    input_filename = 'train_set_arun.pkl'
elif mode == 'val':
    input_filename = 'val_set_arun.pkl'
elif mode == 'test_real_onlyfirsttwoseqs':
    input_filename = 'test_set_real_onlyfirsttwoseqs.pkl'

with open(os.path.join('data', input_filename), 'rb') as f:
    dataset = pickle.load(f)

missing_rate_list = [0.5, 0.6, 0.7, 0.8,  0.9]
# missing_rate_list = [0.9]
signals_list = []
measurements_list = []
signals = dataset['signals']
for curr_missing_rate in missing_rate_list:
    measurements = generate_freq_measurements_randomsubsetmissing(signals, curr_missing_rate)
    signals_list.append(signals)
    measurements_list.append(measurements)
    # pdb.set_trace()
signals = np.concatenate(signals_list, axis=0)
measurements = np.concatenate(measurements_list, axis=0)

save_dict = {}
save_dict['signals'], save_dict['measurements'] = signals, measurements

# pdb.set_trace()

if len(missing_rate_list) != 1:
    with open(os.path.join('data', f'{mode}_set_arun_multiplemissingrates_randomgaps.pkl'), 'wb') as f:
        pickle.dump(save_dict,f)
else:
    # pdb.set_trace()
    with open(os.path.join('data', f'{mode}_set_arun_multiplemissingrates_randomgaps_{int(missing_rate_list[0]*100)}.pkl'), 'wb') as f:
        pickle.dump(save_dict,f)
