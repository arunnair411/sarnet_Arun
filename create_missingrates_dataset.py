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

case = 'block_missing' ## CASE 1 - block missing
# case = 'random_missing' ## CASE 2 - random missing

case_function_dict = {'random_missing': generate_freq_measurements_randomsubsetmissing, 'block_missing': generate_freq_measurements}
case_filename_dict = {'random_missing': 'randomgaps', 'block_missing': 'blockgaps'}
# dataset_names, is_train = ['train_set_arun_extended.pkl', 'train_set_arun_generative_modeled_extended.pkl'], True 
# dataset_names, is_train = ['val_set_arun.pkl', 'val_set_arun_generative_modeled.pkl'], False
dataset_names, is_train = ['test_set_real_onlyfirsttwoseqs.pkl'], False
for dataset_name in dataset_names:   
    with open(os.path.join('data', dataset_name), 'rb') as f:
        dataset = pickle.load(f)
    signals = dataset['signals']
    missing_rate_list = [0.5, 0.6, 0.7, 0.8,  0.9]
    
    for curr_missing_rate in tqdm(missing_rate_list):
        measurements = case_function_dict[case](signals, curr_missing_rate)
        save_dict = {}
        save_dict['signals'], save_dict['measurements'] = signals, measurements
        output_file_name = dataset_name.split('.')[0]
        with open(os.path.join('data', f'{output_file_name}_{case_filename_dict[case]}_{int(curr_missing_rate*100)}.pkl'), 'wb') as f:
            pickle.dump(save_dict,f)
