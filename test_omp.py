import os, pickle
import pdb
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit

from data import SARDataGenerator, generate_freq_measurements
# from sarnet_config import SARConfig as conf
# from sar_utilities import snr, snr_l1
from utils import snr_akshay

dim = 1000
max_sparse=50 # Don't think this really matters... only want the dictionary
sparsity_pattern='block' # Don't think this really matters... only want the dictionary
support_dist='block-rootdec' # Don't think this really matters... only want the dictionary
scale=0.1 # Don't think this really matters... only want the dictionary
missing_rate = 0.5
num_omp_coefs = 50
results_snr = {}

template_path = 'data/SimTxPulse.mat'
template = sio.loadmat(template_path)['st']
padded_template = np.concatenate([template, np.zeros((dim-template.shape[0],1))], axis=0)
padded_template = padded_template.T[:,np.newaxis,:]
corrupted_atom = np.squeeze(generate_freq_measurements(padded_template, missing_rate))[:,np.newaxis]

corrupted_data_gen = SARDataGenerator(corrupted_atom,
                                      dim=dim,
                                      max_sparse=max_sparse,
                                      sparsity_pattern=sparsity_pattern,
                                      support_dist=support_dist,
                                      scale=scale)

datagen = SARDataGenerator(template,
                           dim=dim,
                           max_sparse=max_sparse,
                           sparsity_pattern=sparsity_pattern,
                           support_dist=support_dist,
                           scale=scale)

corrupted_dict = np.copy(corrupted_data_gen._dictionary)
clean_dict = np.copy(datagen._dictionary)

# signals, measurements = datagen.generate_batch(n_test, missing_rate=missing_rate)
# dataset_name = 'test_set_arun_2.pkl'
# dataset_name = 'test_set_real.pkl'
# dataset_name = 'test_set_real_C1.pkl'
# dataset_name = 'test_set_real_C2.pkl'

# dataset_names = ['test_set_real_C1.pkl', 'test_set_real_C2.pkl', 'test_set_real_C3.pkl', 'test_set_real_C4.pkl', 'test_set_real_C5.pkl',
#                 'test_set_real_T1.pkl', 'test_set_real_T2.pkl', 'test_set_real_T3.pkl', 'test_set_real_T4.pkl', 'test_set_real_T5.pkl']
                # 'test_set_real_G1.pkl', 'test_set_real_G2.pkl']
dataset_names = ['test_set_real_G2.pkl']
for dataset_name in dataset_names:
    with open(os.path.join('data', dataset_name), 'rb') as f:
        dataset = pickle.load(f)
    signals, measurements = dataset['signals'], dataset['measurements']
    result_list = []
    gain_list = []
    for num_omp_coefs in range(1,25+1):
        print(num_omp_coefs)
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=num_omp_coefs)
        omp.fit(corrupted_dict, measurements[:,0,:].T)
        preds = clean_dict.dot(omp.coef_.T).T[:,np.newaxis,:]
        SNR_meas, SNR_pred, SNR_gain = snr_akshay(signals, measurements, preds)
        print("Sim OMP L2 SNR_in: %.6f, SNR_out: %.6f, SNR_gain: %.6f" %(SNR_meas, SNR_pred, SNR_gain))
        results_snr['sim'] = [SNR_meas, SNR_pred, SNR_gain]
        result_list.append([SNR_meas, SNR_pred, SNR_gain])
        gain_list.append(SNR_gain)
    pdb.set_trace()
    # data_dir = 'real_sar_data'
    # for s in os.listdir(data_dir):
    #     if 'mat' in s:
    #         real_signals = sio.loadmat(os.path.join(data_dir, s))['Data'].T[:,np.newaxis,:]
    #         if real_signals.shape[-1] != conf.dim:
    #             padding = conf.dim - real_signals.shape[-1]
    #             n_signals = real_signals.shape[0]
    #             real_signals = np.concatenate([real_signals, np.zeros((n_signals,1,padding))],axis=-1)
    #             # real_signals = np.concatenate([np.zeros((n_signals,1,padding)), real_signals],axis=-1)
    #         # multiplier = 1.0/np.max(np.abs(real_signals))
    #         # real_signals = multiplier*real_signals
    #         measurements = generate_freq_measurements(real_signals, missing_rate)
    #         omp = OrthogonalMatchingPursuit(n_nonzero_coefs=num_omp_coefs)
    #         omp.fit(corrupted_dict, measurements[:,0,:].T)
    #         preds = clean_dict.dot(omp.coef_.T).T[:,np.newaxis,:]

    #         SNR_meas, SNR_pred, SNR_gain = snr(real_signals, measurements, preds)
    #         print("%s OMP L2 SNR_in: %.6f, SNR_out: %.6f, SNR_gain: %.6f" %(s, SNR_meas, SNR_pred, SNR_gain))
    #         results_snr[s] = [SNR_meas, SNR_pred, SNR_gain]

    # with open('results/results_omp.pkl', 'wb') as f:
    #     pickle.dump(results_snr, f)
