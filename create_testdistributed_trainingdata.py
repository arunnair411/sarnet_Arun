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

mode = 'val_CTsplit'
if mode == 'train':
    n_data = 50000
    inner_loop_total = 5000
    seed = 0
elif mode == 'val':
    n_data = 6250
    inner_loop_total = 625
    seed = 100
elif mode == 'train_Csplit':
    n_data = 50000
    inner_loop_total = 10000
    seed = 200
elif mode == 'val_Csplit':
    n_data = 6250
    inner_loop_total = 1250
    seed = 300
elif mode == 'train_Tsplit':
    n_data = 50000
    inner_loop_total = 10000
    seed = 400
elif mode == 'val_Tsplit':
    n_data = 6250
    inner_loop_total = 1250
    seed = 500    
elif mode == 'train_CTsplit':
    n_data = 50000
    inner_loop_total = 10000
    seed = 600
elif mode == 'val_CTsplit':
    n_data = 6250
    inner_loop_total = 1250
    seed = 700    

# dim = 1000
dim = 1024
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

# signals, measurements = datagen.generate_batch(n_test, missing_rate=missing_rate) # signals is n_datax1xdim
# dataset_name = 'test_set_arun_2.pkl'
# dataset_name = 'test_set_real.pkl'
# dataset_name = 'test_set_real_C1.pkl'
# dataset_name = 'test_set_real_C2.pkl'

dataset_names = ['test_set_real_C1.pkl', 'test_set_real_C2.pkl', 'test_set_real_C3.pkl', 'test_set_real_C4.pkl', 'test_set_real_C5.pkl',
                'test_set_real_T1.pkl', 'test_set_real_T2.pkl', 'test_set_real_T3.pkl', 'test_set_real_T4.pkl', 'test_set_real_T5.pkl']
                # 'test_set_real_G1.pkl', 'test_set_real_G2.pkl'] # Removing G_1 and G_2.pkl
# dataset_names = ['test_set_real_C1.pkl']
pdb.set_trace()
if mode in ['train_Csplit', 'val_Csplit']:
    dataset_names = dataset_names[0:5]
elif mode in ['train_Tsplit', 'val_Tsplit']:
    dataset_names = dataset_names[5:]
elif mode in ['train_CTsplit', 'val_CTsplit']:
    dataset_names = dataset_names[0:3] + dataset_names[5:7]

random.seed(seed)
np.random.seed(seed)
count = 0
signals = np.zeros((n_data, 1, dim))
for dataset_name in dataset_names:
    with open(os.path.join('data', dataset_name), 'rb') as f:
        dataset = pickle.load(f)
    signals_loaded, _ = dataset['signals'], dataset['measurements']
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=num_omp_coefs)
    omp.fit(clean_dict, signals_loaded[:,0,:].T)
    preds = clean_dict.dot(omp.coef_.T).T[:,np.newaxis,:]
    for iter_idx in tqdm(range(inner_loop_total)): # Process each file 5000 times
        # Choose a certain column of it
        col_idx = random.randint(0,preds.shape[0]-1)
        # Normalize by a random multiplier of the l2-norm
        multiplier = np.linalg.norm(signals_loaded[col_idx,0], 2, axis=0)
        # random_multiplier = random.random()*0.5 + 0.25 # Changed this
        random_multiplier = random.random() + 0.25
        temp = preds[col_idx,0] * 1. / (random_multiplier * multiplier)
        # cyclically move it around by +-150 points
        roll_idx = random.randint(-150, 150)
        temp = np.roll(temp, roll_idx)
        if random.random()>0.5: # Possibly invert it
            temp = -temp
        signals[count,0,:] = temp
        count+=1
measurements = generate_freq_measurements(signals, missing_rate)        

save_dict = {}
save_dict['signals'], save_dict['measurements'] = signals, measurements
save_dict['missing_rate'] = missing_rate
pdb.set_trace()
with open(os.path.join('data', f'{mode}_set_arun_testdistributed.pkl'), 'wb') as f:
    pickle.dump(save_dict,f)
