import numpy as np
import scipy.io as sio
import os, pickle, random, pdb
from tqdm import tqdm

def snr_mixer(target_snr, x, n):
    return x + n/np.linalg.norm(n) * np.linalg.norm(x)  / np.power(10, target_snr/20.0)
    

def add_interference(signals, is_train, chosen_snr=False):
    interference = sio.loadmat('data/mat_data/interference/rfi_Arun.mat')['rfi_resampled'] # Almost 180000000 in length
    measurements = np.zeros_like(signals)

    if is_train:
        snr_choices = np.array([-15, -10, -5, 0, 5, 10])
        interference_start_idx, interference_end_idx = 1000, 180000000//2-signals.shape[-1] # Keeping first half of it for training
    else:
        # snr_choices = np.array([-12.5, -7.5, -2.5, 2.5, 7.5]) # Think about not matching train and test SNR exactly - issue is it raises questions about  other experiments
        snr_choices = np.array([-15, -10, -5, 0, 5, 10])
        interference_start_idx, interference_end_idx = 180000000//2, 180000000 # Keeping second half of it for testing

    if chosen_snr is not False:
        snr_choices = np.array([chosen_snr])

    for j in tqdm(range(signals.shape[0])):
        target_snr = random.choice(snr_choices)
        interference_start = random.randint(interference_start_idx, interference_end_idx)
        measurements[j,:,:] = snr_mixer(target_snr, signals[j], interference[interference_start:interference_start+signals.shape[-1], 0])
    
    return measurements


if __name__ == '__main__':
    # ## CODE BLOCK 1 - For non-sepcific SNR train and test data
    # # # Old-----------------------------------------------------------------------
    # # # dataset_name, is_train = 'train_CTsplit_set_arun_testdistributed.pkl', True
    # # # dataset_name, is_train = 'val_CTsplit_set_arun_testdistributed.pkl', True
    # # # dataset_name, is_train = 'test_CTsplit_set_arun_testdistributed.pkl', True
    # # # dataset_name, is_train = 'train_set_arun.pkl', True
    # # # dataset_name, is_train = 'test_set_arun.pkl', False
    # # # Relevant------------------------------------------------------------------
    # # dataset_names, is_train = ['train_set_arun_extended.pkl', 'train_set_arun_generative_modeled_extended.pkl'], True 
    # # dataset_names, is_train = ['val_set_arun.pkl', 'val_set_arun_generative_modeled.pkl'], False
    # # dataset_names, is_train = ['test_set_real_onlyfirsttwoseqs.pkl'], False
    
    # for dataset_name in dataset_names:   
    #     with open(os.path.join('data', dataset_name), 'rb') as f:
    #         dataset = pickle.load(f)                    
    #     signals = dataset['signals']    
    #     measurements = add_interference(signals, is_train=is_train)
    #     save_dict = {}
    #     save_dict['signals'], save_dict['measurements'] = signals, measurements 
    #     # pdb.set_trace()
    #     with open(os.path.join('data', "_".join(dataset_name.split('_')[0:1]+['interference']+dataset_name.split('_')[1:])), 'wb') as f:
    #         pickle.dump(save_dict,f)

    # CODE BLOCK 2 For SNR Specific test data
    snr_list = [-15, -10, -5, 0, 5, 10]
    for curr_snr in snr_list:
        # dataset_name, is_train = 'val_set_arun.pkl', False
        dataset_name, is_train = 'val_set_arun_generative_modeled.pkl', False
        # dataset_name, is_train = 'test_set_real_onlyfirsttwoseqs.pkl', False
        with open(os.path.join('data', dataset_name), 'rb') as f:
            dataset = pickle.load(f)
        signals = dataset['signals']
        measurements = add_interference(signals, is_train=is_train, chosen_snr = curr_snr)            
        save_dict = {}
        save_dict['signals'], save_dict['measurements'] = signals, measurements        
        output_filename = "_".join(dataset_name.split('_')[0:1]+['interference']+dataset_name.split('_')[1:])
        output_filename = output_filename[:-4]+'_'+str(curr_snr)+output_filename[-4:]
        with open(os.path.join('data', output_filename), 'wb') as f:
            pickle.dump(save_dict,f)        
