import os, pickle, random, pdb
import numpy as np
import scipy.io as sio
import torch
import sklearn.preprocessing

class SARDataGenerator(object):

    """
    DataGenerator class to sample SAR measurements from a given template
    Must initialize with `template`
    `template` - SAR pulse that is sent out from an aperture. Used to construct
                 the dictionary from which signals are generated (dx1 numpy array)
    Keyword Inputs:
        `dim` - dimension of signals (int)
        `max_sparse` - maximum sparsity of the inputs with respect to the template
                       dictionary (int)
        `sparsity_pattern` - pattern of support of sparse codes, 'uniform' vs 'clustered' vs 'block'
        `support_dist` - distribution from which to draw non-zero entries
        `scale` - scale parameter for distribution of non-zero entries
    """

    def __init__(self, template, dim=1000,
                 max_sparse=200,
                 sparsity_pattern='uniform',
                 support_dist='laplace',
                 scale=0.1,
                 num_clusters=3,
                 normalize_signals='none'):
        self._dim = dim
        self._max_sparse = min(max_sparse, dim)
        self._sparsity_pattern = sparsity_pattern
        self._support_dist = support_dist
        self._scale = scale
        self._num_clusters = num_clusters
        self._normalize_signals = normalize_signals
        if self._dim < template.shape[0]:
            print('Warning: desired signal dimension (%d) is smaller than template (%d)'%(self._dim, template.shape[0]))
            self._template = template[:self._dim]
        else:
            self._template = np.concatenate([template, np.zeros((self._dim-template.shape[0],1))])
        self._dictionary = np.concatenate([np.concatenate([self._template[self._dim-i:], self._template[:self._dim-i]]) for i in range(self._dim)], axis=1)

    def sample_support(self):
        if self._sparsity_pattern == 'uniform':
            sparsity = np.random.randint(1, self._max_sparse)
            p = np.random.permutation(self._dim)
            return p[:sparsity]
        elif self._sparsity_pattern == 'clustered':
            start_idx = np.random.randint(0, self._dim - self._max_sparse)
            sparsity = np.random.randint(1, self._max_sparse)
            return range(start_idx, start_idx + sparsity)
        elif self._sparsity_pattern == 'block':
            if self._num_clusters * self._max_sparse > self._dim:
                self._max_sparse = int(self.dim/(10*self._num_clusters)) * 10
                print('Warning: desired length of signals exceeds \
                       max spacing between clusters, changed sparsity \
                       to %d'%(self._max_sparse,))
            # spacing = ((self._dim - self._max_sparse) / (self._num_clusters * 100)) * 100
            spacing = np.ceil(((self._dim - self._max_sparse) / (self._num_clusters * 100)) * 100).astype('int16') # Arun: Needed to convert it to int otherwise it throws an error when defining starts
            starts = [spacing*i+x for i,x in enumerate(sorted(random.sample(range(spacing), self._num_clusters)))]
            supports = [range(s, s+self._max_sparse) for s in starts]
            return supports

    def sample_sparse_code(self):
        supp = self.sample_support()
        alpha = np.zeros((self._dim, 1))
        if self._support_dist == 'laplace':
            alpha_s = np.random.laplace(scale=self._scale, size=(len(supp)))
            alpha[supp,0] = alpha_s
        if self._support_dist == 'normal':
            alpha_s = np.random.normal(scale=self._scale, size=(len(supp)))
            alpha[supp,0] = alpha_s
        if self._support_dist == 'block-laplace':
            for s in supp:
                alpha_s = np.random.laplace(scale=self._scale, size=(len(s)))
                alpha[s,0] = alpha_s
        if self._support_dist == 'block-normal':
            for s in supp:
                alpha_s = np.random.laplace(scale=self._scale, size=(len(s)))
                alpha[s,0] = alpha_s
        if self._support_dist == 'block-hamming':
            for s in supp:
                alpha_s = np.random.laplace(scale=1.0) * np.hamming(len(s))
                alpha[s,0] = alpha_s
        if self._support_dist == 'block-exp':
            for s in supp:
                alpha_s = np.random.laplace(scale=1.0) \
                          * np.array([np.exp(-1.*self._scale * i) for i in range(len(s))])
                alpha[s,0] = alpha_s
        if self._support_dist == 'block-sqrexp':
            for s in supp:
                alpha_s = np.random.laplace(scale=1.0) \
                          * np.array([np.exp(-1.*self._scale * (i**2)) for i in range(len(s))])
                alpha[s,0] = alpha_s
        if self._support_dist == 'block-rootexp':
            for s in supp:
                alpha_s = np.random.laplace(scale=1.0) \
                          * np.array([np.exp(-1.*self._scale * (i**0.5)) for i in range(len(s))])
                alpha[s,0] = alpha_s
        if self._support_dist == 'block-rootdec':
            for s in supp:
                alpha_s = np.random.laplace(scale=1.0) \
                          * np.array([ max(1 - 0.1*(i**0.5), 0) for i in range(len(s))])
                alpha[s,0] = alpha_s
        return alpha

    def generate_signals(self, n_signals):
        """
        Inputs:
                `n_signals` - number of signals to generate from dictionary (int)
        Outputs:
                array containing all signals (n_signals x 1 x dim numpy array)
        """
        sparse_codes = np.concatenate([self.sample_sparse_code() for i in range(n_signals)], axis=1)
        signals = self._dictionary.dot(sparse_codes)
        if self._normalize_signals == 'max':
            m = np.max(np.abs(signals), axis=0)
            signals = signals * (1./m)
        elif self._normalize_signals == 'l2':
            m = np.linalg.norm(signals, 2, axis=0)
            signals = signals * (1./m)
        return signals.T[:,np.newaxis,:]

    def generate_batch(self, n_signals, missing_rate=0.5):
        signals = self.generate_signals(n_signals)
        measurements = generate_freq_measurements(signals, missing_rate)
        return signals, measurements

    def generate_and_save_dataset(self, dataset_size, save_dir='data',
                                  save_name='train_set.pkl', missing_rate=0.5):
        save_dict = {}
        save_dict['signals'], save_dict['measurements'] = self.generate_batch(dataset_size, missing_rate=missing_rate)
        save_dict['max_sparse'] = self._max_sparse
        save_dict['missing_rate'] = missing_rate
        save_dict['sparsity_pattern'] = self._sparsity_pattern

        with open(os.path.join(save_dir, save_name), 'wb') as f:
            pickle.dump(save_dict,f)

        return save_dict['signals'], save_dict['measurements']

def generate_freq_measurements(signals, missing_rate, energy_band=(380e6, 2080e6), sampling_period=2.668e-11):
    """
    Inputs:
        `signals` - Ground Truth SAR signals measured at an aperture (num_signals x 1 x d numpy array)
        `missing_rate` - fraction of spectrum that is missing (float)
    Keyword Inputs:
        `energy_band` -  2-tuple containing start and end frequencies for the spectrum of the signals
                         generated from the template (tuple of floats)
        `sampling_period` - sampling period in seconds (float)
    Outputs:
        array containing all corrupted signals (num_signals x 1 x d numpy array)
    """
    num_signals,_,dim = signals.shape
    sampling_freq = 1. / (sampling_period + 1e-32)
    df = sampling_freq / dim

    bandwidth = energy_band[1] - energy_band[0]
    missing_bandwidth = round(bandwidth * missing_rate)
    f_start = energy_band[0] + round(0.1*bandwidth)
    f_end = f_start + missing_bandwidth
    f_start_idx = np.int_(np.ceil(f_start / df))
    f_end_idx = np.int_(np.ceil(f_end / df))

    measurements = []
    for i in range(num_signals):
        spectrum = np.fft.fft(signals[i,0,:])
        corrupted_spectrum = np.copy(spectrum)
        corrupted_spectrum[f_start_idx:f_end_idx] = 0.0+0.0j
        corrupted_spectrum[dim // 2 + 1: dim - 1] = np.conj(corrupted_spectrum[dim // 2 - 1: 1: -1])
        measurements.append(np.fft.ifft(corrupted_spectrum).real[:,np.newaxis])

    return np.concatenate(measurements, axis=1).T[:,np.newaxis,:]

def generate_freq_measurements_2D(signals, missing_rate, energy_band=(380e6, 2080e6), sampling_period=2.668e-11):
    """
    Inputs:
        `signals` - Ground Truth SAR signals measured at an aperture (num_signals x 1 x slowTimeDim x d numpy array)
        `missing_rate` - fraction of spectrum that is missing (float)
    Keyword Inputs:
        `energy_band` -  2-tuple containing start and end frequencies for the spectrum of the signals
                         generated from the template (tuple of floats)
        `sampling_period` - sampling period in seconds (float)
    Outputs:
        array containing all corrupted signals (num_signals x 1 x slowTimeDim x d numpy array)
    """
    num_signals, _, slowTimeDim, dim = signals.shape
    sampling_freq = 1. / (sampling_period + 1e-32)
    df = sampling_freq / dim

    bandwidth = energy_band[1] - energy_band[0]
    missing_bandwidth = round(bandwidth * missing_rate)
    f_start = energy_band[0] + round(0.1*bandwidth)
    f_end = f_start + missing_bandwidth
    f_start_idx = np.int_(np.ceil(f_start / df))
    f_end_idx = np.int_(np.ceil(f_end / df))

    measurements = []
    for i in range(num_signals):
        temp_store = np.zeros((slowTimeDim, dim))
        for j in range(slowTimeDim):
            spectrum = np.fft.fft(signals[i,0,j,:])
            corrupted_spectrum = np.copy(spectrum)
            corrupted_spectrum[f_start_idx:f_end_idx] = 0.0+0.0j
            corrupted_spectrum[dim // 2 + 1: dim - 1] = np.conj(corrupted_spectrum[dim // 2 - 1: 1: -1])
            temp_store[j] = np.fft.ifft(corrupted_spectrum).real
        measurements.append(temp_store[np.newaxis, :, :].copy())

    return np.concatenate(measurements, axis=0)[:,np.newaxis,:,:]

def generate_freq_measurements_modified(signals, missing_rate, energy_band=(0,80)):
    """
    Inputs:
        `signals` - Ground Truth SAR signals measured at an aperture (num_signals x 1 x d numpy array)
        `missing_rate` - fraction of spectrum that is missing (float)
    Keyword Inputs:
        `energy_band` -  2-tuple of ints containing start and end indices for the spectrum of the signals
                         generated from the dictionary
    Outputs:
        array containing all corrupted signals (num_signals x 1 x d numpy array)
    """
    num_signals,_,dim = signals.shape
    start, end = energy_band
    bandwidth = end - start
    missing_bandwidth = np.int(missing_rate * bandwidth)

    f_start = end // 2 - missing_bandwidth // 2
    f_end = end // 2 + missing_bandwidth // 2

    measurements = []
    for i in range(num_signals):
        spectrum = np.fft.fft(signals[i,0,:])
        corrupted_spectrum = np.copy(spectrum)
        corrupted_spectrum[f_start:f_end] = 0.0+0.0j
        corrupted_spectrum[dim // 2 + 1: dim - 1] = np.conj(corrupted_spectrum[dim // 2 - 1: 1: -1])
        measurements.append(np.fft.ifft(corrupted_spectrum).real[:,np.newaxis])

    return np.concatenate(measurements, axis=1).T[:,np.newaxis,:]

def create_dataset_akshay(params, dataset_size=50000, dataset_name='train_set_akshay.pkl'):
    template_path = os.path.join(os.getcwd(), 'data/SimTxPulse.mat')
    # trainset_size = 50000 # Renamed to dataset_size
    dim=1000
    max_sparse=50
    sparsity_pattern='block'
    support_dist='block-rootdec'
    scale=0.1
    num_clusters=3
    normalize_signals='l2'
    missing_rate = 0.50

    online_or_fixed = 'fixed'                               # Online doesn't work (and I'm not sure I care enough to get it working...)

    # if hasattr(conf, 'trainset_path') is False:
    if online_or_fixed == 'fixed':
        if not os.path.isfile(os.path.join('data', dataset_name)): # The filename doesn't already exist
            template = sio.loadmat(template_path)['st']
            datagen = SARDataGenerator(template,
                                    dim=dim,
                                    max_sparse=max_sparse,
                                    sparsity_pattern=sparsity_pattern,
                                    support_dist=support_dist,
                                    scale=scale,
                                    num_clusters=num_clusters,
                                    normalize_signals=normalize_signals)
            signals, measurements = datagen.generate_and_save_dataset(dataset_size, missing_rate=missing_rate, save_name=dataset_name)
            signals, measurements = torch.Tensor(signals), torch.Tensor(measurements)
        else:
            with open(os.path.join('data', dataset_name), 'rb') as f:
                dataset = pickle.load(f)
                signals, measurements = torch.Tensor(dataset['signals']), torch.Tensor(dataset['measurements'])
        gen_dataset = torch.utils.data.TensorDataset(signals, measurements)
        print('Loaded Training Dataset')   
    # TODO: Online data generation doesn't currently work...
    # elif online_or_fixed == 'online':
    #     template = sio.loadmat(conf.template_path)['st']
    #     datagen = SARDataGenerator(template,
    #                             dim=dim,
    #                             max_sparse=max_sparse,
    #                             sparsity_pattern=sparsity_pattern,
    #                             support_dist=support_dist,
    #                             scale=scale,
    #                             num_clusters=num_clusters,
    #                             normalize_signals=normalize_signals)
    #     print('Created Training Simulator')
    
    return gen_dataset

def create_dataset_arun(params, dataset_size=50000, dataset_name='train_set_arun.pkl', invert_waveforms=False, line_length = 1000):
    if '_set_arun.pkl' in dataset_name:
        num_files = 500
        dataset_dir = '20200903_data'
    elif '_set_arun_2.pkl' in dataset_name or '_set_arun_3.pkl' in dataset_name:
        num_files = 5000
        dataset_dir = '20200923_data'

    if not os.path.isfile(os.path.join('data', dataset_name)): # The filename doesn't already exist
        ROOT_DIR = os.getcwd()
        save_dir = os.path.join(ROOT_DIR, 'data')

        save_dict = {}

        save_dict['max_sparse'] = 10 # TODO: Set in the simulation
        missing_rate = 0.50
        save_dict['missing_rate'] = missing_rate
        save_dict['sparsity_pattern'] = 'uniform' # ish

        # CASE 0 - Generate (50000, 1, 1000) data to use as a replacement for Akshay's training data
        # save_name = 'sar_dataset_50_max_sparse_10_uniform.pkl'
        signals = np.zeros((dataset_size, 1, line_length))

        for idx in range(dataset_size):
            print(idx)
            # Choose file name to read after seeding the random number generator
            if 'train' in dataset_name:
                random.seed(idx)
            elif 'val' in dataset_name:
                random.seed(idx+13371337)
            elif 'test' in dataset_name:
                random.seed(idx+13371337*2)                
            file_idx = random.randint(1,num_files) # Inclusive of both end points
            file_name = f'{file_idx}.mat'
            data = sio.loadmat(os.path.join('/data/arun/projects/sarnet/data', dataset_dir, file_name))['data'][300:300+line_length,:].astype(np.float32) # 300:1300 is relevant based on the simulation parameters I set for 1000 length signal

            # Choose a certain column of it
            col_idx = random.randint(0,data.shape[1]-1)
            if invert_waveforms:
                if random.random()>0.5:
                    signals[idx,0] = -data[:,col_idx]
                else:
                    signals[idx,0] = -data[:,col_idx]
            else:
                signals[idx,0] = data[:,col_idx]

            # Normalize by a random multiplier of the l2-norm
            multiplier = np.linalg.norm(signals[idx,0], 2, axis=0)
            # random_multiplier = random.random()*0.5 + 0.25 # Changed this
            random_multiplier = random.random() + 0.25
            signals[idx,0] = signals[idx,0] * 1. / (random_multiplier * multiplier)

        measurements = generate_freq_measurements(signals, missing_rate)

        # save_dict['signals'], save_dict['measurements'] = self.generate_batch(dataset_size, missing_rate=missing_rate)
        save_dict['signals'], save_dict['measurements'] = signals, measurements

        with open(os.path.join(save_dir, dataset_name), 'wb') as f:
            pickle.dump(save_dict,f)

        signals, measurements = torch.Tensor(signals), torch.Tensor(measurements)
    else:
        with open(os.path.join('data', dataset_name), 'rb') as f:
            dataset = pickle.load(f)

            signals, measurements = torch.Tensor(dataset['signals']), torch.Tensor(dataset['measurements'])

    print(f"Dimensions of signals tensor is {signals.shape}")
    print(f"Dimensions of measurements tensor is {measurements.shape}")        
    gen_dataset = torch.utils.data.TensorDataset(signals, measurements)
    print('Loaded Training Dataset')

    return gen_dataset

def create_dataset_arun_2D(params, dataset_size=50000, dataset_name='train_set_arun_2D.pkl', invert_waveforms=False):
    num_files = 5000
    dataset_dir = '20200923_data'

    if not os.path.isfile(os.path.join('data', dataset_name)): # The filename doesn't already exist
        ROOT_DIR = os.getcwd()
        save_dir = os.path.join(ROOT_DIR, 'data')

        save_dict = {}

        save_dict['max_sparse'] = 10 # TODO: Set in the simulation
        missing_rate = 0.50
        save_dict['missing_rate'] = missing_rate
        save_dict['sparsity_pattern'] = 'uniform' # ish

        # CASE 0 - Generate (50000, 1, 1000) data to use as a replacement for Akshay's training data
        # save_name = 'sar_dataset_50_max_sparse_10_uniform.pkl'
        slow_time_dim = 3
        line_length = 1024
        signals = np.zeros((dataset_size, 1, slow_time_dim, line_length))

        for idx in range(dataset_size):
            print(idx)
            # Choose file name to read after seeding the random number generator
            if 'train' in dataset_name:
                random.seed(idx)
            elif 'val' in dataset_name:
                random.seed(idx+13371337)
            elif 'test' in dataset_name:
                random.seed(idx+13371337*2)                
            file_idx = random.randint(1,num_files) # Inclusive of both end points
            file_name = f'{file_idx}.mat'
            data = sio.loadmat(os.path.join('/data/arun/projects/sarnet/data', dataset_dir, file_name))['data'][300:300+line_length,:].astype(np.float32) # 300:1300 is relevant based on the simulation parameters I set for 1000 length signal

            # Choose column spacing
            decision_var = random.random()
            # Choose a certain subset of columns
            if line_length<=0.5: # with probability 0.5 - spacing of 1 element between lines
                col_idx = random.randint(0,data.shape[1]-3)
                signals[idx,0] = data[:,col_idx:col_idx+3].T
            elif line_length<=0.8: # with probability 0.3 - spacing of 2 elements between lines
                col_idx = random.randint(0,data.shape[1]-5)
                signals[idx,0] = data[:,[col_idx,col_idx+2,col_idx+4]].T
            else: # with probability 0.2 - spacing of 4 elements between lines
                col_idx = random.randint(0,data.shape[1]-9)
                signals[idx,0] = data[:,[col_idx,col_idx+4,col_idx+8]].T            
            if invert_waveforms:
                if random.random()>0.5:
                    signals[idx,0] = -signals[idx,0]
                else:
                    signals[idx,0] = signals[idx,0]
            else:
                signals[idx,0] = signals[idx,0]

            # Normalize by a random multiplier of the l2-norm
            multiplier = np.mean(np.linalg.norm(signals[idx,0], 2, axis=1)) # Mean l2 norm of the 3 columns, each of which should be really close to the other...
            # random_multiplier = random.random()*0.5 + 0.25 # Changed this
            random_multiplier = random.random() + 0.25
            signals[idx,0] = signals[idx,0] * 1. / (random_multiplier * multiplier)

        measurements = generate_freq_measurements_2D(signals, missing_rate)

        # save_dict['signals'], save_dict['measurements'] = self.generate_batch(dataset_size, missing_rate=missing_rate)
        save_dict['signals'], save_dict['measurements'] = signals, measurements

        with open(os.path.join(save_dir, dataset_name), 'wb') as f:
            pickle.dump(save_dict,f)

        signals, measurements = torch.Tensor(signals), torch.Tensor(measurements)
    else:
        with open(os.path.join('data', dataset_name), 'rb') as f:
            dataset = pickle.load(f)

            signals, measurements = torch.Tensor(dataset['signals']), torch.Tensor(dataset['measurements'])

    print(f"Dimensions of signals tensor is {signals.shape}")
    print(f"Dimensions of measurements tensor is {measurements.shape}")        
    gen_dataset = torch.utils.data.TensorDataset(signals, measurements)
    print('Loaded Training Dataset')

    return gen_dataset

def create_dataset_real(params, dataset_name='test_set_real.pkl', line_length = 1000):

    # pdb.set_trace()
    if '_set_real.pkl' in dataset_name or '_set_real_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C1.mat', '/data/arun/projects/sarnet/real_sar_data/C2.mat',
        '/data/arun/projects/sarnet/real_sar_data/C3.mat', '/data/arun/projects/sarnet/real_sar_data/C4.mat',
        '/data/arun/projects/sarnet/real_sar_data/C5.mat', '/data/arun/projects/sarnet/real_sar_data/G1.mat',
        '/data/arun/projects/sarnet/real_sar_data/G2.mat', '/data/arun/projects/sarnet/real_sar_data/T1.mat',
        '/data/arun/projects/sarnet/real_sar_data/T2.mat', '/data/arun/projects/sarnet/real_sar_data/T3.mat',
        '/data/arun/projects/sarnet/real_sar_data/T4.mat', '/data/arun/projects/sarnet/real_sar_data/T5.mat']
    elif '_set_real_onlyfirsttwoseqs.pkl' in dataset_name or '_set_real_onlyfirsttwoseqs_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C1.mat', '/data/arun/projects/sarnet/real_sar_data/C2.mat']        
    elif '_set_real_C1.pkl' in dataset_name or '_set_real_C1_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C1.mat']
    elif '_set_real_C2.pkl' in dataset_name or '_set_real_C2_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C2.mat']
    elif '_set_real_C3.pkl' in dataset_name or '_set_real_C3_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C3.mat']
    elif '_set_real_C4.pkl' in dataset_name or '_set_real_C4_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C4.mat']
    elif '_set_real_C5.pkl' in dataset_name or '_set_real_C5_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C5.mat']
    elif '_set_real_T1.pkl' in dataset_name or '_set_real_T1_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/T1.mat']
    elif '_set_real_T2.pkl' in dataset_name or '_set_real_T2_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/T2.mat']
    elif '_set_real_T3.pkl' in dataset_name or '_set_real_T3_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/T3.mat']
    elif '_set_real_T4.pkl' in dataset_name or '_set_real_T4_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/T4.mat']
    elif '_set_real_T5.pkl' in dataset_name or '_set_real_T5_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/T5.mat']
    elif '_set_real_G1.pkl' in dataset_name or '_set_real_G1_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/G1.mat']
    elif '_set_real_G2.pkl' in dataset_name or '_set_real_G2_1024.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/G2.mat']
    if not os.path.isfile(os.path.join('data', dataset_name)): # The filename doesn't already exist
        ROOT_DIR = os.getcwd()
        save_dir = os.path.join(ROOT_DIR, 'data')

        save_dict = {}
        
        dim = line_length
        missing_rate = 0.50
        normalize_signals='l2'
        save_dict['missing_rate'] = missing_rate
        save_dict['normalize_signals'] = normalize_signals
        save_dict['dim'] = dim
        EPS = 1e-32
        
        temp_list = []
        for real_file_name in real_file_names:
            real_signals = sio.loadmat(real_file_name)['Data']
            
            if normalize_signals == 'max':
                multiplier = np.max(np.abs(real_signals), axis=0) + EPS
                real_signals = real_signals * (1./multiplier)
            elif normalize_signals == 'l2':
                multiplier = np.linalg.norm(real_signals, 2, axis=0) + EPS
                real_signals = real_signals * (1./multiplier)
            real_signals = real_signals.T[:,np.newaxis,:]

            if real_signals.shape[-1] != dim:
                padding = dim - real_signals.shape[-1]
                n_signals = real_signals.shape[0]
                # real_signals = np.concatenate([real_signals, np.zeros((n_signals,1,padding))],axis=-1)
                real_signals = np.concatenate([np.zeros((n_signals,1,padding)), real_signals],axis=-1)
            temp_list.append(real_signals)
        
        signals = np.concatenate(temp_list)
        measurements = generate_freq_measurements(signals, missing_rate)

        # save_dict['signals'], save_dict['measurements'] = self.generate_batch(dataset_size, missing_rate=missing_rate)
        save_dict['signals'], save_dict['measurements'] = signals, measurements

        with open(os.path.join(save_dir, dataset_name), 'wb') as f:
            pickle.dump(save_dict,f)

        signals, measurements = torch.Tensor(signals), torch.Tensor(measurements)
    else:
        with open(os.path.join('data', dataset_name), 'rb') as f:
            dataset = pickle.load(f)

            signals, measurements = torch.Tensor(dataset['signals']), torch.Tensor(dataset['measurements'])

    print(f"Dimensions of signals tensor is {signals.shape}")
    print(f"Dimensions of measurements tensor is {measurements.shape}")        
    gen_dataset = torch.utils.data.TensorDataset(signals, measurements)
    
    print('Loaded Real Dataset')           

    return gen_dataset

def create_dataset_real_2D(params, dataset_name='test_set_real_2D.pkl'):

    if '_set_real_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C1.mat', '/data/arun/projects/sarnet/real_sar_data/C2.mat',
        '/data/arun/projects/sarnet/real_sar_data/C3.mat', '/data/arun/projects/sarnet/real_sar_data/C4.mat',
        '/data/arun/projects/sarnet/real_sar_data/C5.mat', '/data/arun/projects/sarnet/real_sar_data/G1.mat',
        '/data/arun/projects/sarnet/real_sar_data/G2.mat', '/data/arun/projects/sarnet/real_sar_data/T1.mat',
        '/data/arun/projects/sarnet/real_sar_data/T2.mat', '/data/arun/projects/sarnet/real_sar_data/T3.mat',
        '/data/arun/projects/sarnet/real_sar_data/T4.mat', '/data/arun/projects/sarnet/real_sar_data/T5.mat']
    elif '_set_real_onlyfirsttwoseqs_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C1.mat', '/data/arun/projects/sarnet/real_sar_data/C2.mat']        
    elif '_set_real_C1_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C1.mat']
    elif '_set_real_C2_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C2.mat']
    elif '_set_real_C3_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C3.mat']
    elif '_set_real_C4_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C4.mat']
    elif '_set_real_C5_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/C5.mat']
    elif '_set_real_T1_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/T1.mat']
    elif '_set_real_T2_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/T2.mat']
    elif '_set_real_T3_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/T3.mat']
    elif '_set_real_T4_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/T4.mat']
    elif '_set_real_T5_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/T5.mat']
    elif '_set_real_G1_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/G1.mat']
    elif '_set_real_G2_2D.pkl' in dataset_name:
        real_file_names = ['/data/arun/projects/sarnet/real_sar_data/G2.mat']
    if not os.path.isfile(os.path.join('data', dataset_name)): # The filename doesn't already exist
        ROOT_DIR = os.getcwd()
        save_dir = os.path.join(ROOT_DIR, 'data')

        save_dict = {}

        slow_time_dim = 3
        line_length = 1024
        dim = line_length
        missing_rate = 0.50
        normalize_signals='l2'
        save_dict['missing_rate'] = missing_rate
        save_dict['normalize_signals'] = normalize_signals
        save_dict['dim'] = dim
        EPS = 1e-32
        
        temp_list = []
        for real_file_name in real_file_names:
            real_signals = sio.loadmat(real_file_name)['Data']
            
            if normalize_signals == 'max':
                multiplier = np.max(np.abs(real_signals), axis=0) + EPS
                real_signals = real_signals * (1./multiplier)
            elif normalize_signals == 'l2':
                multiplier = np.linalg.norm(real_signals, 2, axis=0) + EPS
                real_signals = real_signals * (1./multiplier)
            real_signals = real_signals.T

            if real_signals.shape[-1] != dim:
                padding = dim - real_signals.shape[-1]
                n_signals = real_signals.shape[0]
                # real_signals = np.concatenate([real_signals, np.zeros((n_signals,1,padding))],axis=-1)
                real_signals = np.concatenate([np.zeros((n_signals,padding)), real_signals],axis=-1)
            for j in range(0, real_signals.shape[0]-slow_time_dim, 1):
                temp_list.append(real_signals[j:j+3,:][np.newaxis, :, :])
        
        signals = np.concatenate(temp_list, axis=0)[:,np.newaxis,:,:]
        measurements = generate_freq_measurements_2D(signals, missing_rate)

        # save_dict['signals'], save_dict['measurements'] = self.generate_batch(dataset_size, missing_rate=missing_rate)
        save_dict['signals'], save_dict['measurements'] = signals, measurements

        with open(os.path.join(save_dir, dataset_name), 'wb') as f:
            pickle.dump(save_dict,f)

        signals, measurements = torch.Tensor(signals), torch.Tensor(measurements)
    else:
        with open(os.path.join('data', dataset_name), 'rb') as f:
            dataset = pickle.load(f)

            signals, measurements = torch.Tensor(dataset['signals']), torch.Tensor(dataset['measurements'])

    print(f"Dimensions of signals tensor is {signals.shape}")
    print(f"Dimensions of measurements tensor is {measurements.shape}")        
    gen_dataset = torch.utils.data.TensorDataset(signals, measurements)
    
    print('Loaded Real Dataset')           

    return gen_dataset
