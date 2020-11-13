import os, pickle, pdb
import numpy as np
import scipy.io as sio
from sklearn.linear_model import OrthogonalMatchingPursuit
from tqdm import tqdm

## 0 - Common Code - SNR calculation
def snr_akshay(truth, x_in, x_out):
    # Truth (=signals tensor) is of size 1800x1x1024 (or so)
    # x_in (=measurements tensor) is of size 1800x1x1024 (or so)
    # x_out (=preds tensor) is of size 1800x1x1024 (or so)
    EPS = 1e-32
    n_signals = truth.shape[0]
    sig_norms = np.array([np.linalg.norm(truth[i].ravel()) + EPS for i in range(n_signals)]) # np.ravel() Return a contiguous flattened array.
    errs_in = np.array([np.linalg.norm((truth[i] - x_in[i]).ravel()) + EPS for i in range(n_signals)])
    errs_out = np.array([np.linalg.norm((truth[i] - x_out[i]).ravel()) + EPS for i in range(n_signals)])

    snr_in = 20.*(np.log10(sig_norms) - np.log10(errs_in))
    snr_out = 20.*(np.log10(sig_norms) - np.log10(errs_out))
    avg_snr_in = np.mean(snr_in)
    avg_snr_out = np.mean(snr_out)
    avg_snr_gain = avg_snr_out - avg_snr_in
    return avg_snr_in, avg_snr_out, avg_snr_gain


## 1 - Baseline for Interference suppression
def kill_freqs(measurements, kill_idx_start, kill_idx_end, sampling_period=2.668e-11):
    num_signals,_,dim = measurements.shape
    sampling_freq = 1. / (sampling_period + 1e-32)
    df = sampling_freq / dim

    f_start_idx = kill_idx_start
    f_end_idx = kill_idx_end+1

    signals = []
    for i in range(num_signals):
        spectrum = np.fft.fft(measurements[i,0,:])
        cleaned_spectrum = np.copy(spectrum)
        cleaned_spectrum[f_start_idx:f_end_idx] = 0.0+0.0j
        cleaned_spectrum[dim // 2 + 1: dim - 1] = np.conj(cleaned_spectrum[dim // 2 - 1: 1: -1])
        signals.append(np.fft.ifft(cleaned_spectrum).real[:,np.newaxis])

    return np.concatenate(signals, axis=1).T[:,np.newaxis,:]

def calculate_interference_baseline():
    snr_list = [-15, -10, -5, 0, 5, 10]
    results_dir_names_template = '20201103_arun_extended_and_generative_testononlyfirsttwoseqs_interference_'

    for split in ['val', 'test']:
    # for split in ['test']:
        for snr in snr_list:
            results_dir = os.path.join('results', results_dir_names_template+str(snr), split)
            with open(os.path.join(results_dir, 'results.pkl'), 'rb') as f:
                dataset = pickle.load(f)
            signals, measurements, outputs = dataset['signals'], dataset['measurements'], dataset['outputs']
            kill_idx_start, kill_idx_end = 7, 13 # Both inclusive
            signals_enhanced = kill_freqs(measurements, kill_idx_start, kill_idx_end)
            SNR_meas, SNR_pred, SNR_gain = snr_akshay(signals, measurements, signals_enhanced)
            print(f"Split: {split} L2 SNR_in: %.6f, SNR_out: %.6f, SNR_gain: %.6f" %(SNR_meas, SNR_pred, SNR_gain))

            dataset['baseline_outputs'] = signals_enhanced

            with open(os.path.join('scratch', 'with_baselines', f'{results_dir_names_template+str(snr)}_results.pkl'), 'wb') as f:
                pickle.dump(dataset, f)

            pdb.set_trace()
    return
    # # plotting study
    # import matplotlib.pyplot as plt
    # chosen_signal = np.squeeze(signals[1337])
    # chosen_measurment = np.squeeze(measurements[1337])
    # plt.subplot(611)
    # plt.plot(chosen_signal)
    # plt.subplot(612)
    # chosen_signal_spectrum = np.fft.fft(chosen_signal)
    # plt.plot(np.abs(chosen_signal_spectrum))
    # plt.subplot(613)
    # plt.plot(chosen_measurment)
    # plt.subplot(614)
    # chosen_measurement_spectrum = np.fft.fft(chosen_measurment)
    # plt.plot(np.abs(chosen_measurement_spectrum))
    # plt.subplot(615)
    # plt.plot(chosen_measurment-chosen_signal)
    # plt.subplot(616)
    # chosen_difference_spectrum = np.fft.fft(chosen_measurment-chosen_signal)
    # # plt.plot(np.abs(chosen_difference_spectrum)) #
    # plt.plot(np.abs(chosen_difference_spectrum)[0:50]) # Majority of interference at indices 7-13 - just set them to zero (256:476 MHz for ts = 2.668e-11, or 34-63MHz for ts=2e-10)
    # plt.savefig('temp.png')
    # plt.close()

## 2 - Baseline for filling in random gaps 

def generate_matchedmask_freq_measurements(padded_template, curr_signal, curr_measurement, missing_rate, energy_band=(380e6, 2080e6), sampling_period=2.668e-11):
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
    num_signals,_,dim = curr_signal.shape
    sampling_freq = 1. / (sampling_period + 1e-32)
    df = sampling_freq / dim

    bandwidth = energy_band[1] - energy_band[0]
    missing_bandwidth = round(bandwidth * missing_rate)
    f_start = energy_band[0]
    f_end = energy_band[1]
    f_start_idx = np.int_(np.ceil(f_start / df))
    f_end_idx = np.int_(np.ceil(f_end / df))

    measurements = []

    signal_spectrum = np.fft.fft(curr_signal[0,0,:])[:dim//2]
    measurement_spectrum = np.fft.fft(curr_measurement[0,0,:])[:dim//2]
    difference_idxs = np.where(np.abs(np.abs(measurement_spectrum)-np.abs(signal_spectrum))>1e-4)
    # pdb.set_trace()

    pulse_spectrum = np.fft.fft(np.squeeze(padded_template))
    corrupted_pulse_spectrum = np.copy(pulse_spectrum)
    # pdb.set_trace()
    corrupted_pulse_spectrum[difference_idxs] = 0.0+0.0j
    corrupted_pulse_spectrum[dim // 2 + 1: dim - 1] = np.conj(corrupted_pulse_spectrum[dim // 2 - 1: 1: -1])
    corrupted_pulse = np.fft.ifft(corrupted_pulse_spectrum).real[:, np.newaxis]

    return corrupted_pulse

# plt.subplot(411)
# plt.plot(np.abs(spectrum)[f_start_idx:f_end_idx+1])
# plt.subplot(412)
# spectrum_2 = np.fft.fft(measurements[0,0,:])
# temp_2 = np.abs(spectrum_2)[f_start_idx:f_end_idx+1]
# plt.plot(temp_2)
# plt.subplot(413)
# temp_3 = np.abs(np.abs(spectrum)-np.abs(spectrum_2))[f_start_idx:f_end_idx+1]
# plt.plot(temp_3)
# plt.subplot(414)
# plt.plot(temp_3+temp_2)
# plt.savefig('temp.png')
# plt.close()
# len(np.where(temp_3>1e-4)[0])/len(temp_3)*100

def calculate_omp_randomgaps_performance(missing_rate, signals, measurements):
    from data import SARDataGenerator, generate_freq_measurements

    dim = 1024
    template_path = 'data/SimTxPulse.mat'
    template = sio.loadmat(template_path)['st']
    padded_template = np.concatenate([template, np.zeros((dim-template.shape[0],1))], axis=0)
    padded_template = padded_template.T[:,np.newaxis,:]
    max_num_omp = 25

    max_sparse=50                   # None of these 4 matter - just need to give something to the dictionary
    sparsity_pattern='block'    
    support_dist='block-rootdec'
    scale=0.1 

    datagen = SARDataGenerator(template,
                            dim=dim,
                            max_sparse=max_sparse,
                            sparsity_pattern=sparsity_pattern,
                            support_dist=support_dist,
                            scale=scale)

    clean_dict = np.copy(datagen._dictionary)

    output_snr_list = []
    input_snr_list = []
    snr_gain_list = []

    best_output_snr = -20

    decrease_count = 0
    preds = np.zeros_like(measurements)
    for num_omp_coefs in tqdm(range(1,max_num_omp+1)):
        for datapoint_idx in tqdm(range(measurements.shape[0])):
            # print(num_omp_coefs)
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=num_omp_coefs)
            
            curr_measurement = np.squeeze(measurements[datapoint_idx])[np.newaxis, np.newaxis, :]
            curr_signal = np.squeeze(signals[datapoint_idx])[np.newaxis, np.newaxis, :]
            corrupted_atom = np.squeeze(generate_matchedmask_freq_measurements(padded_template, curr_signal, curr_measurement, missing_rate))[:,np.newaxis]
            corrupted_data_gen = SARDataGenerator(corrupted_atom,
                                                dim=dim,
                                                max_sparse=max_sparse,
                                                sparsity_pattern=sparsity_pattern,
                                                support_dist=support_dist,
                                                scale=scale)
            corrupted_dict = np.copy(corrupted_data_gen._dictionary)        
            omp.fit(corrupted_dict, measurements[datapoint_idx,0,:].T)
            preds[datapoint_idx,0] = clean_dict.dot(omp.coef_.T).T
        SNR_meas, SNR_pred, SNR_gain = snr_akshay(signals, measurements, preds)
        # pdb.set_trace()
        if SNR_pred>best_output_snr:
            best_output_snr = SNR_pred
            baseline_outputs = preds

        # print("Sim OMP L2 SNR_in: %.6f, SNR_out: %.6f, SNR_gain: %.6f" %(SNR_meas, SNR_pred, SNR_gain))
        # results_snr['sim'] = [SNR_meas, SNR_pred, SNR_gain]
        # result_list.append([SNR_meas, SNR_pred, SNR_gain])
        # gain_list.append(SNR_gain)
        input_snr_list.append(SNR_meas)
        output_snr_list.append(SNR_pred)
        snr_gain_list.append(SNR_gain)
        if len(output_snr_list)>2 and output_snr_list[-1]<output_snr_list[-2]: # it's started decreasing
            decrease_count +=1
            if decrease_count>0: # Stop on first decrease...
                return np.array(input_snr_list), np.array(output_snr_list), np.array(snr_gain_list), baseline_outputs
    return np.array(input_snr_list), np.array(output_snr_list), np.array(snr_gain_list), baseline_outputs

def calculate_randomgaps_baseline():
    
    missing_rates = [0.5, 0.6, 0.7, 0.8, 0.9]
    results_dir_names_template = '20201103_arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps_'

    for split in ['val', 'test']:
    # for split in ['test']:
        for missing_rate in missing_rates:
            results_dir = os.path.join('results', results_dir_names_template+str(int(missing_rate*100)), split)
            with open(os.path.join(results_dir, 'results.pkl'), 'rb') as f:
                dataset = pickle.load(f)
            signals, measurements, outputs = dataset['signals'], dataset['measurements'], dataset['outputs']

            input_snr_array, output_snr_array, snr_gain_array, baseline_outputs = calculate_omp_randomgaps_performance(missing_rate, signals, measurements)
            print(f"Split: {split} K:%d L2 SNR_in: %.6f, SNR_out: %.6f, SNR_gain: %.6f" %(np.argmax(output_snr_array)+1, np.max(input_snr_array), np.max(output_snr_array), np.max(snr_gain_array)))
            
            dataset['baseline_outputs'] = baseline_outputs

            with open(os.path.join('scratch', 'with_baselines', f'{results_dir_names_template+str(int(missing_rate*100))}_results.pkl'), 'wb') as f:
                pickle.dump(dataset, f)

            pdb.set_trace()

    return
    # # plotting study
    # import matplotlib.pyplot as plt
    # chosen_signal = np.squeeze(signals[1337])
    # chosen_measurment = np.squeeze(measurements[1337])
    # plt.subplot(611)
    # plt.plot(chosen_signal)
    # plt.subplot(612)
    # chosen_signal_spectrum = np.fft.fft(chosen_signal)
    # plt.plot(np.abs(chosen_signal_spectrum))
    # plt.subplot(613)
    # plt.plot(chosen_measurment)
    # plt.subplot(614)
    # chosen_measurement_spectrum = np.fft.fft(chosen_measurment)
    # plt.plot(np.abs(chosen_measurement_spectrum))
    # plt.subplot(615)
    # plt.plot(chosen_measurment-chosen_signal)
    # plt.subplot(616)
    # chosen_difference_spectrum = np.fft.fft(chosen_measurment-chosen_signal)
    # # plt.plot(np.abs(chosen_difference_spectrum)) #
    # plt.plot(np.abs(chosen_difference_spectrum)[0:50]) # Majority of interference at indices 7-13 - just set them to zero (256:476 MHz for ts = 2.668e-11, or 34-63MHz for ts=2e-10)
    # plt.savefig('temp.png')
    # plt.close()


## 3 - Baseline for filling in block gaps 

def calculate_omp_blockgaps_performance(missing_rate, signals, measurements):
    from data import SARDataGenerator, generate_freq_measurements

    dim = 1024
    template_path = 'data/SimTxPulse.mat'
    template = sio.loadmat(template_path)['st']
    padded_template = np.concatenate([template, np.zeros((dim-template.shape[0],1))], axis=0)
    padded_template = padded_template.T[:,np.newaxis,:]
    corrupted_atom = np.squeeze(generate_freq_measurements(padded_template, missing_rate))[:,np.newaxis]
    max_num_omp = 25

    max_sparse=50                   # None of these 4 matter - just need to give something to the dictionary
    sparsity_pattern='block'    
    support_dist='block-rootdec'
    scale=0.1 
            
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

    output_snr_list = []
    input_snr_list = []
    snr_gain_list = []
    
    best_output_snr = -20

    decrease_count = 0
    for num_omp_coefs in tqdm(range(1,max_num_omp+1)):
        # print(num_omp_coefs)
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=num_omp_coefs)
        omp.fit(corrupted_dict, measurements[:,0,:].T)
        preds = clean_dict.dot(omp.coef_.T).T[:,np.newaxis,:]
        SNR_meas, SNR_pred, SNR_gain = snr_akshay(signals, measurements, preds)

        if SNR_pred>best_output_snr:
            best_output_snr = SNR_pred
            baseline_outputs = preds

        # print("Sim OMP L2 SNR_in: %.6f, SNR_out: %.6f, SNR_gain: %.6f" %(SNR_meas, SNR_pred, SNR_gain))
        # results_snr['sim'] = [SNR_meas, SNR_pred, SNR_gain]
        # result_list.append([SNR_meas, SNR_pred, SNR_gain])
        # gain_list.append(SNR_gain)
        input_snr_list.append(SNR_meas)
        output_snr_list.append(SNR_pred)
        snr_gain_list.append(SNR_gain)
        if len(output_snr_list)>2 and output_snr_list[-1]<output_snr_list[-2]: # it's started decreasing
            decrease_count +=1
            if decrease_count>2:
                return np.array(input_snr_list), np.array(output_snr_list), np.array(snr_gain_list), baseline_outputs
    return np.array(input_snr_list), np.array(output_snr_list), np.array(snr_gain_list), baseline_outputs

def calculate_blockgaps_baseline():
    
    missing_rates = [0.5, 0.6, 0.7, 0.8, 0.9]
    results_dir_names_template = '20201103_arun_extended_and_generative_testononlyfirsttwoseqs_blockgaps_'

    for split in ['val', 'test']:
    # for split in ['test']:
        for missing_rate in missing_rates:
            results_dir = os.path.join('results', results_dir_names_template+str(int(missing_rate*100)), split)
            with open(os.path.join(results_dir, 'results.pkl'), 'rb') as f:
                dataset = pickle.load(f)
            signals, measurements, outputs = dataset['signals'], dataset['measurements'], dataset['outputs']

            input_snr_array, output_snr_array, snr_gain_array, baseline_outputs = calculate_omp_blockgaps_performance(missing_rate, signals, measurements)
            print(f"Split: {split} K:%d L2 SNR_in: %.6f, SNR_out: %.6f, SNR_gain: %.6f" %(np.argmax(output_snr_array)+1, np.max(input_snr_array), np.max(output_snr_array), np.max(snr_gain_array)))
            
            dataset['baseline_outputs'] = baseline_outputs

            with open(os.path.join('scratch', 'with_baselines', f'{results_dir_names_template+str(int(missing_rate*100))}_results.pkl'), 'wb') as f:
                pickle.dump(dataset, f)

            pdb.set_trace()

    return
    # # plotting study
    # import matplotlib.pyplot as plt
    # chosen_signal = np.squeeze(signals[1337])
    # chosen_measurment = np.squeeze(measurements[1337])
    # plt.subplot(611)
    # plt.plot(chosen_signal)
    # plt.subplot(612)
    # chosen_signal_spectrum = np.fft.fft(chosen_signal)
    # plt.plot(np.abs(chosen_signal_spectrum))
    # plt.subplot(613)
    # plt.plot(chosen_measurment)
    # plt.subplot(614)
    # chosen_measurement_spectrum = np.fft.fft(chosen_measurment)
    # plt.plot(np.abs(chosen_measurement_spectrum))
    # plt.subplot(615)
    # plt.plot(chosen_measurment-chosen_signal)
    # plt.subplot(616)
    # chosen_difference_spectrum = np.fft.fft(chosen_measurment-chosen_signal)
    # # plt.plot(np.abs(chosen_difference_spectrum)) #
    # plt.plot(np.abs(chosen_difference_spectrum)[0:50]) # Majority of interference at indices 7-13 - just set them to zero (256:476 MHz for ts = 2.668e-11, or 34-63MHz for ts=2e-10)
    # plt.savefig('temp.png')
    # plt.close()