import numpy as np
import torch
import torch.nn
import os, sys
import torch
import pdb
from itertools import repeat
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from utils import snr_akshay

import multiprocessing
from multiprocessing import Pool
PROCESSES = multiprocessing.cpu_count()//2

# from .custom_losses import dice_coeff, psnr_loss, RMSELoss, LSDLoss
from .custom_losses import snr_loss

def eval_net(g_net, criterion_g, params, val_or_test_string, val_or_test_loader):
    # If test set results are to be stored, initialize the requisite variables
    
    tot_l1loss = 0.
    criterion_l1loss = torch.nn.L1Loss()
    tot_MSELoss = 0.
    criterion_mse = torch.nn.MSELoss()
    tot_smoothl1loss = 0.
    criterion_smoothl1loss = torch.nn.SmoothL1Loss()
    tot_snrloss = 0.
    tot_snrloss_donothing = 0.
    tot_snrloss_gain = 0.
    criterion_snr = snr_loss

    tot_loss = 0.
    running_counter = 0
    with torch.no_grad():
    # with torch.set_grad_enabled(False): # Alternatively
        for i, sample in enumerate(val_or_test_loader):
            # print(f"Current Test Idx is {i}")
            target_output, input_data = sample

            input_data = input_data.to(params['device'])
            target_output = target_output.to(params['device'])

            # Pass the minibatch through the network to get predictions WITH AUTOCASTING
            with autocast(enabled=False):
                preds = g_net(input_data)

                # Obtain the clean and degraded results and loss values
                # tot_PSNR += psnr_loss(preds, target_output).item() * input_data.shape[0] # mean * number of samples # Occasionally throwing an erorr, got annoyed, setting it to 0
                if params['criterion_g'] == 'l1andfftloss':                    
                    tot_loss +=         criterion_g(input_data, preds, target_output).item() * input_data.shape[0]                  # mean * number of samples
                else:
                    tot_loss +=         criterion_g(preds.view(-1), target_output.view(-1)).item() * input_data.shape[0]            # mean * number of samples
                tot_l1loss +=       criterion_l1loss(preds.view(-1), target_output.view(-1)).item() * input_data.shape[0]           # mean * number of samples
                tot_smoothl1loss += criterion_smoothl1loss(preds.view(-1), target_output.view(-1)).item() * input_data.shape[0]     # mean * number of samples
                tot_MSELoss +=      criterion_mse(preds.view(-1), target_output.view(-1)).item() * input_data.shape[0]              # mean * number of samples
                tot_snrloss +=      criterion_snr(preds, target_output).item() * input_data.shape[0]                                # mean * number of samples
                tot_snrloss_donothing += criterion_snr(input_data, target_output).item() * input_data.shape[0]                      # mean * number of samples                
            running_counter = running_counter + input_data.shape[0]
    tot_snrloss_gain += tot_snrloss-tot_snrloss_donothing
    return (tot_l1loss/running_counter, tot_MSELoss/running_counter, tot_smoothl1loss/running_counter, tot_snrloss/running_counter, tot_snrloss_donothing/running_counter, tot_snrloss_gain/running_counter, tot_loss/running_counter)

def write_imgs(g_net, params, val_or_test_string, val_or_test_loader):
    # If test set results are to be stored, initialize the requisite variables
    running_counter = 0
    with torch.no_grad():
    # with torch.set_grad_enabled(False): # Alternatively
        for i, sample in enumerate(val_or_test_loader):
            # print(f"Current Test Idx is {i}")
            target_output, input_data = sample

            input_data = input_data.to(params['device'])
            target_output = target_output.to(params['device'])

            # Pass the minibatch through the network to get predictions WITH AUTOCASTING
            with autocast(enabled=False):
                preds = g_net(input_data)
            # if i%100==0: # Don't need this any more... just take some, say first 5, of the results. With a test batch size of 1000, => 50000/1000=50 batches => 50*5=250 images
                num_images_per_batch = 5 # Number of images per batch to plot and store
                multi_pool = multiprocessing.Pool(processes=PROCESSES)
                input_data_cpu    = input_data.cpu().numpy()
                preds_cpu         = preds.cpu().numpy()
                target_output_cpu = target_output.cpu().numpy()            
                # params
                # out_list = np.array(multi_pool.starmap(inner_loop_fn, zip(range(running_counter, running_counter + input_data.shape[0], 1),  input_data_cpu, preds_cpu, target_output_cpu, repeat(params), repeat(val_or_test_string),)))
                # out_list = np.array(multi_pool.starmap(inner_loop_fn, zip(range(running_counter, running_counter + input_data.shape[0], input_data.shape[0]//2),  input_data_cpu[np.ix_([0, input_data.shape[0]//2])], preds_cpu[np.ix_([0, input_data.shape[0]//2])], target_output_cpu[np.ix_([0, input_data.shape[0]//2])], repeat(params), repeat(val_or_test_string),)))
                out_list = np.array(multi_pool.starmap(inner_loop_fn, zip(range(running_counter, running_counter + num_images_per_batch, 1),  input_data_cpu[0:num_images_per_batch], preds_cpu[0:num_images_per_batch], target_output_cpu[0:num_images_per_batch], repeat(params), repeat(val_or_test_string),)))
                # Close the parallel pool
                multi_pool.close()
                multi_pool.join()

            running_counter = running_counter + input_data.shape[0]
    return

def inner_loop_fn(idx, input_data, preds, target_output, params, val_or_test_string):    

    if params['save_test_val_results'] is not False:
        # 0 - Create results_dir to output files to
        os.makedirs(params['results_dir'], exist_ok=True)
        os.makedirs(os.path.join(params['results_dir'], val_or_test_string), exist_ok=True)
        
        # 1 - Calculate metrics, plot the image and write it to file
        if len(target_output.shape)==2: #1xd
            SNR_meas, SNR_pred, SNR_gain = snr_akshay(target_output, input_data, preds)
        else: #1xslowTimexd
            SNR_meas, SNR_pred, SNR_gain = snr_akshay(target_output[:,0,:], input_data[:,0,:], preds[:,0,:])
            target_output = target_output[:,0,:]
            input_data = input_data[:,0,:]
            preds = preds[:,0,:]
            
        plt.figure()
        # TODO: generalize it to images with more than 1 column
        plt.plot(np.squeeze(target_output), 'b-', label='ground truth')
        plt.plot(np.squeeze(input_data), 'g-', label='noisy input')
        plt.plot(np.squeeze(preds), 'r-', label='network output')
        plt.legend()
        plt.title('Predicted signal %s:%d from UNet trained on L1\nSNRip = %.3f SNRop = %.3f SNR Gain = %.3f dB'%(params['dataset'], idx+1, SNR_meas, SNR_pred, SNR_gain))
        plt.tight_layout()
        plt.savefig(os.path.join(params['results_dir'], val_or_test_string, '%s_%d.png'%(params['dataset'], idx+1)))
        plt.close()
    
    # return (PESQ_donothing, PESQ_enhanced)
    return (True, True)
