## Image 1 - 2D reconstructions
import matplotlib.pyplot as plt
from matplotlib.pyplot import legend
import seaborn as sns
# sns.set_theme(style="darkgrid")

import numpy as np
import os, pickle

# with open(os.path.join('results', '20201103_arun_extended_and_generative_testononlyfirsttwoseqs_interference_-15', 'test', 'results.pkl' ), 'rb') as f:
#     dataset_interference =  pickle.load(f)

with open(os.path.join('results', '20201103_arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps_90', 'test', 'results.pkl' ), 'rb') as f: # TODO: Replace this
    dataset_randommissing =  pickle.load(f)

# with open(os.path.join('results', '20201103_arun_extended_and_generative_testononlyfirsttwoseqs_blockgaps_50', 'test', 'results.pkl' ), 'rb') as f:
    # dataset_gapmissing =  pickle.load(f)

with open(os.path.join('scratch', 'with_baselines', '20201103_arun_extended_and_generative_testononlyfirsttwoseqs_interference_-15_results.pkl'), 'rb') as f:
    dataset_interference =  pickle.load(f)

with open(os.path.join('scratch', 'with_baselines', '20201103_arun_extended_and_generative_testononlyfirsttwoseqs_blockgaps_90_results.pkl'), 'rb') as f:
    dataset_gapmissing =  pickle.load(f)

with open(os.path.join('scratch', 'results_randomgap_90.pkl' ), 'rb') as f: 
    dataset_randommissing_2 =  pickle.load(f)

dataset_randommissing['baseline_outputs'] = dataset_randommissing_2['outputs']

from mpl_toolkits.axes_grid1 import ImageGrid, AxesGrid

# common-processing
def process_data(data):
    data = data/np.max(np.abs(data))
    data = data + 1e-32
    data_logcompressed = 20*np.log10(np.abs(data))
    return data_logcompressed

start_idx = 400
end_idx = 800
# im1
im1 = process_data(np.squeeze(dataset_interference['signals']).T[start_idx:end_idx,:1800])
# im2
im2 = process_data(np.squeeze(dataset_interference['measurements']).T[start_idx:end_idx,:1800])
# im3
im3 = process_data(np.squeeze(dataset_interference['baseline_outputs']).T[start_idx:end_idx,:1800])
# im4
im4 = process_data(np.squeeze(dataset_interference['outputs']).T[start_idx:end_idx,:1800])
# im5
im5 = process_data(np.squeeze(dataset_randommissing['signals']).T[start_idx:end_idx,:1800])
# im6
im6 = process_data(np.squeeze(dataset_randommissing['measurements']).T[start_idx:end_idx,:1800])
# im7
im7 = process_data(np.squeeze(dataset_randommissing['baseline_outputs']).T[start_idx:end_idx,:1800]) # TODO: Switch to this!
# im7 = process_data(np.squeeze(dataset_randommissing['outputs']).T[start_idx:end_idx,:1800])
# im8
im8 = process_data(np.squeeze(dataset_randommissing['outputs']).T[start_idx:end_idx,:1800])
# im9
im9 = process_data(np.squeeze(dataset_gapmissing['signals']).T[start_idx:end_idx,:1800])
# im10
im10 = process_data(np.squeeze(dataset_gapmissing['measurements']).T[start_idx:end_idx,:1800])
# im11
im11 = process_data(np.squeeze(dataset_gapmissing['baseline_outputs']).T[start_idx:end_idx,:1800])
# im12
im12 = process_data(np.squeeze(dataset_gapmissing['outputs']).T[start_idx:end_idx,:1800])

fig = plt.figure(figsize=(9., 12.))
grid = AxesGrid(fig, (1,1,1),  # similar to subplot(111)
                 nrows_ncols=(3, 4),  # creates 3x3 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 share_all=True,
                 label_mode="L",
                 cbar_mode="single",
                #  cbar_mode = "edge",
                 cbar_location="right",                 
                 )
imgs_to_delete =[]
for idx, (ax, im) in enumerate(zip(grid, [im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11, im12])):
    if idx == 0:
        ax.set_title('Clean')
        # ax.set_ylabel('Bandlimited \n Interference', rotation=0, labelpad=10)
        ax.set_ylabel('Radio Frequency \n Interference (RFI)', fontsize = 'large')
    elif idx ==1:
        ax.set_title('Noisy')
    elif idx ==2:
        ax.set_title('Baseline')
    elif idx ==3:
        ax.set_title('UNet')
    elif idx ==4:
        ax.set_ylabel('Random \n Spectral Gaps', fontsize='large')
        # ax.set_ylabel('Random \n Spectral Gaps', rotation=0, labelpad=10)
    elif idx ==8:
        ax.set_ylabel('Centered Block\n Spectral Gap', fontsize='large')
        # ax.set_ylabel('Block \n Spectral Gap', rotation=0, labelpad=10)

    # Iterating over the grid returns the Axes.
    # if idx not in [0,8]:
    img  = ax.imshow(im, vmin=-40, vmax=0, cmap='jet', aspect=4.0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])

    # # In my search to basically only show middle clean plot as the ones above and below are repeats...
    # if idx==0 or idx == 8:
    #     imgs_to_delete.append(img)
    #     # ax.axis('off') # This also deletes text, so instead do the below
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)
    #     ax.spines["left"].set_visible(False)
    #     ax.spines["bottom"].set_visible(False)        

# # In my search to basically only show middle clean plot as the ones above and below are repeats...
# for img in imgs_to_delete:
#     img.remove()

grid.cbar_axes[0].colorbar(img)
grid.cbar_axes[0].set_title('dB', rotation=0)

# Reference
fig.text(0.133,0.536,"(e)", color ="w")
fig.text(0.320,0.536,"(f)", color ="k")
fig.text(0.507,0.536,"(g)", color ="w")
fig.text(0.694,0.536,"(h)", color ="w")

fig.text(0.133,0.661,"(a)", color ="w")
fig.text(0.320,0.661,"(b)", color ="k")
fig.text(0.507,0.661,"(c)", color ="k")
fig.text(0.694,0.661,"(d)", color ="w")

fig.text(0.133,0.411,"(i)", color ="w")
fig.text(0.320,0.411,"(j)", color ="k")
fig.text(0.507,0.411,"(k)", color ="w")
fig.text(0.694,0.411,"(l)", color ="w")


# for cax in grid.cbar_axes:
#     cax.toggle_label(False)

plt.savefig('temp.png', dpi=200)
plt.close()

###################################################################################
## Image 2 - Axial Slice - 2 * 3 figure with signal on top FFT underneath

import pandas as pd

# common-processing
def process_data_2(data):
    data = data/np.max(np.abs(data))
    data = data + 1e-32
    col_idx = 400
    return data[col_idx, :]

t_start, t_end = 250, 800
time_idx_slice = slice(t_start, t_end)
fft_idx_slice = slice(0, 60)

data_1a = process_data_2(np.squeeze(dataset_interference['signals']))[time_idx_slice]
data_1b = process_data_2(np.squeeze(dataset_interference['measurements']))[time_idx_slice]
data_1c = process_data_2(np.squeeze(dataset_interference['baseline_outputs']))[time_idx_slice]
data_1d = process_data_2(np.squeeze(dataset_interference['outputs']))[time_idx_slice]

data_2a = np.abs(np.fft.rfft(data_1a))[fft_idx_slice]
data_2b = np.abs(np.fft.rfft(data_1b))[fft_idx_slice]
data_2c = np.abs(np.fft.rfft(data_1c))[fft_idx_slice]
data_2d = np.abs(np.fft.rfft(data_1d))[fft_idx_slice]

data_3a = process_data_2(np.squeeze(dataset_randommissing['signals']))[time_idx_slice]
data_3b = process_data_2(np.squeeze(dataset_randommissing['measurements']))[time_idx_slice]
data_3c = process_data_2(np.squeeze(dataset_randommissing['baseline_outputs']))[time_idx_slice]
# data_3c = process_data_2(np.squeeze(dataset_randommissing['outputs']))[time_idx_slice] # TODO
data_3d = process_data_2(np.squeeze(dataset_randommissing['outputs']))[time_idx_slice]

data_4a = np.abs(np.fft.rfft(data_3a))[fft_idx_slice]
data_4b = np.abs(np.fft.rfft(data_3b))[fft_idx_slice]
data_4c = np.abs(np.fft.rfft(data_3c))[fft_idx_slice]+0.3 #
data_4d = np.abs(np.fft.rfft(data_3d))[fft_idx_slice]

data_5a = process_data_2(np.squeeze(dataset_gapmissing['signals']))[time_idx_slice]
data_5b = process_data_2(np.squeeze(dataset_gapmissing['measurements']))[time_idx_slice]
data_5c = process_data_2(np.squeeze(dataset_gapmissing['baseline_outputs']))[time_idx_slice]
data_5d = process_data_2(np.squeeze(dataset_gapmissing['outputs']))[time_idx_slice]

data_6a = np.abs(np.fft.rfft(data_5a))[fft_idx_slice]
data_6b = np.abs(np.fft.rfft(data_5b))[fft_idx_slice]
data_6c = np.abs(np.fft.rfft(data_5c))[fft_idx_slice]
data_6d = np.abs(np.fft.rfft(data_5d))[fft_idx_slice]

sampling_frequency = 1/(2.668e-11)
freq_vector = np.array(list(range(0, dataset_interference['signals'].shape[-1]//2+1))) * sampling_frequency/dataset_interference['signals'].shape[-1]

freq_vector = freq_vector/1e9 # To move it to GHz
freq_vector = freq_vector[fft_idx_slice] # Slice it to same size as data to display

time_vector = list(range(dataset_interference['signals'].shape[-1]))
time_vector = time_vector[t_start:t_end]

# LIMIT YLIM
data_dict_1 = {'Clean': data_1a, 'Noisy': data_1b, 'Baseline': data_1c, 'UNet': data_1d, 'x': time_vector}
data_dict_2 = {'Clean': data_2a, 'Noisy': data_2b, 'Baseline': data_2c, 'UNet': data_2d, 'x': freq_vector}
data_dict_3 = {'Clean': data_3a, 'Noisy': data_3b, 'Baseline': data_3c, 'UNet': data_3d, 'x': time_vector}
data_dict_4 = {'Clean': data_4a, 'Noisy': data_4b, 'Baseline': data_4c, 'UNet': data_4d, 'x': freq_vector}
data_dict_5 = {'Clean': data_5a, 'Noisy': data_5b, 'Baseline': data_5c, 'UNet': data_5d, 'x': time_vector}
data_dict_6 = {'Clean': data_6a, 'Noisy': data_6b, 'Baseline': data_6c, 'UNet': data_6d, 'x': freq_vector}

data_frame_1 = pd.DataFrame(data_dict_1)
data_frame_2 = pd.DataFrame(data_dict_2)
data_frame_3 = pd.DataFrame(data_dict_3)
data_frame_4 = pd.DataFrame(data_dict_4)
data_frame_5 = pd.DataFrame(data_dict_5)
data_frame_6 = pd.DataFrame(data_dict_6)

# Import seaborn
import seaborn as sns
# Apply the default theme
sns.set_theme()

# fig = plt.figure(figsize=(18., 24.))

# sns.lineplot(data=data_dict_1, ax = axes[0])

fig = plt.figure(figsize=(20,10))
plt.subplot(2,3,1)
sns.lineplot(x='x', y='value', hue='variable', data=pd.melt(data_frame_1, ['x']))
plt.ylabel('Magnitude')
plt.xlabel('Sample')
plt.title('Radio Frequency Interference (RFI)', fontsize = 'x-large')
plt.legend([],[], frameon=False)
plt.subplot(2,3,2)
sns.lineplot(x='x', y='value', hue='variable', data=pd.melt(data_frame_3, ['x']))
plt.legend([],[], frameon=False)
plt.xlabel('Sample')
plt.title('Random Spectral Gaps', fontsize = 'x-large')
plt.subplot(2,3,3)
g = sns.lineplot(x='x', y='value', hue='variable', data=pd.melt(data_frame_5, ['x']))
plt.xlabel('Sample')
plt.title('Centered Block Spectral Gap', fontsize = 'x-large')
handles, labels = g.get_legend_handles_labels()
g.legend(handles=handles[0:], labels=labels[0:]) # From https://stackoverflow.com/questions/51579215/remove-seaborn-lineplot-legend-title
plt.legend(loc='lower left')
plt.subplot(2,3,4)
 #https://stackoverflow.com/questions/52308749/how-do-i-create-a-multiline-plot-using-seaborn
sns.lineplot(x='x', y='value',  hue='variable', data=pd.melt(data_frame_2, ['x']))
plt.legend([],[], frameon=False)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Magnitude')
plt.subplot(2,3,5)
sns.lineplot(x='x', y='value',  hue='variable', data=pd.melt(data_frame_4, ['x']))
plt.xlabel('Frequency (GHz)')
plt.legend([],[], frameon=False)
plt.ylabel('')
plt.subplot(2,3,6)
sns.lineplot(x='x', y='value',  hue='variable', data=pd.melt(data_frame_6, ['x']))
# sns.lineplot(data=data_dict_6, x=freq_vector)
plt.legend([],[], frameon=False)
plt.xlabel('Frequency (GHz)')
plt.ylabel('')

# Reference
fig.text(0.133,0.845,"(a)", color ="k")
fig.text(0.407,0.845,"(b)", color ="k")
fig.text(0.681,0.845,"(c)", color ="k")

fig.text(0.133,0.428,"(d)", color ="k")
fig.text(0.407,0.428,"(e)", color ="k")
fig.text(0.681,0.428,"(f)", color ="k")



plt.savefig('temp_2.png', dpi=200)
plt.close()

# fmri = sns.load_dataset("fmri")
# sns.relplot(
#     data=fmri, kind="line",
#     x="timepoint", y="signal", col="region",
#     hue="event", style="event",
# )

###################################################################################
## Image 3 - Quantitative evaluation of the results - can code it up to autodo instead of depending on me writing lists
# 3 graphs from left to right - interference, random-missing, gap misisng
# in each graph, we have noisy-simulated, baseline-simulated, unet-simulated, noisy-real, baseline-real, unet-real

#( For this, the SNR is on the x axis itself, doesn't make sense to include noisy data here
data_1b = np.array([-6.69,-1.77,3.05,7.56,11.38,14.09])
data_1c = np.array([12.85,18.72,22.72,25.89,28.23,29.59])
data_2b = np.array([-6.72,-1.75,3.05,7.42,11.1,13.64])
data_2c = np.array([11.48,14.89,17.94,20.78,23.07,24.35])
x = np.array([-15,-10,-5,0,5,10])

data_dict_1 = {'Simulated - Baseline': data_1b, 'Simulated - UNet': data_1c, 'Real - Baseline': data_2b, 'Real - UNet': data_2c, 'x': x}

data_3a = np.array([3.23, 2.54, 1.78, 1.26, 0.69])
# data_3b = np.array([5,5,5,5,5])
data_3b = np.array([21.31, 20.33,16.35,11.47,6.22])
data_3c = np.array([22.71, 22.44, 21.74,20.94,19.24])
data_4a = np.array([3.2,2.52,1.79,1.26,0.69])
# data_4b = np.array([6,6,6,6,6]) # TODO
data_4b = np.array([22.2, 21.71,19.71,15.83,8.77])
data_4c = np.array([15.59,14.94,13.91,12.91,10.99])
x = np.array([50, 60, 70, 80, 90])

data_dict_2 = {'Simulated - Noisy': data_3a, 'Simulated - Baseline': data_3b, 'Simulated - UNet': data_3c, 'Real - Noisy': data_4a, 'Real - Baseline': data_4b, 'Real - UNet': data_4c, 'x': x}

data_5a = np.array([1.29, 0.86,0.56,0.4,0.34])
data_5b = np.array([11.87,9.84,8.42,6.56,4.74])
data_5c = np.array([23.7,22.61,20.72,19.55,18.43])
data_6a = np.array([1.37,0.92,0.65,0.48,0.42])
data_6b = np.array([8.05,7.09,3.92,1.26,1.08])
data_6c = np.array([13.25,12.08,10.66,10.06,9.63])
x = np.array([50, 60, 70, 80, 90])

data_dict_3 = {'Simulated - Noisy': data_5a, 'Simulated - Baseline': data_5b, 'Simulated - UNet': data_5c, 'Real - Noisy': data_6a, 'Real - Baseline': data_6b, 'Real - UNet': data_6c, 'x': x}


data_frame_1 = pd.DataFrame(data_dict_1)
data_frame_2 = pd.DataFrame(data_dict_2)
data_frame_3 = pd.DataFrame(data_dict_3)

# Import seaborn
import seaborn as sns
# Apply the default theme
sns.set_theme()

# fig = plt.figure(figsize=(18., 24.))

# sns.lineplot(data=data_dict_1, ax = axes[0])

fig = plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
g = sns.lineplot(x='x', y='value', hue='variable', marker="o", data=pd.melt(data_frame_1, ['x']), legend=False, palette=sns.color_palette()[1:3]+sns.color_palette()[4:6])
plt.ylabel('Output SNR (dB)')
plt.xlabel('Input SNR (dB)')
plt.title('Radio Frequency Interference (RFI)', fontsize = 'x-large')
plt.ylim(-7.5, 35)
plt.subplot(1,3,2)
g = sns.lineplot(x='x', y='value', hue='variable', marker="o", data=pd.melt(data_frame_2, ['x']), legend=False)
plt.ylabel('SNR (dB)')
plt.xlabel('Missing Percentage')
plt.title('Random Spectral Gaps', fontsize = 'x-large')
plt.ylim(0, 35)

plt.subplot(1,3,3)
g = sns.lineplot(x='x', y='value', hue='variable', marker="o", data=pd.melt(data_frame_3, ['x']))
plt.ylabel('SNR (dB)')
plt.xlabel('Missing Percentage')
plt.title('Centered Block Spectral Gap', fontsize = 'x-large')
plt.ylim(0, 35)
# legend = g.legend()
# legend.texts[0].set_text("Whatever else")
handles, labels = g.get_legend_handles_labels()
g.legend(handles=handles[0:], labels=labels[0:]) # From https://stackoverflow.com/questions/51579215/remove-seaborn-lineplot-legend-title
# plt.setp(g.get_legend().get_texts(), fontsize='8') # for legend text

# Reference
fig.text(0.133,0.795,"(a)", color ="k")
fig.text(0.407,0.795,"(b)", color ="k")
fig.text(0.681,0.795,"(c)", color ="k")


plt.savefig('temp_3.png', dpi=200)
plt.close()

# fmri = sns.load_dataset("fmri")
# sns.relplot(
#     data=fmri, kind="line",
#     x="timepoint", y="signal", col="region",
#     hue="event", style="event",
# )

## Image 4 - Simple plotting for PlotNeuralNet
clean_data = data_frame_5['Clean'].to_numpy()
noisy_data = data_frame_5['Noisy'].to_numpy()
# fig = plt.figure(figsize=(9., 12.))
fig = plt.figure()
plt.subplot(211)
plt.plot(clean_data)
plt.axis("off")
plt.subplot(212)
plt.plot(noisy_data)
plt.axis("off")
plt.savefig('temp_4.png', dpi=300)
plt.close()