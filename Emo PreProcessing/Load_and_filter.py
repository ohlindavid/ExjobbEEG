import mne
import mne
import numpy as np
from os import path as op
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,corrmap)


data_path_dat = "C:/Users/Kioskar/Desktop/Raw Sterre data/subj10/OriginalData.dat"
raw = mne.io.read_raw_curry(data_path_dat, preload=False, verbose=None)
print('Data loaded, with preload=False: ' + data_path_dat +  ', ' + str(raw.info["nchan"]) + ' channels X ' + str(raw.n_times) + ' points')
channel_names = raw.ch_names
print(channel_names)
channel_names =  channel_names[0:-3]
raw.set_channel_types({'Trigger':'stim','VEOG-':'eog',})
def fixTriggers(x):
    y = (np.round(x*10**6-61440)).astype(int)
    return y
raw.load_data()
raw.apply_function(fixTriggers,picks='Trigger')
events = mne.find_events(raw,output='onset',min_duration=1, consecutive=True, initial_event=True)
np.savetxt("events_10",events,delimiter=',')

print(events[:])
print(raw.info)
#raw.plot()
#plt.show()
#fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'],first_samp=raw.first_samp)
#fig.subplots_adjust(right=0.7)  # make room for legend

# Apply Lowpass & Highpass filter.
lowfreq = 0.5
highfreq = 100
raw.filter(lowfreq,highfreq,picks=['eeg'])

# Save the Raw.
raw.save("raw_filtered100_10.fif",overwrite=True)
print(events)
print(raw.info)
print(raw)
