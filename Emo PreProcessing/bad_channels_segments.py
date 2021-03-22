import mne
import mne
import numpy as np
from os import path as op
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,corrmap)
subjects = ["8","9","10"]
for subject in subjects:
    # Identify Bad regions.
    fname = "C:\\Users\\Kioskar\\Desktop\\preprocessing\\raw_filtered100_" + subject + ".fif"
    save_to_path = "C:\\Users\\Kioskar\\Desktop\\Testing exjobb\\EmoDecode1\\"

    raw = mne.io.Raw(fname)
    def fixTriggers(x):
        y = (np.round(x)).astype(int)
        return y
    raw.load_data()
    raw.apply_function(fixTriggers,picks='Trigger')
    events = mne.find_events(raw,output='onset',min_duration=1, consecutive=True, initial_event=True)
    #fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'],first_samp=raw.first_samp)
    #fig.subplots_adjust(right=0.7)  # make room for legend
    channel_names = raw.ch_names
    channel_names =  channel_names[0:-3]
    print(events)
    print(channel_names)
    print(raw.info)
    # Export to find out-of-batch segments.
    df = pd.DataFrame(events)
    df.to_csv("all_events",header=False,index=False)
    oob = np.loadtxt("bad_seg_"+subject,delimiter=',')
    oob[-1,-1] = len(raw["Trigger"][0][0])-oob[-1,0]
    oob[0,0] = 1000
    oob = oob/1000-1
    # Remove bad segments.
    bad_seg = mne.Annotations(onset=oob[:,0],
                               duration=oob[:,1],
                               description=['bad'] * len(oob[:,0]))
    #raw = raw.copy()
    raw.annotations.append = bad_seg
    print(raw.info)

    # ICA artefact removal. Eye and not eye.
    ica = ICA(n_components=15, random_state=97)
    ica.fit(raw,picks=['eeg'],reject_by_annotation=True)
    #raw.load_data()
    #ica.plot_sources(raw, show_scrollbars=False)
    #plt.show()
    #ica.plot_components(outlines='skirt')
    #plt.show()

    # Choose which ICs to exclude in raw data.
    ica.exclude = range(0,15)
    #3: 6: 8: 1 8 9: 10 10: 10 Exclude ICs. from raw.
    ica.apply(raw)

    # Apply epoching with decimation and baseline correction.
    epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=1.5, baseline=(-0.1,0),reject_by_annotation=True)#,event_id=event_dict)
    epochs.load_data()
    epochs.resample(512)
    # Inspect so things are correct.

    classes = ['100','110','120','130','140','150']
    classes_save = ["CnegF","BneuF","AposF","CnegS","BNeuS","APosS"]
    #epochs[classes].plot()
    epochs.drop_bad()
    plt.show()

    if(subject!="10"):
        subject = "0" + subject

    for i in range(0,6):
        length = len(epochs[classes[i]])
        print(length)
        for j in range(0,length):
            df = epochs[classes[i]][j].to_data_frame()
            df = pd.DataFrame(df)
            print(df[channel_names])
            df[channel_names].to_csv(save_to_path + "Study/Morlet/Subj" + subject + "/" +  classes_save[i] + "_" + str(j),header=False,index=False)
            print(save_to_path + "Study/Morlet/Subj" + subject + "/" + classes_save[i] + "_" + str(j))

    classes = ['20','21','22','23','24','25']
    classes_save = ["CnegF","BneuF","AposF","CnegS","BNeuS","APosS"]
    for i in range(0,6):
        length = len(epochs[classes[i]])
        print(length)
        for j in range(0,length):
            df = epochs[classes[i]][j].to_data_frame()
            df = pd.DataFrame(df)
            print(df[channel_names])
            df[channel_names].to_csv(save_to_path + "Retrieval/Morlet/Subj" + subject + "/" + classes_save[i] + "_" + str(j),header=False,index=False)
            print(save_to_path + "Retrieval/Morlet/Subj" + subject + "/" + classes_save[i] + "_" + str(j))
