
# ICA artefact removal. Eye and not eye.
ica = ICA(n_components=15, random_state=97)
ica.fit(raw,picks=['eeg'])
#raw.load_data()
#ica.plot_sources(raw, show_scrollbars=False)
#plt.show()
#ica.plot_components()
#plt.show()
# blinks

#for i in range(0,15):
#    ica.plot_properties(raw, picks=[i])
#    plt.show()
# Choose which ICs to exclude in raw data.
ica.exclude = [0,1,3,4,5,9,11,12,13]
# EOG: 9, 11
# Muscle: 12 ,13
# Hearbeat? 2

# Exclude ICs. from raw.
ica.apply(raw)

plot = raw.plot(events=events,duration=60, n_channels=65,show=True,block=True)
plt.show()

# Apply epoching with decimation and baseline correction.

epochs = mne.Epochs(raw, events, tmin=-1.5, tmax=2.5, baseline=(-0.1,0))#,event_id=event_dict)
epochs.load_data()
epochs.resample(512)
epochs.drop_bad()

# Inspect so things are correct.


classes = ['100','110','120','130','140','150']
classes_save = ["APosF","APosS","BNeuF","BNeuS","CNegF","CNegS"]
for i in range(0,6):
    length = len(epochs[classes[i]])
    print(length)
    for j in range(0,length):
        df = epochs[classes[i]][j].to_data_frame()
        df = pd.DataFrame(df)
        print(df[channel_names])
        df[channel_names].to_csv(save_to_path + classes_save[i] + "_" + str(j),header=False,index=False)
