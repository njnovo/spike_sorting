import spikeinterface.full as si
import spikeinterface.widgets as sw
import spikeinterface.extractors as se 

import matplotlib.pyplot as plt
import numpy as np



si.set_default_plotter_backend(backend="matplotlib")





# file path
recording_file = '/Users/nielsnovotny/Downloads/cambridge_data.bin'


# load the generic date
# parameters to load the bin/dat format
num_channels = int(64)
sampling_frequency = 20000
gain = 0.195
channel_offset = 0
dtype="int16"
time_axis = 1

recording = si.read_binary(recording_file, num_channels=num_channels, sampling_frequency=sampling_frequency,
                                        dtype=dtype, gain_to_uV=gain)



# blackrock data can be loaded with the following
# recording = se.read_blackrock(folder_path="blackrock-folder")
# if the data is from the probe: recording.get_probe()

plt.figure(figsize=(10, 3))


#plot voltage traces (shows raw or preprocessed signals)
# basic = sw.plot_traces(recording, backend="matplotlib", mode ="line", time_range=(0, 30), channel_ids=[0, 1, 2, 3]) #mode = "line" each channel’s voltage trace individually , "map" color-coded matrix of voltage values, or "auto" ()
# plt.show()

#plot spike raster (spike times per unit)
raster = sw.plot_rasters(recording, backend="matplotlib")
plt.show()





