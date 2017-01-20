class MCS_Data_Channel(object):

    def get_stats(self, data):
        data_mean = np.mean(data)
        data_std = np.std(data)
        return data_mean, data_std

    def __init__(self, voltage_data, time_data, channel_number, sampling_rate):
        self.voltage_data = voltage_data
        self.time_data = time_data
        self.channel_number = channel_number
        self.sampling_rate = sampling_rate

        voltage_mean, voltage_std = self.get_stats(voltage_data)
        self.voltage_mean = voltage_mean
        self.voltage_std = voltage_std
        
        self.spike_threshold = voltage_mean + 5*voltage_std # 5 stds above the mean

def get_data(full_file_path):
    print ("Processing " + full_file_path) 
    fd = ns.File(full_file_path)
    sampling_rate = (1.0/fd.time_stamp_resolution)
    counter = len(fd.entities)
    all_channels = []

    for i in range(0, counter):
        analog1 = fd.entities[i] #open channel 
        if analog1.entity_type == 2:
            channel = analog1.label[-2:] #identify channel 
            if not channel.startswith('A'): #if it is not an analog channel and if the channel is in the range of channels in the pattern
                data, times, count = analog1.get_data() #load data
                min_data = abs(min(data))
                data2 = [d + min_data for d in data]
                mcs_data_channel = MCS_Data_Channel(data2, times, channel, sampling_rate)
                all_channels.append(mcs_data_channel)
    return all_channels


print("Importing Libraries...\n")
import os, time
import matplotlib.pyplot as plt 
import numpy as np
import neuroshare as ns
from Tkinter import Tk
from tkFileDialog import askdirectory

full_file_path = 'spont.mcd'

all_channels = get_data(full_file_path)

for channel in all_channels:
    plt.plot(channel.voltage_data)
    threshold_line = [channel.spike_threshold for x in channel.voltage_data]
    plt.plot(threshold_line, color='r')
    plt.title(str(channel.channel_number))
    plt.show()