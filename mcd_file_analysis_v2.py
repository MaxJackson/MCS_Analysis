class MCS_Spike(object): # The MCS_Spike object, with fields for channel#, start time, the time series of data, and the spike height
    channel = 0
    start_time = 0.0
    timestamp_data = [0.0]
    voltage_data = [0.0]

    def analyze_positive_peak(self, voltage_data, timestamp_data):
        area_under_curve_positive = 0.0
        first_half_peak_positive = 0
        second_half_peak_positive = 0
        spike_positive_start_time = 0
        spike_positive_end_time = 0
        voltage_data = self.voltage_data
        time_data = self.timestamp_data

        spike_max = max(voltage_data)
        max_index = np.argmax(voltage_data)
        spike_max_time = time_data[max_index]

        for p in range((np.argmax(voltage_data)), 0, -1):
            area_under_curve_positive += (float(voltage_data[p]))
            if voltage_data[p] <= math.floor((max(voltage_data)/2)):
                first_half_peak_positive = p
            if voltage_data[p] <= 0:
                spike_positive_start_time = time_data[p]
                break
        
        #Start from the maxVoltage timepoint and work forwards until you hit a zero 
        for p in range((np.argmax(voltage_data)), 1000-(np.argmax(voltage_data)), 1):
            area_under_curve_positive += (float(voltage_data[p]))
            if voltage_data[p] <= (math.floor((max(voltage_data))/2)):
                second_half_peak_positive = p
            if voltage_data[p] <= 0:
                spike_positive_end_time = time_data[p]
                break 
            if p == len(timestamp_data) - 1:
                spike_positive_start_time = timestamp_data[p]
                break
                    
        spike_half_peak_width_positive = (second_half_peak_positive - first_half_peak_positive) * 0.02

        # print(spike_max)
        # print(spike_max_time)
        # print(spike_positive_start_time)
        # print("")
        if spike_max_time - spike_positive_start_time != 0:
            spike_positive_slope = spike_max / (spike_max_time - spike_positive_start_time)
        else:
            spike_positive_slope = 0

        return spike_max, spike_positive_start_time, spike_max_time, spike_positive_end_time, area_under_curve_positive, spike_half_peak_width_positive, spike_positive_slope

    def analyze_negative_peak(self, voltage_data, timestamp_data):
        area_under_curve_negative = 0
        first_half_peak_negative = 0
        second_half_peak_negative = 0
        spike_negative_start_time = 0
        spike_negative_end_time = 0
        voltage_data = self.voltage_data
        time_data = self.timestamp_data

        spike_min = min(voltage_data)

        min_index = np.argmin(voltage_data)

        spike_min_time = time_data[min_index]

        #Start from the minVoltage timepoint and work backwards until you hit a zero                    
        for p in range(min_index, 0, -1):
            area_under_curve_negative += (float(voltage_data[p]))
            if voltage_data[p] >= (math.floor((min(voltage_data))/2)):
                first_half_peak_negative = p
            if voltage_data[p] >= 0:
                spike_negative_start_time = timestamp_data[p]
                break
        
        #Start from the minVoltage timepoint and work forwards until you hit a zero 
        for p in range(min_index, 1000-(np.argmin(voltage_data)), 1):
            area_under_curve_negative += (float(voltage_data[p]))
            if voltage_data[p] >= (math.floor((min(voltage_data))/2)):
                second_half_peak_negative = p
            if voltage_data[p] >= 0:
                spike_negative_end_time = timestamp_data[p]
                break 
            if p == len(timestamp_data) - 1:
                spike_negative_end_time = timestamp_data[p]
                break

        
        spike_half_peak_width_negative = (second_half_peak_negative - first_half_peak_negative) * 0.02

        spike_negative_slope = spike_min / (spike_min_time - spike_negative_start_time)

        return spike_min, spike_negative_start_time, spike_min_time, spike_negative_end_time, area_under_curve_negative, spike_half_peak_width_negative, spike_negative_slope

    def __init__(self, channel, voltage_data, timestamp_data): 
        
        self.channel = channel
        self.voltage_data = [x - voltage_data[0] for x in voltage_data]
        self.timestamp_data = timestamp_data

        # Positive Peak
        spike_max, spike_positive_start_time, spike_max_time, spike_positive_end_time, area_under_curve_positive, spike_half_peak_width_positive, spike_positive_slope = self.analyze_positive_peak(voltage_data, timestamp_data)

        self.spike_max = spike_max      
        self.spike_positive_start_time = spike_positive_start_time
        self.spike_max_total_time = spike_max_time
        self.spike_positive_end_time = spike_positive_end_time
        self.positive_time = spike_positive_end_time - spike_positive_start_time
        self.area_under_curve_positive = area_under_curve_positive
        self.spike_half_peak_width_positive = spike_half_peak_width_positive
        self.spike_positive_slope = spike_positive_slope  

        # Negative Peak
        spike_min, spike_negative_start_time, spike_min_time, spike_negative_end_time, area_under_curve_negative, spike_half_peak_width_negative, spike_negative_slope = self.analyze_negative_peak(voltage_data, timestamp_data)

        self.spike_min = spike_min
        self.spike_negative_start_time = spike_negative_start_time 
        self.spike_min_time = spike_min_time
        self.spike_negative_end_time = spike_negative_end_time
        self.spike_negative_total_time = spike_negative_end_time - spike_negative_start_time
        self.area_under_curve_negative = area_under_curve_negative
        self.spike_half_peak_width_negative = spike_half_peak_width_negative
        self.spike_negative_slope = spike_negative_slope
        
        self.area_under_curve_total = area_under_curve_positive + area_under_curve_negative

        self.spike_max_min_interval = spike_min_time - spike_max_time

class MCS_Data_Channel(object):

    def get_waveform_stats(self, all_spikes):
        all_spikes = self.all_spikes
        if len(all_spikes) == 0:
            return [0], [0]

        all_spike_data = []
        average_waveform = []
        std_waveform = []
        voltage_index_data = []

        for spike in all_spikes:
            all_spike_data.append(spike.voltage_data)

        for voltage_index in range(0, len(all_spike_data[0])):
            voltage_index_data = []
            for spike_index in range(0, len(all_spike_data)):
                voltage_index_data.append(all_spike_data[spike_index][voltage_index])
            average_waveform.append(np.mean(voltage_index_data))
            std_waveform.append(np.std(voltage_index_data))

        return average_waveform, std_waveform


    def get_stats(self, data):
        data_mean = np.mean(data)
        data_std = np.std(data)
        return data_mean, data_std

    def get_spikes(self, channel_number, voltage_data, time_data, spike_threshold):
        all_spikes = []
        this_spike_voltage_data = []
        this_spike_time_data = []
        pass_index = 0
        print(spike_threshold)
        for index in range(150, len(voltage_data) - 350):
            if index > pass_index:
                if voltage_data[index] > spike_threshold:
                    this_spike_voltage_data = voltage_data[index-150:index+350]
                    this_spike_time_data = time_data[index-150:index+350]
                    this_spike = MCS_Spike(channel_number, this_spike_voltage_data, this_spike_time_data)
                    all_spikes.append(this_spike)
                    pass_index = index + 350
        return all_spikes

    def get_over_threshold_number(self, voltage_data, spike_threshold):
        over = 0
        for data_point in voltage_data:
            if data_point >= spike_threshold:
                over += 1
        return over


    def __init__(self, voltage_data, time_data, channel_number, sampling_rate):
        self.voltage_data = voltage_data
        self.time_data = time_data
        self.channel_number = int(channel_number)
        self.sampling_rate = sampling_rate

        voltage_mean, voltage_std = self.get_stats(voltage_data)
        self.voltage_mean = voltage_mean
        self.voltage_std = voltage_std

        spike_threshold = voltage_mean + 5*voltage_std
        self.spike_threshold = spike_threshold# 5 stds above the mean

        #over_threshold_number = self.get_over_threshold_number(voltage_data, spike_threshold)
        #self.over_threshold_number = over_threshold_number

        all_spikes = self.get_spikes(channel_number, voltage_data, time_data, spike_threshold)
        self.all_spikes = all_spikes

        average_waveform, std_waveform = self.get_waveform_stats(all_spikes)
        self.average_waveform = average_waveform
        self.std_waveform = std_waveform

def get_data(full_file_path, channels_to_read):
    print ("Processing " + full_file_path) 
    fd = ns.File(full_file_path)
    sampling_rate = (1.0/fd.time_stamp_resolution)
    print(sampling_rate)
    counter = len(fd.entities)
    all_channels = []

    for i in range(0, counter):
        analog1 = fd.entities[i] #open channel 
        if analog1.entity_type == 2:
            channel = analog1.label[-2:] #identify channel 
            #print(channel)
            if not channel.startswith('A') and int(channel) in channels_to_read: #if it is not an analog channel and if the channel is in the range of channels in the pattern
                data, times, count = analog1.get_data() #load data
                #min_data = abs(min(data))
                #data2 = [d + min_data for d in data]
                data2 = [d + data[0] for d in data]
                mcs_data_channel = MCS_Data_Channel(data2, times, channel, sampling_rate)
                if len(mcs_data_channel.all_spikes) > 0:
                    all_channels.append(mcs_data_channel)
    return all_channels

def get_spikes_for_channel(channel):
    all_spikes = []
    this_spike_voltage_data = []
    this_spike_time_data = []
    pass_index = 0
    for index in range(150, len(channel.voltage_data) - 350):
        if index > pass_index:
            if channel.voltage_data[index] > channel.spike_threshold:
                this_spike_voltage_data = channel.voltage_data[index-150:index+350]
                this_spike_time_data = channel.time_data[index-150:index+350]
                pass_index = index + 350

def plotMEAWaveforms(channels, input_file):
    f, axarr = plt.subplots(8, 8, squeeze=True)
    plt.subplots_adjust(hspace=0.001)
    plt.subplots_adjust(wspace=0.001)
    for i in range(0, len(channels)):
        this_channel = channels[i]
        rawFlag = 0
        if rawFlag == 0:
            ypos = np.floor(this_channel.channel_number/10) - 1
            xpos = (this_channel.channel_number % 10) - 1
        if rawFlag == 1:
            xpos = np.floor(this_channel.channel_number/10) - 1
            ypos = (this_channel.channel_number % 10) - 1
        Xs = range(0, len(this_channel.average_waveform))
        axarr[xpos, ypos].plot(this_channel.average_waveform)
        axarr[xpos, ypos].errorbar(Xs, this_channel.average_waveform, this_channel.std_waveform, linestyle='None', capsize=0, capthick=0)
        
        axMax = 200
        axMin = -200
        #axarr[xpos, ypos].axis([0, 150, axMin, axMax])
        
        #axarr[xpos, ypos].text(90, -100, this_channel.channel_number,  fontsize='small')
        #axarr[xpos, ypos].text(150, axMin+10, round((activeChannelCounts[i]/recordingTime),2), fontsize='small')
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') 
        plt.tick_params(axis='y', which='both', left='off', right='off', labelbottom='off') 

    axarr[0,0].set_frame_on(False)
    axarr[0,7].set_frame_on(False)
    axarr[7,0].set_frame_on(False)
    axarr[7,7].set_frame_on(False)
    
    for i in range(0,8):
        plt.setp([a.get_xticklabels() for a in axarr[i, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[:, i]], visible=False)
        plt.setp([a.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') for a in axarr[:,i]])
        plt.setp([a.tick_params(axis='y', which='both', left='off', right='off', labelbottom='off') for a in axarr[:,i]])
    
    full_mea_plot_image_file = input_file.split('.')[0] + '_mea_plot_2.png'

    f.suptitle(input_file)
    f.savefig(full_mea_plot_image_file)


print("Importing Libraries...\n")
import os, time, math
import matplotlib.pyplot as plt 
import numpy as np
import neuroshare as ns
from Tkinter import Tk
from tkFileDialog import askdirectory

full_file_path = '-600 p31.mcd'
#full_file_path = 'spont.mcd'

channels_to_read = range(0, 100)
#channels_to_read = [47]
all_channels = get_data(full_file_path, channels_to_read)

#for channel in all_channels:
    # print(channel.over_threshold_number)
    # plt.plot(channel.voltage_data)
    # threshold_line = [channel.spike_threshold for x in channel.voltage_data]
    # plt.plot(threshold_line, color='r')
    # plt.title(str(channel.channel_number))
    # plt.show()


    # for spike in channel.all_spikes:
    #     plt.plot(spike.voltage_data)
    #     print(len(spike.voltage_data))
    # plt.title(channel.channel_number)
    # plt.show()

print("Generating full-MEA waveform plots")
plotMEAWaveforms(all_channels, full_file_path)