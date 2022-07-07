
import numpy as np
import matplotlib.pyplot as plt
from PY import seq3000
from PY import seq64

import pickle
# Read recorded data from file 
pickle_in = open("Reorded_signal_SR_6Msps_freq_915MHz_PY3000_PY64repeated_duration_10s.bin","rb")          # open file for binary  reading
RX_signal = pickle.load(pickle_in)  # load file into variable



#remove DC from the recorded signal
RX_signal = RX_signal - np.mean(RX_signal)


#some recording related parametrs are set here. 
#these parameters are not saved as a meta data in the 
#recorded data 
center_freq = 915e6
sample_rate = 6e6

# generate the signal replica
c_rrc, c_pstv, c_ngtv =seq3000()


no_signal_sample = len(c_rrc)

# we assume that at each time instance a block of
#data is read from the hardware
# no_sample_chunk_read_from_adc variable is used to
# reflec this
no_sample_chunk_read_from_adc = 1000

# The RX buffer is programmed based on the length of the 
# PN sequence 

no_Samp_in_Rx_buff = no_signal_sample + no_sample_chunk_read_from_adc
RX_buffer = np.zeros(no_Samp_in_Rx_buff, dtype="complex64") # this is the buffer of the received signal

# no of chunks is a convinient choice for the number of samples that 
# are expected to be processed

no_of_chunks = 1000

# the correlation is done over all samples. In the following code
#we try to "emulate" hardware implementation of the correlator based
# +/- add (no multiplication)

corr_array = np.zeros(no_sample_chunk_read_from_adc*no_of_chunks, dtype="complex64")
ii = 0
for i in np.arange(no_of_chunks):             # 1000 chunks of data are transmitted
                                        # and received
    
    # The following sends TX signal from buffer and fills RX circular buffer
    print(i)
    RX_buffer_index = (np.arange(no_sample_chunk_read_from_adc) + ii )%no_Samp_in_Rx_buff
    RX_signal_index = (np.arange(no_sample_chunk_read_from_adc) + ii)
    RX_buffer[RX_buffer_index] = RX_signal[RX_signal_index]
    
    for j in np.arange(no_sample_chunk_read_from_adc):
        pstv_index = (c_pstv + j + ii- no_signal_sample )%no_Samp_in_Rx_buff
        ngtv_index = (c_ngtv + j + ii- no_signal_sample)%no_Samp_in_Rx_buff
        corr_array[ii+j] = np.sum(RX_buffer[pstv_index]) - np.sum(RX_buffer[ngtv_index])
    ii = ii + no_sample_chunk_read_from_adc 

plt.figure(1)
time = np.arange(len(corr_array))/sample_rate/1e-3
plt.plot(time, np.abs(corr_array),  label='Correlation value')
plt.grid(True)
plt.xlabel('Time (ms)')
plt.ylabel('Correlation')
plt.title('Correlation vs Time ')
plt.legend()

plt.figure(2)
time = np.arange(len(RX_signal[1000000:1400000]))/sample_rate/1e-3
plt.plot(time,np.real(RX_signal[1000000:1400000]), label='I')
plt.plot(time, np.imag(RX_signal[1000000:1400000]), label='Q')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Received Signal vs Time ')
plt.legend()

# the following is meant to do the autocorrelation and estimate the 
# CFO
corr_length = 120000
seq_length = 128
correlation = np.zeros(corr_length,dtype='complex64')
for i in np.arange(corr_length):
    print(i)
    correlation[i] = np.sum(np.conjugate(RX_signal[i:i+seq_length])*RX_signal[i+seq_length:i+2*seq_length])
plt.figure(3)
time = np.arange(len(correlation))/sample_rate/1e-3
plt.plot(time, np.abs(correlation))
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Autocorrelation based on 64 bit PN seq vs Time ')
plt.legend()

# CFO estimation is done using the following
y_argmax= np.argmax(np.abs(correlation))
CFO = np.angle(correlation[y_argmax])/(2*np.pi*seq_length/sample_rate)

# The CFO estimate is used to remove frequency offset from the sugnal
time = np.arange(len(RX_signal))/6e6
RX_signal = RX_signal * np.exp(-1J*2*np.pi*CFO*time)

# the following is the repeat of estimation the correlation 

RX_buffer = np.zeros(no_Samp_in_Rx_buff, dtype="complex64") # this is the buffer of the received signal
corr_array = np.zeros(no_sample_chunk_read_from_adc*no_of_chunks, dtype="complex64")
ii = 0
for i in np.arange(no_of_chunks):             # 1000 chunks of data are transmitted
                                        # and received
    
    # The following sends TX signal from buffer and fills RX circular buffer
    print(i)
    RX_buffer_index = (np.arange(no_sample_chunk_read_from_adc) + ii )%no_Samp_in_Rx_buff
    RX_signal_index = (np.arange(no_sample_chunk_read_from_adc) + ii)
    RX_buffer[RX_buffer_index] = RX_signal[RX_signal_index]
    
    for j in np.arange(no_sample_chunk_read_from_adc):
        pstv_index = (c_pstv + j + ii- no_signal_sample )%no_Samp_in_Rx_buff
        ngtv_index = (c_ngtv + j + ii- no_signal_sample)%no_Samp_in_Rx_buff
        corr_array[ii+j] = np.sum(RX_buffer[pstv_index]) - np.sum(RX_buffer[ngtv_index])
    ii = ii + no_sample_chunk_read_from_adc 

plt.figure(4)
time = np.arange(len(corr_array))/sample_rate/1e-3
plt.plot(time, np.abs(corr_array),  label='Correlation value')
plt.grid(True)
plt.xlabel('Time (ms)')
plt.ylabel('Correlation')
plt.title('Correlation after CFO correction vs Time ')
plt.legend()

plt.figure(5)
time = np.arange(len(RX_signal[1000000:1400000]))/sample_rate/1e-3
plt.plot(time,np.real(RX_signal[1000000:1400000]), label='I')
plt.plot(time, np.imag(RX_signal[1000000:1400000]), label='Q')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Received Signal After CFO coreraction vs Time ')
plt.legend()
plt.show()



