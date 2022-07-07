
from signal import signal
from this import s
from tkinter.colorchooser import Chooser
import numpy as np
import matplotlib.pyplot as plt
from PY import seq3000
from PY_sub_samp_poly import PY_SSDF
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
frame_length = 10*len(c_rrc)
no_signal_sample = len(c_rrc)

# generate the polynomial for estimating the sub sample timing estimation

Poly_coeef_for_sub_sample_timing_est= PY_SSDF()

# we assume that at each time instance a block of
#data is read from the hardware
# no_sample_chunk_read_from_adc variable is used to
# reflec this
no_sample_chunk_read_from_adc = 1000

# estimate how many chuncks should be read 

total_no_chunks = len(RX_signal) // no_sample_chunk_read_from_adc

""" The hardware implementation should 
continuously correlate the signal
however in the proof of concept impl
ementation in this code
we use the fact that the signal 
is transmitted periodically
to estimate the time of arrival. 
For this purpose we first 
estimate the first peak. From 
then on we jump into the 
approximate location of the 
next peak """

""" Here we could choose to estimate 
the CFO and correct it or not
For this implementation eventhough 
we use wired setup and SNR 
is high and the two clocks are only 
600-800 ppb apart we use CFO
correction  """

# CFO estimation and correction starts here

# the following is meant to do the autocorrelation and estimate the 
# CFO
corr_length = 120000
seq_length = 128
correlation = np.zeros(corr_length,dtype='complex128')
for i in np.arange(corr_length):
    print(i)
    correlation[i] = np.sum(np.conjugate(RX_signal[i:i+seq_length])*RX_signal[i+seq_length:i+2*seq_length])

""" plt.figure(3)
time = np.arange(len(correlation))/sample_rate/1e-3
plt.plot(time, np.abs(correlation))
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Autocorrelation based on 64 bit PN seq vs Time ')
plt.legend() """

# CFO estimation is done using the following
y_argmax= np.argmax(np.abs(correlation))
CFO = np.angle(correlation[y_argmax])/(2*np.pi*seq_length/sample_rate)

# The CFO estimate is used to remove frequency offset from the sugnal
time = np.arange(len(RX_signal))/6e6
RX_signal = RX_signal * np.exp(-1J*2*np.pi*CFO*time)
RX_signal = RX_signal.astype('complex128')
""" At this point we know that the remaining CFO 
does not degrade the 
correlation peak  """

""" The goal of the next few lines of code is 
to estimate the first peak of the correlation
in order to establish the approximate position 
of the signal we know that in 60000 
samples ther should be 6000 samples that are 
occupied by the signal so we take 
70000 signal samples or equivalently 70 chunks 
of data to find the first peak """

# The RX buffer is programmed based on the length of the 
# PN sequence 

no_Samp_in_Rx_buff = no_signal_sample + no_sample_chunk_read_from_adc
RX_buffer = np.zeros(no_Samp_in_Rx_buff, dtype="complex128") # this is the buffer of the received signal


# no of chunks is a convinient choice for the number of samples that 
# are expected to be processed
# As described above we originally choose 70 chunks
no_of_chunks = 70

# the correlation is done over all samples. In the following code
#we try to "emulate" hardware implementation of the correlator based
# +/- add (no multiplication)

corr_array = np.zeros(no_sample_chunk_read_from_adc*no_of_chunks, dtype="complex128")
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

""" plt.figure(1)
time = np.arange(len(corr_array))/sample_rate/1e-3
plt.plot(time, np.abs(corr_array),  label='Correlation value')
plt.grid(True)
plt.xlabel('Time (ms)')
plt.ylabel('Correlation')
plt.title('Correlation vs Time ')
plt.legend() """

# RX_signal = RX_signal[0:int(len(RX_signal)/10)]
II = np.argmax(np.abs(corr_array))
iseek = 0
No_Time_Measurement_Points = int(len(RX_signal)/(frame_length))

Time_at_receiver = np.zeros(No_Time_Measurement_Points, dtype='float64')
while iseek < len(RX_signal):
    index = II + iseek + np.arange(10000)-7000
    TX_buffer = RX_signal[index]
    no_of_chunks = 10
    RX_buffer = np.zeros(no_Samp_in_Rx_buff, dtype="complex128") # this is the buffer of the received signal
    corr_array = np.zeros(no_sample_chunk_read_from_adc*no_of_chunks, dtype="complex128")
    ii = 0
    for i in np.arange(no_of_chunks):      # 1000 chunks of data are transmitted
                                           # and received
    
        # The following sends TX signal from buffer and fills RX circular buffer
        RX_buffer_index = (np.arange(no_sample_chunk_read_from_adc) + ii )%no_Samp_in_Rx_buff
        RX_signal_index = (np.arange(no_sample_chunk_read_from_adc) + ii)
        RX_buffer[RX_buffer_index] = TX_buffer[RX_signal_index]
        
        for j in np.arange(no_sample_chunk_read_from_adc):
            pstv_index = (c_pstv + j + ii- no_signal_sample )%no_Samp_in_Rx_buff
            ngtv_index = (c_ngtv + j + ii- no_signal_sample)%no_Samp_in_Rx_buff
            corr_array[ii+j] = np.sum(RX_buffer[pstv_index]) - np.sum(RX_buffer[ngtv_index])
        ii = ii + no_sample_chunk_read_from_adc 
    max_index = np.argmax(np.abs(corr_array))
    prompt = corr_array[max_index]
    early = corr_array[max_index-1]
    late = corr_array[max_index+1]

    # here we calculate discrimiantion function
    disc = (np.abs(late)-np.abs(early))/np.abs(prompt)

    # in the prior section of this simulation code the polynomial for
    # correcting time estimate has been calculated 
    # Poly_coeef_for_sub_sample_timing_est(.) is that polynomial 
    # correction is the value of the timing correction in (ns)
    correction = Poly_coeef_for_sub_sample_timing_est(disc)*1e-9
    

    # the following is using the sample value of the time of arrival with 
    # sampling resolution (i.e. resolution of 166.66 ns)
    TOA_Est = (max_index+iseek)/sample_rate - correction 
    Time_at_receiver[int(iseek/frame_length)]= TOA_Est 
    
    print(TOA_Est)
    
    iseek = iseek + frame_length
   

plt.figure(1)
time = np.arange(len(Time_at_receiver))*10e-3
plt.plot(time, Time_at_receiver,  label='RX time')
plt.grid(True)
plt.xlabel('TX Time (s)')
plt.ylabel('RX Time (s)')
plt.title('RX time vs TX time')
plt.legend() 

L = len(Time_at_receiver)
Time_derivative = (-Time_at_receiver[0:L-1] + Time_at_receiver[1:L]) - 0.01
plt.figure(2)
time = np.arange(len(Time_derivative))*10e-3
plt.plot(time, Time_derivative*1e9,  label='T_offset change in 10ms')
plt.grid(True)
plt.xlabel('TX Time (s)')
plt.ylabel('Error(ns)')
plt.title('Time Offset Change in Receiver in 10 ms as a function of Time in Transmitter')
plt.legend()

window = 0.01*np.ones(100,dtype='float64')
smooth_derv = np.convolve(Time_derivative,window)
plt.figure(3)
time = np.arange(len(smooth_derv))*10e-3
plt.plot(time, smooth_derv*1e9,  label='Smooth T_offset change in 10ms')
plt.grid(True)
plt.xlabel('TX Time (s)')
plt.ylabel('Error(ns)')
plt.title('Moving average of Time Offset Change in Receiver in 10 ms as a function of Time in Transmitter')
plt.legend()
plt.show()




