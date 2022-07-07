import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from PY import seq3000
def PY_SSDF():
    f_s = 6e6
    up_samp_factor = 1000
    mdl_fs = up_samp_factor*f_s
    c_rrc, c_pstv, c_ngtv =seq3000() 

    # Add zeros at the end for calculating correlation  
    b =[1,0]
    c_rrc_time_template = np.kron(b,c_rrc)
    c_rrc_time_template_up = signal.resample(c_rrc_time_template,up_samp_factor*len(c_rrc_time_template))
    c_rrc_time_template_sign = np.sign(c_rrc_time_template)


    length = np.shape(c_rrc_time_template_up)[0]

    t_corr_time_up = (np.arange(length)) * 1/(f_s*up_samp_factor) *1e9
    c_rrc_time_template_ds = c_rrc_time_template_up[0:-1:int(up_samp_factor)]
    length = np.shape(c_rrc_time_template_ds)[0]
    t_corr_time_ds = (np.arange(length)) * 1/(f_s) *1e9; 

    """ plt.figure(1)
    plt.plot(t_corr_time_up, np.real(c_rrc_time_template_up), label ='Fs=6GHz')
    plt.plot(t_corr_time_ds, np.real(c_rrc_time_template_ds), label ='Fs=6MHz from 6GHz sig') 
    plt.plot(t_corr_time_ds, np.real(c_rrc_time_template), label ='Fs=6MHz original') 
    plt.grid(True)
    plt.xlabel('time (ns)')
    plt.ylabel('Amplitude Real')
    plt.title('Real Value of ZaiNar PY Signal')
    plt.legend()

    plt.figure(2)
    plt.plot(t_corr_time_up, np.imag(c_rrc_time_template_up), label ='Fs=6GHz')
    plt.plot(t_corr_time_ds, np.imag(c_rrc_time_template_ds), label ='Fs=6MHz from 6GHz sig') 
    plt.plot(t_corr_time_ds, np.imag(c_rrc_time_template), label ='Fs=6MHz original') 
    plt.grid(True)
    plt.xlabel('time (ns)')
    plt.ylabel('Amplitude Imag')
    plt.title('Imag Value of ZaiNar PY Signal')
    plt.legend() """

    c_rrc_freq_template = np.fft.fftshift(np.abs(np.fft.fft(c_rrc_time_template)))
    f_tt = np.linspace(-3e6,3e6,len(c_rrc_time_template),endpoint=False)
    c_rrc_freq_template_ds = np.fft.fftshift(np.abs(np.fft.fft(c_rrc_time_template_ds)))
    f_ds = np.linspace(-3e6,3e6,len(c_rrc_freq_template_ds),endpoint=False)

    """ plt.figure(3)
    plt.plot(f_tt/1e6, 20*np.log10(np.abs(c_rrc_freq_template)/np.max(np.abs(c_rrc_freq_template))), label ='Fs=6GHz')
    plt.plot(f_ds/1e6, 20*np.log10(np.abs(c_rrc_freq_template_ds)/np.max(np.abs(c_rrc_freq_template))), label ='Fs=6MHz')
    plt.grid(True)
    plt.xlabel('Freq (MHz)')
    plt.ylabel('FFT amplitude (dB)')
    plt.title('FFT amplitude of ZaiNar PY Signal for template and DS template')
    plt.legend() """



    sig_corr = signal.correlate(c_rrc_time_template, c_rrc_time_template_sign)
    length = np.shape(sig_corr)[0]
    t_corr = (np.arange(length)-length/2+1/2) * 1/(f_s) *1e9; 
    sig_corr_up = signal.resample(sig_corr,up_samp_factor*len(sig_corr))
    length = len(sig_corr_up)
    t_corr_up = (np.arange(length)-length/2+1/2*up_samp_factor) * 1/(up_samp_factor*f_s) *1e9

    """ plt.figure(4)
    plt.plot(t_corr, np.abs(sig_corr),'ro', label ='6MHz sampling rate')
    plt.plot(t_corr_up, np.abs(sig_corr_up), label ='6GHz sampling rate') 
    plt.grid(True)
    plt.xlabel('time (ns)')
    plt.ylabel('correlation value')
    plt.title('PY correlation Peak')
    plt.legend() """


    index = np.argmax(np.abs(sig_corr_up))
    p_index_min = index-int(up_samp_factor/2)   
    p_index_max = index+int(up_samp_factor/2)
    prompt = sig_corr_up[p_index_min:p_index_max]
    early = sig_corr_up[p_index_min-up_samp_factor:p_index_max-up_samp_factor]
    late = sig_corr_up[p_index_min+up_samp_factor:p_index_max+up_samp_factor]

    # the following is used if we want to see early/prompt/late signals
    length = np.shape(prompt)[0]
    t = (np.arange(length)-length/2) * 1/(f_s*up_samp_factor) *1e9; 
    """ plt.figure(5)
    plt.plot(t, np.abs(prompt), label= 'prompt')
    plt.plot(t, np.abs(early), label= 'early' ) 
    plt.plot( t, np.abs(late), label = 'late')
    plt.grid(True)
    plt.xlabel('time (ns)')
    plt.ylabel('correlation value')
    plt.title('Early/Prompt/late Near correlation Peak')
    plt.legend() """


    length = np.shape(prompt)[0]
    t = (np.arange(length)-length/2) * 1/(f_s*up_samp_factor) *1e9; 
    disc = (np.abs(late)-np.abs(early))/np.abs(prompt)
    z = np.polyfit(disc,t,3)
    p = np.poly1d(z)
    """ plt.figure(6)
    plt.grid(True)
    plt.xlabel('discrimination function value')
    plt.ylabel('time adjustment (ns)')
    plt.title('discrimination function')
    plt.plot(disc, t, label='true value')
    plt.plot(disc, p(disc),  label='polynomial fit')
    plt.legend()

    plt.figure(7)
    plt.plot(disc, t-p(disc) )
    plt.grid(True)
    plt.xlabel('discrimination function value')
    plt.ylabel('error of time adjustment (ns)')
    plt.title('error of time adjustiment using polyfit vs true value')
    plt.show() """

    """ 
    At this point we are ready to use the discrimination function to
    estimate time of arrival for this purpose we only use the function
    p(disc) 
    """
    return p
# del length, t_corr_time_up, c_rrc_time_template_ds, t_corr_time_ds, \
#   sig_corr, t_corr, sig_corr_up, t_corr_up, index, p_index_min,\
#   p_index_max, prompt, early, late, t, disc 
if __name__=='__main__':
    f_s = 6e6
    up_samp_factor = 1000
    mdl_fs = up_samp_factor*f_s
    c_rrc, c_pstv, c_ngtv =seq3000() 

    # Add zeros at the end for calculating correlation  
    b =[1,0]
    c_rrc_time_template = np.kron(b,c_rrc)
    c_rrc_time_template_up = signal.resample(c_rrc_time_template,up_samp_factor*len(c_rrc_time_template))
    c_rrc_time_template = np.sign(c_rrc_time_template)
    N_sim = 100   # number of sample point to estimat the Time of Arrival
    max_range = 10000 # samples @6GHz = 1.66 us = 500m
    SNR = np.array([ 20., 15., 10.,5.,0.,-5.,-10.])
    Range_error = .0*SNR
    p = PY_SSDF() 
    for j in np.arange(len(SNR)):
        snr = SNR[j]
        for i in np.arange(N_sim):
            ##################################################################
            #
            # This portion is  simulating delay and noise for the RX signal
            delay = np.random.randint(0,max_range)
            rx_sig = np.roll(c_rrc_time_template_up, delay)
            rx_sig = rx_sig[0:-1:int(up_samp_factor)]
            length = len(rx_sig)
            sig_power = np.sum(np.square(np.abs(rx_sig)))/(length/2)
            noise_power = sig_power/(np.power(10, snr/10)) 
            noise_sigma = np.sqrt(noise_power)
            rx_noise = np.random.normal(0, noise_sigma, length)
            rx_sig = rx_sig + rx_noise

            ##################################################################
            #
            # MR is calculating the correlation and giving the index of the 
            # correlation peak
            sig_corr = signal.correlate(rx_sig, c_rrc_time_template)
            index = np.argmax(np.abs(sig_corr))
            
            # the folloing is getting the value for the the peak (prompt) 
            # early and late
            prompt = sig_corr[index]
            early = sig_corr[index-1]
            late = sig_corr[index+1]

            # here we calculate discrimiantion function
            disc = (np.abs(late)-np.abs(early))/np.abs(prompt)

            # in the prior section of this simulation code the polynomial for
            # correcting time estimate has been calculated p(.) is that polynomial 
            # correction is the value of the timing correction in (ns)
            correction = p(disc) 

            # the following is using the sample value of the time of arrival with 
            # sampling resolution (i.e. resolution of 166.66 ns)
            TOA_Est = (index-((len(sig_corr)+1)/2-1))*1/f_s 
            # here we use the correction to get sub-sample resolution
            TOA_Est= TOA_Est - correction*1e-9
            # the ground truth is set with the resolution of 6GHz sampling rate or
            # 0.166 ns
            TOA_GT = delay/(f_s*up_samp_factor)
            # the error is calculated based on these values
            error = TOA_Est - TOA_GT 
            # here we prepare the Range_error vector for plotting purposes
            Range_error[j] = Range_error[j] + np.power(error*1e9, 2)
            print(snr, TOA_Est, TOA_GT, error, correction, disc)
    plt.figure(15)
    plt.semilogy(SNR, (np.sqrt(Range_error/N_sim)), label= 'Range Error')
    plt.semilogy(SNR, (np.sqrt(Range_error/N_sim)), 'ro', label= 'Range Error')
    plt.semilogy(SNR, 1e9/(2*6e6*np.pi*np.sqrt(np.power(10, (SNR+37.78)/10))), 'go', label= 'Theoretical limit ML estimate')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('TOA estmatin error  (ns)')
    plt.title('Time of arrival estimation error vs SNR')
    plt.legend()
    plt.show()
