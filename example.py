############################################################################
################## Narrow Band Least Squares Method ########################
############################################################################
### Breaks up frequencies into multiple bands (similar to PMCC method) 
### Normal least squares uses the entire frequency  band for calculation 
### Authors: Sneha Bhetanabhotla, Alex Iezzi, and Robin Matoza 
### University of California Santa Barbara 
### Contact: Alex Iezzi (amiezzi@ucsb.edu) 
### Last Modified: September 28, 2021 
############################################################################

###############
### Imports ###
###############
from waveform_collection import gather_waveforms 
from obspy.core import UTCDateTime, Stream, read
import numpy as np
import math as math
from scipy import signal 
import matplotlib.pyplot as plt
from narrow_band_least_squares import narrow_band_least_squares, narrow_band_least_squares_parallel
from helpers import get_freqlist, get_winlenlist, filter_data, write_txtfile, read_txtfile
from plotting import broad_filter_response_plot, processing_parameters_plot, narrow_band_plot, baz_freq_plot
from manuscript_plotting import array_processing_combined_plot
from array_processing.algorithms.helpers import getrij
from array_processing.tools.plotting import array_plot
from lts_array import ltsva


##############################################################################
##################
### User Input ###
##################

### Data information ###
# Data collection
SOURCE = 'IRIS'                                     # Data source; 'IRIS' or 'local'


# IRIS Example
NETWORK = 'IM'
STATION = 'I53H?'
LOCATION = '*'
CHANNEL = 'BDF'
START = UTCDateTime('2018-12-19T01:45:00')
END = START + 20*60



'''
# Bogoslof IRIS Example
NETWORK = 'AV'
STATION = 'DLL*'
LOCATION = '*'
CHANNEL = 'HD*'
START = UTCDateTime('2017-01-02T23:00:00')
END = START + 60*60
'''

'''
# KENI IRIS Example
NETWORK = 'AV'
STATION = 'KENI'
LOCATION = '*'
CHANNEL = 'HD*'
START = UTCDateTime('2021-09-30T07:55:00')
END = START + 10*60
'''
'''
# Pavlof 2014 IRIS Example
NETWORK = 'AV'
STATION = 'DLL'
LOCATION = '*'
CHANNEL = 'HD*'
START = UTCDateTime('2014-11-12T12:00:00')
END = UTCDateTime('2014-11-16T12:00:00')
'''
'''
# IRIS Clev
NETWORK = 'AV'
STATION = 'DLL'
LOCATION = '*'
CHANNEL = 'HD*'
START = UTCDateTime('2017-05-17T04:00:00')
END = UTCDateTime('2017-05-17T04:30:00')
'''

'''
# RIOE Local Example
START = UTCDateTime('2010-05-28T13:00:00')          # start time for processing (UTCDateTime)
#END = UTCDateTime('2010-05-29T00:00:00')            # end time for processing (UTCDateTime)
END = UTCDateTime('2010-05-28T14:30:00')            # end time for processing (UTCDateTime)
latlist = [-1.74812, -1.74749, -1.74906, -1.74805]
lonlist = [-78.62735, -78.62708, -78.62742, -78.62820]
calib = -0.000113  # Pa/count
data_dir = '/Users/aiezzi/Desktop/NSF_Postdoc/Array_Processing_Research/PMCC_Training/data/'       # directory where data is located
'''
'''
# CDWR Local Example
START = UTCDateTime('2005-03-09T01:00:00')          # start time for processing (UTCDateTime)
#END = UTCDateTime('2005-03-09T03:00:00')            # end time for processing (UTCDateTime)
END = START + 5400
# Coldwater Array MSH2*
rij = np.array([[-0.0082572, 0.042908, 0.015917, -0.050568], [0.00016988, 0.014625, -0.050269, 0.035474]])
calib = -1./4000. # CDWR data were recorded polarity reversed (hence -1)
data_dir = '/Users/aiezzi/Desktop/NSF_Postdoc/Array_Processing_Research/MSH_CDWR/MSH0309/SAC_DATA/'       # directory where data is located
'''



'''
# Iceland Local Example
START = UTCDateTime('2021-04-28T12:00:00')          # start time for processing (UTCDateTime)
END = UTCDateTime('2021-04-28T23:59:59')            # end time for processing (UTCDateTime)
day = '118'
latlist = [63.89614, 63.89676, 63.89631, 63.89639]
lonlist = [-22.27762, -22.27785, -22.27881, -22.27809]
#calib = 1  # Pa/count
calib = (15.259e-9)/(46e-6)  # Pa/count
#data_dir = '/Volumes/Seagate_HD/Iceland_Research_2021/fagradsfjall_data/data/MSEED/'       # directory where data is located
data_dir = '/Volumes/IEZZI_USB/Iceland_Research_2021/fagradsfjall_data/data/MSEED/'       # directory where data is located
'''


### Filtering ###
FMIN = 0.1                  # [Hz]
FMAX = 9.                   # [Hz] #should not exceed Nyquist
nbands = 8                # number of frequency bands 
freq_band_type = 'octave'   # indicates linear or logarithmic spacing for frequency bands; 'linear' or 'log'
filter_type = 'cheby1'      # filter type; 'butter', 'cheby1'
filter_order = 2
filter_ripple = 0.01


### Window Length ###
WINOVER = 0.5               # window overlap
window_length = 'constant'  # 'constant' or 'adaptive'
WINLEN = 50                 # window length [s]; used if window_length = 'constant' AND if window_length = 'adaptive' (because of broadband processing)
WINLEN_1 = 60              # window length for band 1 (lowest frequency) [s]; only used if window_length = 'adaptive'
WINLEN_X = 30               # window length for band X (highest frequency) [s]; only used if window_length = 'adaptive'

### Array processing ###
ALPHA = 1.0                 # Use ordinary least squares processing (not trimmed least squares)
mdccm_thresh = 0.6          # Threshold value of MdCCM for plotting; Must be between 0 and 1

### Figure Save Options ###
save_dir = '/Users/aiezzi/Desktop/NSF_Postdoc/Array_Processing_Research/narrow_band_least_squares/Figures/'         # directory in which to save figures

#save_dir = '/Volumes/IEZZI_USB/Iceland_Research_2021/Least_Squares_Figures/'
file_type = '.png'                          # file save type
dpi_num = 300                               # dots per inch for plot save

 


##############################################################################
######################
### End User Input ###
######################
##############################################################################


##############################################################################
###################
### Gather Data ###
###################
if SOURCE == 'IRIS':
    st = gather_waveforms(SOURCE, NETWORK, STATION, LOCATION, CHANNEL, START, END, remove_response=True)
    #st = Stream()
    #st += gather_waveforms(SOURCE, NETWORK, STATION, '01', CHANNEL, START, END, remove_response=True)
    #st += gather_waveforms(SOURCE, NETWORK, STATION, '02', CHANNEL, START, END, remove_response=True)
    #st += gather_waveforms(SOURCE, NETWORK, STATION, '04', CHANNEL, START, END, remove_response=True)
    #st += gather_waveforms(SOURCE, NETWORK, STATION, '05', CHANNEL, START, END, remove_response=True)
    #st += gather_waveforms(SOURCE, NETWORK, STATION, '06', CHANNEL, START, END, remove_response=True)
    latlist = [tr.stats.latitude for tr in st]
    lonlist = [tr.stats.longitude for tr in st]
elif SOURCE == 'local':
    # Read in waveforms 
    st = Stream()
    #st += read(data_dir + '*.mseed')
    #st += read(data_dir + '*.SAC')
    st += read(data_dir + 'GL.GL1..HDF.2021.' + day + '.00')
    st += read(data_dir + 'GL.GL2..HDF.2021.' + day + '.00')
    st += read(data_dir + 'GL.GL3..HDF.2021.' + day + '.00')
    st += read(data_dir + 'GL.GL4..HDF.2021.' + day + '.00')
    st.trim(START, END)
    st.merge()
    print(st)
    # Calibrate the data 
    for ii in range (len(st)):
        tr = st[ii]
        tr.data = tr.data*calib


### Array Geometry ###
# Convert array coordinates to array processing geometry
rij = getrij(latlist, lonlist)



##################################################################################
##############################
### Standard Least Squares ###
##############################

### Run standard least squares ###
stf_broad, Fs, sos = filter_data(st, filter_type, FMIN, FMAX, filter_order, filter_ripple)

# Least Squares
vel_broad, baz_broad, t_broad, mdccm_broad, stdict_broad, sig_tau_broad = ltsva(stf_broad, rij, WINLEN, WINOVER, ALPHA)
fig1, axs1 = array_plot(stf_broad, t_broad, mdccm_broad, vel_broad, baz_broad, ccmplot=True, mcthresh=mdccm_thresh, sigma_tau=sig_tau_broad)

#LTS
#ALPHA_LTS = 0.75
#vel_broad, baz_broad, t_broad, mdccm_broad, stdict_broad, sig_tau_broad = ltsva(stf_broad, rij, WINLEN, WINOVER, ALPHA_LTS)
#fig1, axs1 = array_plot(stf_broad, t_broad, mdccm_broad, vel_broad, baz_broad, ccmplot=True, mcthresh=mdccm_thresh, sigma_tau=None, stdict=stdict_broad)


### Plot standard array processing results ###
fig1.savefig(save_dir + 'LeastSquares', dpi=dpi_num)
plt.close(fig1)


### Plot standard filter frequency reponse ###
FMINL = math.log(0.01, 10)
FMAXL = math.log(Fs/2, 10)
freq_resp_list = np.logspace(FMINL, FMAXL, num = 1000)
w_broad, h_broad = signal.sosfreqz(sos,freq_resp_list,fs=Fs)

fig = broad_filter_response_plot(w_broad, h_broad, FMIN, FMAX, filter_type, filter_order, filter_ripple)
fig.savefig(save_dir + 'Filter_Frequency_Response_Broadband', dpi=dpi_num)
plt.close(fig)




##################################################################################
#################################
### Narrow Band Least Squares ###
#################################

### Set Up Narrow Frequency Bands ###                                      
freqlist, nbands, FMAX = get_freqlist(FMIN, FMAX, freq_band_type, nbands)

### Set Up Window Lengths ###
WINLEN_list = get_winlenlist(window_length, nbands, WINLEN, WINLEN_1, WINLEN_X)

### Run Narrow Band Least Squares ###

#vel_array, baz_array, mdccm_array, t_array, num_compute_list, w_array, h_array = narrow_band_least_squares(WINLEN_list, WINOVER, ALPHA, st, rij, nbands, w_broad, h_broad, freqlist, freq_band_type, freq_resp_list, filter_type, filter_order, filter_ripple)
vel_array, baz_array, mdccm_array, t_array, num_compute_list, w_array, h_array = narrow_band_least_squares_parallel(WINLEN_list, WINOVER, ALPHA, st, rij, nbands, w_broad, h_broad, freqlist, freq_band_type, freq_resp_list, filter_type, filter_order, filter_ripple)

print('Nbands:')
print(nbands)
print('Frequency List:')
print(freqlist)
print('Window Length List:')
print(WINLEN_list)



### Plot narrow band least squares array processing results ###
#if freq_band_type == '2_octave_over':
#    fig = narrow_band_overlap_plot(FMIN, FMAX, stf_broad, nbands, freqlist, vel_array, baz_array, mdccm_array, t_array, num_compute_list, mdccm_thresh)
#else:
fig = narrow_band_plot(FMIN, FMAX, stf_broad, nbands, freqlist, freq_band_type, vel_array, baz_array, mdccm_array, t_array, num_compute_list, mdccm_thresh)
fig.savefig(save_dir + 'LeastSquaresButPMCC', dpi=dpi_num)
plt.close(fig)


#fig = baz_freq_plot(FMIN, FMAX, nbands, freqlist, baz_array, mdccm_array, t_array, num_compute_list, mdccm_thresh)
#fig.savefig(save_dir + 'Long_term', dpi=dpi_num)
#plt.close(fig)


### Plot processing parameters ###
fig = processing_parameters_plot(rij, freq_band_type, freqlist, WINLEN_list, nbands, FMIN, FMAX, w_array, h_array, filter_type, filter_order, filter_ripple)
fig.savefig(save_dir + 'Processing_Parameters', dpi=dpi_num)
plt.close(fig)



### Write TXT File ###
#write_txtfile(save_dir, vel_array, baz_array, mdccm_array, t_array, freqlist, num_compute_list)
#stf_broad.write(save_dir + 'broad_filtered_data.mseed', format="MSEED")




'''
##############################
### Test using old results ###
##############################
### Read TXT file ###
vel_array_test, baz_array_test, mdccm_array_test, t_array_test, freqlist_test, num_compute_list_test, nbands_test, FMIN_test, FMAX_test = read_txtfile(save_dir)
#stf_test = read(save_dir + 'broad_filtered_data.mseed')


### Plot narrow band least squares array processing results ###
#fig = pmcc_like_plot(FMIN_test, FMAX_test, stf_test, nbands_test, freqlist, vel_array_test, baz_array_test, mdccm_array_test, t_array_test, num_compute_list_test, mdccm_thresh)
#fig.savefig(save_dir + 'LeastSquaresButPMCC_test', dpi=dpi_num)
fig = baz_freq_plot(FMIN_test, FMAX_test, nbands_test, freqlist_test, baz_array_test, mdccm_array_test, t_array_test, num_compute_list_test, mdccm_thresh)
fig.savefig(save_dir + 'Long_term_test', dpi=dpi_num)
plt.close(fig)
'''



'''
#####################
### Combined Plot ###
#####################
fig = array_processing_combined_plot(FMIN, FMAX, stf_broad, t_broad, mdccm_broad, vel_broad, baz_broad, sig_tau_broad, nbands, freqlist, vel_array, baz_array, mdccm_array, t_array, num_compute_list, mdccm_thresh)
fig.savefig(save_dir + 'Combined_Results', dpi=dpi_num)
plt.close(fig)
'''
