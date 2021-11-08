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
from array_processing.algorithms.helpers import getrij
from array_processing.tools.plotting import array_plot
from lts_array import ltsva


##############################################################################
##################
### User Input ###
##################

### Data information ###
# Data collection
# IRIS Example; Meteor in Alaska
SOURCE = 'IRIS'                                     
NETWORK = 'IM'
STATION = 'I53H?'
LOCATION = '*'
CHANNEL = 'BDF'
START = UTCDateTime('2018-12-19T01:45:00')
END = START + 20*60


### Filtering ###
FMIN = 0.1                  # [Hz]
FMAX = 5.                   # [Hz] #should not exceed Nyquist
NBANDS = 8                # number of frequency bands 
FREQ_BAND_TYPE = 'octave'   # indicates linear or logarithmic spacing for frequency bands; 'linear' or 'log'
FILTER_TYPE = 'cheby1'      # filter type; 'butter', 'cheby1'
FILTER_ORDER = 2
FILTER_RIPPLE = 0.01


### Window Length ###
WINOVER = 0.5               # window overlap
WINDOW_LENGTH_TYPE = 'adaptive'  # 'constant' or 'adaptive'
WINLEN = 50                 # window length [s]; used if WINDOW_LENGTH_TYPE = 'constant' AND if WINDOW_LENGTH_TYPE = 'adaptive' (because of broadband processing)
WINLEN_1 = 60              # window length for band 1 (lowest frequency) [s]; only used if WINDOW_LENGTH_TYPE = 'adaptive'
WINLEN_X = 30               # window length for band X (highest frequency) [s]; only used if WINDOW_LENGTH_TYPE = 'adaptive'

### Array processing ###
ALPHA = 1.0                 # Use ordinary least squares processing (not trimmed least squares)
MDCCM_THRESH = 0.6          # Threshold value of MdCCM for plotting; Must be between 0 and 1

### Figure Save Options ###
#save_dir = '/Users/aiezzi/Desktop/NSF_Postdoc/Array_Processing_Research/narrow_band_least_squares/Figures/'         # directory in which to save figures

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
st = gather_waveforms(SOURCE, NETWORK, STATION, LOCATION, CHANNEL, START, END, remove_response=True)
latlist = [tr.stats.latitude for tr in st]
lonlist = [tr.stats.longitude for tr in st]


### Array Geometry ###
# Convert array coordinates to array processing geometry
rij = getrij(latlist, lonlist)



##################################################################################
##############################
### Standard Least Squares ###
##############################

### Run standard least squares ###
stf_broad, Fs, sos = filter_data(st, FILTER_TYPE, FMIN, FMAX, FILTER_ORDER, FILTER_RIPPLE)


# Least Squares
vel_broad, baz_broad, t_broad, mdccm_broad, stdict_broad, sig_tau_broad = ltsva(stf_broad, rij, WINLEN, WINOVER, ALPHA)
fig1, axs1 = array_plot(stf_broad, t_broad, mdccm_broad, vel_broad, baz_broad, ccmplot=True, mcthresh=MDCCM_THRESH, sigma_tau=sig_tau_broad)


### Plot standard array processing results ###
fig1.savefig('example_figures/LeastSquares', dpi=dpi_num)
plt.close(fig1)


### Plot standard filter frequency reponse ###
FMINL = math.log(0.01, 10)
FMAXL = math.log(Fs/2, 10)
freq_resp_list = np.logspace(FMINL, FMAXL, num = 1000)
w_broad, h_broad = signal.sosfreqz(sos,freq_resp_list,fs=Fs)

fig = broad_filter_response_plot(w_broad, h_broad, FMIN, FMAX, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE)
fig.savefig('example_figures/Filter_Frequency_Response_Broadband', dpi=dpi_num)
plt.close(fig)




##################################################################################
#################################
### Narrow Band Least Squares ###
#################################

### Set Up Narrow Frequency Bands ###                                      
freqlist, NBANDS, FMAX = get_freqlist(FMIN, FMAX, FREQ_BAND_TYPE, NBANDS)

### Set Up Window Lengths ###
WINLEN_list = get_winlenlist(WINDOW_LENGTH_TYPE, NBANDS, WINLEN, WINLEN_1, WINLEN_X)

### Run Narrow Band Least Squares ###
vel_array, baz_array, mdccm_array, t_array, num_compute_list, w_array, h_array = narrow_band_least_squares(WINLEN_list, WINOVER, ALPHA, st, rij, NBANDS, w_broad, h_broad, freqlist, FREQ_BAND_TYPE, freq_resp_list, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE)
#vel_array, baz_array, mdccm_array, t_array, num_compute_list, w_array, h_array = narrow_band_least_squares_parallel(WINLEN_list, WINOVER, ALPHA, st, rij, NBANDS, w_broad, h_broad, freqlist, FREQ_BAND_TYPE, freq_resp_list, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE)



### Plot narrow band least squares array processing results ###
fig = narrow_band_plot(FMIN, FMAX, stf_broad, NBANDS, freqlist, FREQ_BAND_TYPE, vel_array, baz_array, mdccm_array, t_array, num_compute_list, MDCCM_THRESH)
fig.savefig('example_figures/NarrowLeastSquares', dpi=dpi_num)
plt.close(fig)



### Plot processing parameters ###
fig = processing_parameters_plot(rij, FREQ_BAND_TYPE, freqlist, WINLEN_list, NBANDS, FMIN, FMAX, w_array, h_array, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE)
fig.savefig('example_figures/Processing_Parameters', dpi=dpi_num)
plt.close(fig)




