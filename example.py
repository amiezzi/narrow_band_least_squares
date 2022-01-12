############################################################################
################## Narrow-Band Least-Squares Method ########################
############################################################################
### Breaks up frequencies into multiple bands (similar to PMCC method) 
### Ordinary least-squares uses the entire frequency  band for calculation 
### Authors: Sneha Bhetanabhotla, Alex Iezzi, and Robin Matoza 
### University of California Santa Barbara 
### Contact: Alex Iezzi (amiezzi@ucsb.edu) 
### Last Modified: September 28, 2021 
############################################################################

###############
### Imports ###
###############
import os
from waveform_collection import gather_waveforms 
from obspy.core import UTCDateTime
import numpy as np
import math as math
from scipy import signal 
import matplotlib.pyplot as plt
from array_processing.algorithms.helpers import getrij
from lts_array import ltsva
from narrow_band_least_squares import narrow_band_least_squares
from helpers import get_freqlist, get_winlenlist, filter_data
from plotting import ordinary_filter_response_plot, ordinary_plot, narrow_band_processing_parameters_plot, narrow_band_plot, narrow_band_stau_plot, narrow_band_lts_plot, narrow_band_lts_dropped_station_plot



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
FREQ_BAND_TYPE = 'log'   # indicates linear or logarithmic spacing for frequency bands; 'linear' or 'log'
FILTER_TYPE = 'cheby1'      # filter type; 'butter', 'cheby1'
FILTER_ORDER = 2
FILTER_RIPPLE = 0.01


### Window Length ###
WINOVER = 0.5               # window overlap
WINDOW_LENGTH_TYPE = 'constant'  # 'constant' or 'adaptive'
WINLEN = 50                 # window length [s]; used if WINDOW_LENGTH_TYPE = 'constant' AND if WINDOW_LENGTH_TYPE = 'adaptive' (because of broadband processing)
WINLEN_1 = 60              # window length for band 1 (lowest frequency) [s]; only used if WINDOW_LENGTH_TYPE = 'adaptive'
WINLEN_X = 30               # window length for band X (highest frequency) [s]; only used if WINDOW_LENGTH_TYPE = 'adaptive'

### Array processing ###
ALPHA = 0.75                 # Use ordinary least-squares processing (not trimmed least-squares)
MDCCM_THRESH = 0.6          # Threshold value of MdCCM for plotting; Must be between 0 and 1

### Figure Save Options ###
file_type = '.png'                          # file save type
dpi_num = 300                               # dots per inch for plot save

 


##############################################################################
######################
### End User Input ###
######################
##############################################################################
if not os.path.exists('example_figures/'):
    os.makedirs('example_figures/')

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
### Ordinary Least-Squares ###
##############################

### Run ordinary least-squares ###
stf_broad, Fs, sos = filter_data(st, FILTER_TYPE, FMIN, FMAX, FILTER_ORDER, FILTER_RIPPLE)
vel_broad, baz_broad, t_broad, mdccm_broad, stdict_broad, sig_tau_broad = ltsva(stf_broad, rij, WINLEN, WINOVER, ALPHA)

### Plot ordinary array processing results ###
fig = ordinary_plot(stf_broad, vel_broad, baz_broad, mdccm_broad, t_broad, MDCCM_THRESH, ALPHA, stdict_broad, sig_tau_broad)
fig.savefig('example_figures/Ordinary_Least_Squares', dpi=dpi_num)
plt.close(fig)

### Plot ordinary filter frequency reponse ###
FMINL = math.log(0.01, 10)
FMAXL = math.log(Fs/2, 10)
freq_resp_list = np.logspace(FMINL, FMAXL, num = 1000)
w_broad, h_broad = signal.sosfreqz(sos,freq_resp_list,fs=Fs)
fig = ordinary_filter_response_plot(w_broad, h_broad, FMIN, FMAX, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE)
fig.savefig('example_figures/Filter_Frequency_Response_Broadband', dpi=dpi_num)
plt.close(fig)




##################################################################################
#################################
### Narrow-Band Least-Squares ###
#################################

### Set Up Narrow Frequency Bands ###                                      
freqlist, NBANDS, FMAX = get_freqlist(FMIN, FMAX, FREQ_BAND_TYPE, NBANDS)

### Set Up Window Lengths ###
WINLEN_list = get_winlenlist(WINDOW_LENGTH_TYPE, NBANDS, WINLEN, WINLEN_1, WINLEN_X)

### Run Narrow-Band Least-Squares ###
vel_array, baz_array, mdccm_array, t_array, stdict_all, sig_tau_array, num_compute_list, w_array, h_array = narrow_band_least_squares(WINLEN_list, WINOVER, ALPHA, st, rij, NBANDS, w_broad, h_broad, freqlist, FREQ_BAND_TYPE, freq_resp_list, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE)
#vel_array, baz_array, mdccm_array, t_array, stdict_all, sig_tau_array, num_compute_list, w_array, h_array = narrow_band_least_squares_parallel(WINLEN_list, WINOVER, ALPHA, st, rij, NBANDS, w_broad, h_broad, freqlist, FREQ_BAND_TYPE, freq_resp_list, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE)



### Plot narrow-band least-squares array processing results ###
fig = narrow_band_plot(FMIN, FMAX, stf_broad, NBANDS, freqlist, FREQ_BAND_TYPE, vel_array, baz_array, mdccm_array, t_array, num_compute_list, MDCCM_THRESH)
fig.savefig('example_figures/Narrow_Band_Least_Squares', dpi=dpi_num)
plt.close(fig)

if ALPHA == 1.0:
    ### Plot narrow-band least-squares array processing results ###
    fig = narrow_band_stau_plot(FMIN, FMAX, stf_broad, NBANDS, freqlist, FREQ_BAND_TYPE, vel_array, baz_array, mdccm_array, t_array, sig_tau_array, num_compute_list, MDCCM_THRESH, ALPHA)
    fig.savefig('example_figures/Narrow_Band_Least_Squares_Sigma_Tau', dpi=dpi_num)
    plt.close(fig)

elif ALPHA < 1.0:
    ### Plot narrow-band least-squares array processing results ###
    fig = narrow_band_lts_plot(FMIN, FMAX, stf_broad, NBANDS, freqlist, FREQ_BAND_TYPE, vel_array, baz_array, mdccm_array, t_array, stdict_all, num_compute_list, MDCCM_THRESH, ALPHA)
    fig.savefig('example_figures/Narrow_Band_Least_Squares_LTS', dpi=dpi_num)
    plt.close(fig)

    fig = narrow_band_lts_dropped_station_plot(FMIN, FMAX, stf_broad, NBANDS, freqlist, FREQ_BAND_TYPE, vel_array, baz_array, mdccm_array, t_array, stdict_all, num_compute_list, MDCCM_THRESH)
    fig.savefig('example_figures/Narrow_Band_Least_Squares_LTS_Dropped_Stations', dpi=dpi_num)
    plt.close(fig)



### Plot processing parameters ###
fig = narrow_band_processing_parameters_plot(rij, FREQ_BAND_TYPE, freqlist, WINLEN_list, NBANDS, FMIN, FMAX, w_array, h_array, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE)
fig.savefig('example_figures/Processing_Parameters', dpi=dpi_num)
plt.close(fig)




