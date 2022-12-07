####################################################################################
###################### Narrow-Band Least-Squares Method ############################
####################################################################################
### Breaks up the overall frequency limits into multiple sequential frequency bands
### Broadband least-squares uses the entire frequency  band for calculation
### Contact: Alex Iezzi (amiezzi@ucsb.edu)
####################################################################################

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
from narrow_band_least_squares import narrow_band_least_squares_parallel
from helpers import get_freqlist, get_winlenlist, filter_data
from plotting import broadband_filter_response_plot, broadband_plot, narrow_band_processing_parameters_plot, narrow_band_plot, narrow_band_stau_plot, narrow_band_lts_plot, narrow_band_lts_dropped_station_plot



##############################################################################
##################
### User Input ###
##################

### Data information ###
# Data collection
# IRIS Example; Meteor in Alaska
SOURCE = 'IRIS'                             # data source
NETWORK = 'IM'                              # network
STATION = 'I53H?'                           # station name
LOCATION = '*'                              # location code
CHANNEL = 'BDF'                             # channel code
START = UTCDateTime('2018-12-19T01:45:00')  # Start time; obspy UTCDateTime
END = START + 20*60                         # End time; obspy UTCDateTime

### Filtering ###
FMIN = 0.1                  # minimum frequency [Hz]
FMAX = 5.                   # maximum frequency [Hz]; should not exceed Nyquist
NBANDS = 8                  # number of frequency bands
FREQ_BAND_TYPE = 'log'      # indicates spacing for frequency bands; 'linear', 'log', 'octave', '2_octave_over', 'onethird_octave', 'octave_linear'
FILTER_TYPE = 'cheby1'      # filter type; 'butter', 'cheby1'
FILTER_ORDER = 2
FILTER_RIPPLE = 0.01

### Window Length ###
WINOVER = 0.5                   # window overlap
WINDOW_LENGTH_TYPE = 'adaptive' # window length type; 'constant' or 'adaptive'
WINLEN = 50                     # window length [s]; used if WINDOW_LENGTH_TYPE = 'constant' AND if WINDOW_LENGTH_TYPE = 'adaptive' (because of broadband processing)
WINLEN_1 = 60                   # window length for band 1 (lowest frequency) [s]; only used if WINDOW_LENGTH_TYPE = 'adaptive'
WINLEN_X = 30                   # window length for band X (highest frequency) [s]; only used if WINDOW_LENGTH_TYPE = 'adaptive'

### Array processing ###
ALPHA = 1.0                 # Use ordinary least-squares processing (not trimmed least-squares)
MDCCM_THRESH = 0.6          # Threshold value of MdCCM for plotting; Must be between 0 and 1

### Figure Save Options ###
file_type = '.png'                          # file save type
dpi_num = 300                               # dots per inch for plot save


##############################################################################
######################
### End User Input ###
######################
##############################################################################


# Make a directory for example figures
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
###############################
### Broadband Least-Squares ###
###############################

### Run broadband least-squares ###
stf_broad, Fs, sos = filter_data(st, FILTER_TYPE, FMIN, FMAX, FILTER_ORDER, FILTER_RIPPLE)
vel_broad, baz_broad, t_broad, mdccm_broad, stdict_broad, sig_tau_broad, vel_uncert_broad, baz_uncert_broad = ltsva(stf_broad, latlist, lonlist, WINLEN, WINOVER, ALPHA)

### Plot broadband array processing results ###
fig = broadband_plot(stf_broad, vel_broad, baz_broad, mdccm_broad, t_broad, MDCCM_THRESH, ALPHA, stdict_broad, sig_tau_broad)
fig.savefig('example_figures/Broadband_Least_Squares', dpi=dpi_num)
plt.close(fig)

### Plot broadband filter frequency reponse ###
FMINL = math.log(0.01, 10)
FMAXL = math.log(Fs/2, 10)
freq_resp_list = np.logspace(FMINL, FMAXL, num = 1000)
w_broad, h_broad = signal.sosfreqz(sos,freq_resp_list,fs=Fs)
fig = broadband_filter_response_plot(w_broad, h_broad, FMIN, FMAX, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE)
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
#vel_array, baz_array, mdccm_array, t_array, stdict_all, sig_tau_array, num_compute_list, w_array, h_array = narrow_band_least_squares(WINLEN_list, WINOVER, ALPHA, st, latlist, lonlist, NBANDS, w_broad, h_broad, freqlist, FREQ_BAND_TYPE, freq_resp_list, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE)
vel_array, baz_array, mdccm_array, t_array, stdict_all, sig_tau_array, num_compute_list, w_array, h_array = narrow_band_least_squares_parallel(WINLEN_list, WINOVER, ALPHA, st, latlist, lonlist, NBANDS, w_broad, h_broad, freqlist, FREQ_BAND_TYPE, freq_resp_list, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE)


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



### Plot narrow-band least-squares processing parameters ###
fig = narrow_band_processing_parameters_plot(rij, FREQ_BAND_TYPE, freqlist, WINLEN_list, NBANDS, FMIN, FMAX, w_array, h_array, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE)
fig.savefig('example_figures/Narrow_Band_Processing_Parameters', dpi=dpi_num)
plt.close(fig)




