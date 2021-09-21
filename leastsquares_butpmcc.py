############################################################################
### Narrow Band Least Squares Method #######################################
### Breaks up frequencies into multiple bands (similar to PMCC method) #####
### Normal least squares uses the entire frequency  band for calculation ###
### Authors: Sneha Bhetanabhotla, Alex Iezzi, and Robin Matoza #############
############################################################################

###############
### Imports ###
###############
from waveform_collection import gather_waveforms #Commented out by Sneha 
# Added by alex
#import sys
#sys.path.append('/Users/aiezzi/repos/array_processing/')
from obspy.core import UTCDateTime, Stream, read
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import math as math
import pylab as pl
from matplotlib.patches import Rectangle
import matplotlib.colorbar as cbar
from matplotlib import cm
from matplotlib import rcParams
from array_processing.algorithms.helpers import getrij
#from array_processing.tools.plotting import array_plot
from lts_array import ltsva
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import signal # For Chebychev filter


##############################################################################
##################
### User Input ###
##################

### Data information ###
# Data collection
SOURCE = 'local'                                     # Data source; 'IRIS' or 'local'

'''
# IRIS Example
NETWORK = 'IM'
STATION = 'I53H?'
LOCATION = '*'
CHANNEL = 'BDF'
START = UTCDateTime('2018-12-19T01:45:00')
END = START + 20*60
'''

# Local Example
START = UTCDateTime('2010-05-28T13:30:00')          # start time for processing (UTCDateTime)
END = UTCDateTime('2010-05-28T15:30:00')            # end time for processing (UTCDateTime)
FMT = '%Y-%m-%dT%H:%M:%S.%f'                        # date/time format 

# RIOE coordinates (make sure this is not hardcoded)
latlist = [-1.74812, -1.74749, -1.74906, -1.74805]
lonlist = [-78.62735, -78.62708, -78.62742, -78.62820]

#data_dir = '/Users/snehabhetanabhotla/Desktop/Research/data/'                                      # directory where data is located
#data_dir = '/Users/aiezzi/Desktop/NSF_Postdoc/Array_Processing_Research/PMCC_Training/data/'       # directory where data is located
data_dir = '/Users/ACGL/Desktop/NSF_Postdoc/Array_Processing_Research/PMCC_Training/data/'          # directory where data is located


### Filtering ###
FMIN = 0.1                  # [Hz]
FMAX = 10                   # [Hz] #should not exceed 20 Hz 
nbands = 10                 # indicates number of frequency bands 
freq_band_type = 'linear'   # indicates linear or logarithmic spacing for frequency bands; 'linear' or 'log'
filter_type = 'butter'      # filter type; 'butter', 'cheby'


### Array processing ###
WINLEN = 50                 # window length [s]
WINOVER = 0.5               # window overlap
ALPHA = 1.0                 # Use ordinary least squares processing (not trimmed least squares)


### Figure Save Options ###
#save_dir = '/Users/snehabhetanabhotla/Desktop/Research/Plots/LeastSquaresCode/LeastSquaresButPMCC/'			     # directory in which to save figures
#save_dir = '/Users/aiezzi/Desktop/NSF_Postdoc/Array_Processing_Research/narrow_band_least_squares/Figures/'         # directory in which to save figures
save_dir = '/Users/ACGL/Desktop/NSF_Postdoc/Array_Processing_Research/narrow_band_least_squares/Figures/'            # directory in which to save figures
file_type = '.png'                          # file save type
dpi_num = 300                               # dots per inch for plot save
fonts = 14                                  # default font size for plotting
rcParams.update({'font.size': fonts})



##############################################################################
######################
### End User Input ###
######################


##############################################################################
###################
### Gather Data ###
###################
if SOURCE == 'IRIS':
    st = gather_waveforms(SOURCE, NETWORK, STATION, LOCATION, CHANNEL, START, END, remove_response=True)
    latlist = [tr.stats.latitude for tr in st]
    lonlist = [tr.stats.longitude for tr in st]

# This is hardcoded for RIOE...will have to fix!
elif SOURCE == 'local':
    # Read in waveforms 
    st = Stream()
    st += read(data_dir + '20100528.RIOE1.BDF.mseed')
    st += read(data_dir + '20100528.RIOE2.BDF.mseed')
    st += read(data_dir + '20100528.RIOE3.BDF.mseed')
    st += read(data_dir + '20100528.RIOE4.BDF.mseed')
    st.trim(START, END)

    #make this a for loop at some point 
    tr1 = st[0]
    tr2 = st[1]
    tr3 = st[2]
    tr4 = st[3]

    # Calibrate the data (RIOE)
    calib = -0.000113  # Pa/count
    tr1.data = tr1.data*calib
    tr2.data = tr2.data*calib
    tr3.data = tr3.data*calib
    tr4.data = tr4.data*calib

# Time vector for plotting
timevec = st[0].times('matplotlib')

#Default plot to sanity check 
#can also do print(st)
#st.plot() 


##################################################################################
###########################
### Set Up Narrow Bands ###
###########################										

freqrange = FMAX - FMIN

if freq_band_type == 'linear':
    freqinterval = freqrange / nbands
    freqlist = np.arange(FMIN, FMAX+freqinterval, freqinterval)
elif freq_band_type == 'log':
    FMINL = math.log(FMIN, 10)
    FMAXL = math.log(FMAX, 10)
    freqlist = np.logspace(FMINL, FMAXL, num = nbands+1)


##################################################################################
######################
### Array Geometry ###
######################

# Convert array coordinates to array processing geometry
rij = getrij(latlist, lonlist)

### Plot array geometry ###
fig = plt.figure(figsize=(5,5), dpi=dpi_num)
gs = gridspec.GridSpec(1,1)

ax0 = plt.subplot(gs[0,0]) 
ax0.scatter(rij[0], rij[1]) 
ax0.set_xlabel('X [km]', fontsize=fonts+2, fontweight='bold')
ax0.set_ylabel('Y [km]', fontsize=fonts+2, fontweight='bold')
ax0.axis('square')
ax0.grid()

### Save figure ###
plt.tight_layout()
fig.savefig(save_dir + 'Array_Geometry', dpi=dpi_num)
plt.close(fig)




#################################
### Broadband Bandpass filter ###
#################################
stf_broad = st.copy()
if filter_type == 'butter': 
        stf_broad.filter('bandpass', freqmin = FMIN, freqmax = FMAX, corners=2, zerophase = True)
elif filter_type == 'cheby': ### DONT USE YET; NEEDS TO BE FIXED
    order = 2
    ripple = 0.01
    Fs = tempst_filter[0].stats.sampling_rate
    Wn = [tempfmin/(Fs/2), tempfmax/(Fs/2)]
    #b,a = signal.cheby1(order, ripple, Wn, 'bandpass', fs=Fs)
    sos = signal.cheby1(order, ripple, Wn, 'bandpass', fs=Fs, output='sos')
    for ii in range(len(st)):
        # put signal in numpy array
        temp_array = stf_broad[ii].data
        # Filter
        filtered = signal.sosfilt(sos, temp_array)
        #filtered =signal.filtfilt(b,a,temp_array)
        # transform signal back to st
        stf_broad[ii].data = filtered

stf_broad.taper(max_percentage=0.01)    # Taper the waveforms


##################################################################################
###############################
### Initialize Numpy Arrays ###
###############################
#filteredst = [] #list of filtered streams 
#vellist = [] #velocity list 
#bazlist = [] #back azimuth list
#tlist = [] #time? 
#mdccmlist = [] #mdccm 
#stdictlist = [] #?
#sig_taulist = [] #? 

#filteredst = np.empty(shape=(nbands, st[0].stats.npts), dtype=float)



############################
### Run Array Processing ###
############################
for ii in range(nbands): 
#for x in range(nbands - 1): 
#for x in range(1):
    tempst_filter = st.copy()
    tempfmin = freqlist[ii]
    tempfmax = freqlist[ii+1]
    if filter_type == 'butter': 
        tempst_filter.filter('bandpass', freqmin = tempfmin, freqmax = tempfmax, corners=2, zerophase = True)
    elif filter_type == 'cheby': ### DONT USE YET; NEEDS TO BE FIXED
        order = 2
        ripple = 0.01
        Fs = tempst_filter[0].stats.sampling_rate
        Wn = [tempfmin/(Fs/2), tempfmax/(Fs/2)]
        #b,a = signal.cheby1(order, ripple, Wn, 'bandpass', fs=Fs)
        sos = signal.cheby1(order, ripple, Wn, 'bandpass', fs=Fs, output='sos')
        for jj in range(len(st)):
            # put signal in numpy array
            temp_array = tempst_filter[jj].data
            # Filter
            filtered = signal.sosfilt(sos, temp_array)
            #filtered =signal.filtfilt(b,a,temp_array)
            # transform signal back to st
            tempst_filter[jj].data = filtered

    tempst_filter.taper(max_percentage=0.01)    # Taper the waveforms
    #filteredst.append(tempst_filter)
    #filteredst[ii][:]=tempst_filter

    # Run Array Processing 
    vel, baz, t, mdccm, stdict, sig_tau = ltsva(tempst_filter, rij, WINLEN, WINOVER, ALPHA)

    # Convert array processing output to numpy array of floats
    vel_float = []
    for jj in range(len(vel)):
        vel_float.append(float(vel[jj]))
    vel_float = np.array(vel_float)

    baz_float = []
    for jj in range(len(baz)):
        baz_float.append(float(baz[jj]))
    baz_float = np.array(baz_float)

    mdccm_float = []
    for jj in range(len(mdccm)):
        mdccm_float.append(float(mdccm[jj]))
    mdccm_float = np.array(mdccm_float)

    t_float = []
    for jj in range(len(t)):
        t_float.append(float(t[jj]))
    t_float = np.array(t_float)

    ### Save Array Processing Output ###
    if ii == 0:
        vel_array = vel_float
        baz_array = baz_float
        mdccm_array = mdccm_float
        t_array = t_float
    else: 
        vel_array = np.vstack([vel_array, vel_float])
        baz_array = np.vstack([baz_array, baz_float])
        mdccm_array = np.vstack([mdccm_array, mdccm_float])
        t_array = np.vstack([t_array, t_float])



##################################################################################
######################
### PMCC like plot ###
######################
#fig, axs = plt.subplots(2,1)
#fig.set_size_inches(10, 5)
fig = plt.figure(figsize=(15,13), dpi=dpi_num)
gs = gridspec.GridSpec(4,2, width_ratios=[3,0.1], height_ratios=[1,1,1,1])

#cm = 'RdYlBu_r'
cm = 'jet'
cax = (FMIN, FMAX)



# Pressure plot
ax0 = plt.subplot(gs[0,0])  # Pressure Plot
ax0.plot(timevec, stf_broad[0], 'k') # plots pressure for the first band


ax1 = plt.subplot(gs[1,0])  # Backazimuth Plot
ax2 = plt.subplot(gs[2,0])  # Trace Velocity Plot
ax3 = plt.subplot(gs[3,0])  # Scatter Plot


for ii in range(nbands):

    # Frequency band info
    tempfmin = freqlist[ii]
    tempfmax = freqlist[ii+1]
    height_temp = tempfmax - tempfmin # height of frequency rectangles
    tempfavg = tempfmin + (height_temp/2)        # center point of the frequency interval

    # Gather array processing results for this narrow frequency band
    vel_float = vel_array[ii,:]
    baz_float = baz_array[ii,:]
    mdccm_float = mdccm_array[ii,:]
    t_float = t_array[ii,:]

    # Initialize colorbars
    #normal = pl.Normalize(baz_float.min(), baz_float.max())
    normal_baz = pl.Normalize(0, 360)
    colors_baz = pl.cm.jet(normal_baz(baz_float))

    #normal_vel = pl.Normalize(vel_float.min(), vel_float.max()) # This re-normalizing for each narrow frequency band (not right)
    normal_vel = pl.Normalize(0.2,0.5)
    colors_vel = pl.cm.jet(normal_vel(vel_float))



    for jj in range(len(t_float)-1):
        width_temp = t_float[jj+1] - t_float[jj]
        if mdccm_float[jj] >= 0.6: 
        #if np.greater(mdccmlist[ii], 0.6):
            x_temp = t_float[jj]
            y_temp = tempfmin

            # Backazimuth plot 
            rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_baz[jj])
            ax1.add_patch(rect)

            # Trace Velocity Plot 
            rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_vel[jj])
            ax2.add_patch(rect)

            # Scatter plot
            sc = ax3.scatter(t_float[jj], baz_float[jj], c=tempfavg, edgecolors='k', lw=0.3, cmap=cm)
            sc.set_clim(cax)


   
# Pressure plot 
ax0.xaxis_date()
ax0.set_xlim(timevec[0], timevec[-1])
ax0.set_ylabel('Pressure [Pa]', fontsize=fonts+2, fontweight='bold') 
ax0.set_xlabel('Time', fontsize=fonts+2, fontweight='bold') 
ax0.set_title('a)', loc='left', fontsize=fonts+2, fontweight='bold')

# Backazimuth Plot
ax1.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
ax1.set_xlabel('Time', fontsize=fonts+2, fontweight='bold') 
ax1.set_title('b)', loc='left', fontsize=fonts+2, fontweight='bold')
ax1.xaxis_date()
ax1.set_ylim(FMIN,FMAX)
ax1.set_xlim(t_float[0],t_float[-1])

# Trace Velocity Plot
ax2.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
ax2.set_xlabel('Time', fontsize=fonts+2, fontweight='bold') 
ax2.set_title('c)', loc='left', fontsize=fonts+2, fontweight='bold')
ax2.xaxis_date()
ax2.set_ylim(FMIN,FMAX)
ax2.set_xlim(t_float[0],t_float[-1])

# Scatter Plot
ax3.set_ylabel('Backazimuth [deg]', fontsize=fonts+2, fontweight='bold')  
ax3.set_xlabel('Time', fontsize=fonts+2, fontweight='bold') 
ax3.set_title('d)', loc='left', fontsize=fonts+2, fontweight='bold')
ax3.xaxis_date()
ax3.set_ylim(0,360)
ax3.set_xlim(t_float[0],t_float[-1])



# Add colorbar to backazimuth plot
cax = plt.subplot(gs[1,1]) 
cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.jet,norm=normal_baz,orientation='vertical', ticks=[0,90,180,270,360]) 
cax.set_ylabel('Backazimuth [deg]', fontsize=fonts+2, fontweight='bold')

# Add colorbar to trace velocity plot
cax = plt.subplot(gs[2,1]) 
cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.jet,norm=normal_vel,orientation='vertical') 
cax.set_ylabel('Trace Velocity [km/s]', fontsize=fonts+2, fontweight='bold')

# Add colorbar to scatter plot
cbaxes = plt.subplot(gs[3,1]) 
hc = plt.colorbar(sc, cax=cbaxes,orientation='vertical') 
cbaxes.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')


###################
### Save figure ###
###################
plt.tight_layout()
fig.savefig(save_dir + 'LeastSquaresButPMCC', dpi=dpi_num)
plt.close(fig)





#height of rectangles goes from tempfmin and tempfmax, width is from tlist[x] to tlist[x+1]
#each frequency band graph has subplots of baz, v, mdccm, sig tau
#plotting goes inside for loop
#see bulletin_updated for colorbar: backaz from 0 - 360, trace v from 0.25 - 0.45 km/s, goes outside for loop

#%% Array processing and plotting using least squares

#%% Array processing. ALPHA = 1.0: least squares processing.

#hard coded scatter plots for all the lists
""" fig1, axs1 = array_plot(filteredst[0], tlist[0], mdccmlist[0], vellist[0], bazlist[0], ccmplot=True, mcthresh=0.6, sigma_tau=sig_taulist[0])
fig2, axs2 = array_plot(filteredst[1], tlist[1], mdccmlist[1], vellist[1], bazlist[1], ccmplot=True, mcthresh=0.6, sigma_tau=sig_taulist[1])
fig3, axs3 = array_plot(filteredst[2], tlist[2], mdccmlist[2], vellist[2], bazlist[2], ccmplot=True, mcthresh=0.6, sigma_tau=sig_taulist[2])
fig4, axs4 = array_plot(filteredst[3], tlist[3], mdccmlist[3], vellist[3], bazlist[3], ccmplot=True, mcthresh=0.6, sigma_tau=sig_taulist[3])
fig5, axs5 = array_plot(filteredst[4], tlist[4], mdccmlist[4], vellist[4], bazlist[4], ccmplot=True, mcthresh=0.6, sigma_tau=sig_taulist[4])
fig6, axs6 = array_plot(filteredst[5], tlist[5], mdccmlist[5], vellist[5], bazlist[5], ccmplot=True, mcthresh=0.6, sigma_tau=sig_taulist[5])
fig7, axs7 = array_plot(filteredst[6], tlist[6], mdccmlist[6], vellist[6], bazlist[6], ccmplot=True, mcthresh=0.6, sigma_tau=sig_taulist[6])
fig8, axs8 = array_plot(filteredst[7], tlist[7], mdccmlist[7], vellist[7], bazlist[7], ccmplot=True, mcthresh=0.6, sigma_tau=sig_taulist[7])
fig9, axs9 = array_plot(filteredst[8], tlist[8], mdccmlist[8], vellist[8], bazlist[8], ccmplot=True, mcthresh=0.6, sigma_tau=sig_taulist[8])


if freq_band_type == 'linear':
    fig1.savefig(save_dir + 'MCCM_example_least_squares_linear_1.png', dpi=150)
    fig2.savefig(save_dir + 'MCCM_example_least_squares_linear_2.png', dpi=150)
    fig3.savefig(save_dir + 'MCCM_example_least_squares_linear_3.png', dpi=150)
    fig4.savefig(save_dir + 'MCCM_example_least_squares_linear_4.png', dpi=150)
    fig5.savefig(save_dir + 'MCCM_example_least_squares_linear_5.png', dpi=150)
    fig6.savefig(save_dir + 'MCCM_example_least_squares_linear_6.png', dpi=150)
    fig7.savefig(save_dir + 'MCCM_example_least_squares_linear_7.png', dpi=150)
    fig8.savefig(save_dir + 'MCCM_example_least_squares_linear_8.png', dpi=150)
    fig9.savefig(save_dir + 'MCCM_example_least_squares_linear_9.png', dpi=150)
elif freq_band_type == 'log':
    fig1.savefig(save_dir + 'MCCM_example_least_squares_log_1.png', dpi=150)
    fig2.savefig(save_dir + 'MCCM_example_least_squares_log_2.png', dpi=150)
    fig3.savefig(save_dir + 'MCCM_example_least_squares_log_3.png', dpi=150)
    fig4.savefig(save_dir + 'MCCM_example_least_squares_log_4.png', dpi=150)
    fig5.savefig(save_dir + 'MCCM_example_least_squares_log_5.png', dpi=150)
    fig6.savefig(save_dir + 'MCCM_example_least_squares_log_6.png', dpi=150)
    fig7.savefig(save_dir + 'MCCM_example_least_squares_log_7.png', dpi=150)
    fig8.savefig(save_dir + 'MCCM_example_least_squares_log_8.png', dpi=150)
    fig9.savefig(save_dir + 'MCCM_example_least_squares_log_9.png', dpi=150) """





# %%
