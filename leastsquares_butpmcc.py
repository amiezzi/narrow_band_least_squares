############################################################################
### Narrow Band Least Squares Method #######################################
### Breaks up frequencies into multiple bands (similar to PMCC method) #####
### Normal least squares uses the entire frequency  band for calculation ###
### Authors: Sneha Bhetanabhotla, Alex Iezzi, and Robin Matoza #############
############################################################################

###############
### Imports ###
###############
#from waveform_collection import gather_waveforms #Commented out by Sneha 
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
from array_processing.tools.plotting import array_plot
from lts_array import ltsva
import matplotlib.dates as mdates
from datetime import datetime, timedelta



##################
### User Input ###
##################

START = UTCDateTime('2010-05-28T13:30:00')
END = UTCDateTime('2010-05-28T15:30:00')
FMT = '%Y-%m-%dT%H:%M:%S.%f'

# RIOE coordinates
latlist = [-1.74812, -1.74749, -1.74906, -1.74805]
lonlist = [-78.62735, -78.62708, -78.62742, -78.62820]

#data_dir = '/Users/snehabhetanabhotla/Desktop/Research/data/'      # directory where data is located
data_dir = '/Users/aiezzi/Desktop/NSF_Postdoc/Array_Processing_Research/PMCC_Training/data/'       # directory where data is located


# same parameters as PMCC
# Filtering
FMIN = 0.1  # [Hz]
FMAX = 10    # [Hz] #should not exceed 20 Hz 

#same parameters as PMCC
# Array processing
WINLEN = 50  # [s]
WINOVER = 0.5

nbands = 15 #indicates number of frequency bands 
freq_band_type = 'linear' # indicates linear or logarithmic spacing for frequency bands; 'linear' or 'log'

ALPHA = 1.0


### Figure Save Options ###
#save_dir = '/Users/snehabhetanabhotla/Desktop/Research/Plots/LeastSquaresCode/LeastSquaresButPMCC/'			# directory in which to save figures
save_dir = '/Users/aiezzi/Desktop/NSF_Postdoc/Array_Processing_Research/narrow_band_least_squares/Figures/'         # directory in which to save figures
file_type = '.png'
dpi_num = 300
fonts = 14
rcParams.update({'font.size': fonts})




##############################################################################
######################
### End User Input ###
######################
##############################################################################

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

# Time vector 
timevec = st[0].times('matplotlib')

# Calibrate the data
calib = -0.000113  # Pa/count
tr1.data = tr1.data*calib
tr2.data = tr2.data*calib
tr3.data = tr3.data*calib
tr4.data = tr4.data*calib

#Default plot to sanity check 
#can also do print(st)
#st.plot() 

##################################################################################


#%% Grab and filter waveforms										

freqrange = FMAX - FMIN
freqinterval = freqrange / nbands

if freq_band_type == 'linear':
    freqlist = np.arange(FMIN, FMAX, freqinterval)
elif freq_band_type == 'log':
    FMINL = math.log(FMIN, 10)
    FMAXL = math.log(FMAX, 10)
    freqlist = np.logspace(FMINL, FMAXL, num = nbands)

######################################################

rij = getrij(latlist, lonlist)

filteredst = [] #list of filtered streams 
vellist = [] #velocity list 
bazlist = [] #back azimuth list
tlist = [] #time? 
mdccmlist = [] #mdccm 
stdictlist = [] #?
sig_taulist = [] #? 



######################
### PMCC like plot ###
######################
#fig, axs = plt.subplots(2,1)
#fig.set_size_inches(10, 5)
fig = plt.figure(figsize=(15,13), dpi=dpi_num)
gs = gridspec.GridSpec(4,2, width_ratios=[3,0.1], height_ratios=[1,1,1,1])

cm = 'RdYlBu_r'
cax = (FMIN, FMAX)

ax0 = plt.subplot(gs[0,0])  # Pressure Plot
ax1 = plt.subplot(gs[1,0])  # Backazimuth Plot
ax2 = plt.subplot(gs[2,0])  # Trace Velocity Plot
ax3 = plt.subplot(gs[3,0])  # Scatter Plot

for x in range(nbands - 1): 
    tempst_filter = st.copy()
    tempfmin = freqlist[x]
    tempfmax = freqlist[x+1]
    tempst_filter.filter('bandpass', freqmin = tempfmin, freqmax = tempfmax, corners=2, zerophase = True)
    tempst_filter.taper(max_percentage=0.01)
    filteredst.append(tempst_filter)
    #Array Processing 
    vel, baz, t, mdccm, stdict, sig_tau = ltsva(tempst_filter, rij, WINLEN, WINOVER, ALPHA)

    # Convert array processing output to numpy array of floats
    vel_float = []
    for ii in range(len(vel)):
        vel_float.append(float(vel[ii]))
    vel_float = np.array(vel_float)

    baz_float = []
    for ii in range(len(baz)):
        baz_float.append(float(baz[ii]))
    baz_float = np.array(baz_float)

    mdccm_float = []
    for ii in range(len(mdccm)):
        mdccm_float.append(float(mdccm[ii]))
    mdccm_float = np.array(mdccm_float)

    t_float = []
    for ii in range(len(t)):
        t_float.append(float(t[ii]))
    t_float = np.array(t_float)

    height_temp = tempfmax - tempfmin # height of frequency rectangles

    # Pressure plot
    if x == 0:
        ax0.plot(timevec, tempst_filter[0], 'k') # plots pressure for the first band



    # Backazimuth Plot 
    #normal = pl.Normalize(baz_float.min(), baz_float.max())
    normal_baz = pl.Normalize(0, 360)
    colors_baz = pl.cm.jet(normal_baz(baz_float))

    for ii in range(len(t_float)-1):
        width_temp = t_float[ii+1] - t_float[ii]
        if mdccm_float[ii] >= 0.6: 
        #if np.greater(mdccmlist[ii], 0.6):
            x_temp = t_float[ii]
            y_temp = tempfmin
            rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_baz[ii])
            ax1.add_patch(rect)

    # Trace Velocity Plot 
    #normal_vel = pl.Normalize(vel_float.min(), vel_float.max()) # This re-normalizing for each narrow frequency band (not right)
    normal_vel = pl.Normalize(0.2,0.5)
    colors_vel = pl.cm.jet(normal_vel(vel_float))

    for ii in range(len(t_float)-1):
        width_temp = t_float[ii+1] - t_float[ii]
        if mdccm_float[ii] >= 0.6: 
        #if np.greater(mdccmlist[ii], 0.6):
            x_temp = t_float[ii]
            y_temp = tempfmin
            rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_vel[ii])
            ax2.add_patch(rect)


    # Scatter plot
    for ii in range(len(mdccm)):
        if mdccm[ii] >= 0.6:
            sc = ax3.scatter(t[ii],baz[ii], c=tempfmax, edgecolors='k', lw=0.3, cmap=cm)
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


### Save figure ###
plt.tight_layout()
fig.savefig(save_dir + 'LeastSquaresButPMCC', dpi=150)
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
