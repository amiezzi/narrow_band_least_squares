#Least Squares Method 
#Breaks up frequencies into multiple bands (similar to PMCC method) instead of one singular calculation.
#%%User defined parameters

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
from array_processing.algorithms.helpers import getrij
from array_processing.tools.plotting import array_plot
from lts_array import ltsva
import matplotlib.dates as mdates
from datetime import datetime, timedelta


FMT = '%Y-%m-%dT%H:%M:%S.%f'

#same parameters as PMCC
# Filtering
FMIN = 0.1  # [Hz]
FMAX = 10    # [Hz] #should not exceed 20 Hz 

#same parameters as PMCC
# Array processing
WINLEN = 50  # [s]
WINOVER = 0.5

##################################################################################
### Added by Alex ###
#save_dir = '/Users/snehabhetanabhotla/Desktop/Research/Plots/LeastSquaresCode/LeastSquaresButPMCC/'			# directory in which to save figures
#data_dir = '/Users/snehabhetanabhotla/Desktop/Research/data/'		# directory where data is located
save_dir = '/Users/aiezzi/Desktop/'         # directory in which to save figures
data_dir = '/Users/aiezzi/Desktop/NSF_Postdoc/PMCC_Training/data/'       # directory where data is located



START = UTCDateTime('2010-05-28T13:30:00')
END = UTCDateTime('2010-05-28T15:30:00')

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

#Time vector 
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

 
nbands = 10 #indicates number of bands 
type = 'linear' #indicates linear or logarithmic spacing 

freqrange = FMAX - FMIN
freqinterval = freqrange / nbands

if type == 'linear':
    freqlist = np.arange(FMIN, FMAX, freqinterval)
elif type == 'log':
    FMINL = math.log(FMIN, 10)
    FMAXL = math.log(FMAX, 10)
    freqlist = np.logspace(FMINL, FMAXL, num = nbands)


ALPHA = 1.0

######################################################
### Added by Alex ### 
# RIOE coordinates
latlist = [-1.74812, -1.74749, -1.74906, -1.74805]
lonlist = [-78.62735, -78.62708, -78.62742, -78.62820]
######################################################


rij = getrij(latlist, lonlist)

filteredst = [] #list of filtered streams 
vellist = [] #velocity list 
bazlist = [] #back azimuth list
tlist = [] #time? 
mdccmlist = [] #mdccm 
stdictlist = [] #?
sig_taulist = [] #? 

# PMCC like plot
fig, axs = plt.subplots(2,1)
fig.set_size_inches(10, 5)


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

    # Pressure plot
    if x == 0:
        #Pressure Plot
        axs[0].plot(timevec, tempst_filter[0], 'k') #plots pressure for the first band
        axs[0].set_ylabel('Pressure [Pa]') 

    height_temp = tempfmax - tempfmin

    # Plot Backazimuth
    normal = pl.Normalize(baz_float.min(), baz_float.max())
    colors = pl.cm.jet(normal(baz_float))

    for ii in range(len(t_float)-1):
        width_temp = t_float[ii+1] - t_float[ii]
        if mdccm_float[ii] >= 0.6: 
        #if np.greater(mdccmlist[ii], 0.6):
            x_temp = t_float[ii]
            y_temp = tempfmin
            rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors[ii])
            axs[1].add_patch(rect)

   
axs[1].set_ylabel('Frequency [Hz]')   
axs[0].xaxis_date()
axs[1].xaxis_date()
axs[1].set_ylim(FMIN,FMAX)
axs[1].set_xlim(t_float[0],t_float[-1])

# Add colorbar
cax, _ = cbar.make_axes(axs[1]) 
cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.jet,norm=normal) 

# Save figure
fig.savefig(save_dir + 'LeastSquaresButPMCC', dpi=150)


'''
# Scatter Plot Robin Asked For
fig, axs = plt.subplots(2, 1, sharex='col')
fig.set_size_inches(10, 6)

# Specify the colormap.
cm = 'RdYlBu_r'
# Colorbar/y-axis limits for frequency.
cax = (FMIN, FMAX)


for x in range(nbands - 1): 
    tempst_filter = st.copy()
    tempfmin = freqlist[x]
    tempfmax = freqlist[x+1]
    tempst_filter.filter('bandpass', freqmin = tempfmin, freqmax = tempfmax, corners=2, zerophase = True)
    tempst_filter.taper(max_percentage=0.01)
    filteredst.append(tempst_filter)
    #Array Processing 
    vel, baz, t, mdccm, stdict, sig_tau = ltsva(tempst_filter, rij, WINLEN, WINOVER, ALPHA)


    #Adding output values to lists 
    #vellist.append(vel)
    #bazlist.append(baz)
    #tlist.append(t)
    #mdccmlist.append(mdccm)
    #stdictlist.append(stdict)
    #sig_taulist.append(sig_tau)

    # Pressure plot
    if x == 0:
        #Pressure Plot
        axs[0].plot(timevec, tempst_filter[0], 'k') #plots pressure for the first band
        axs[0].set_ylabel('Pressure [Pa]') 


    #Backazimuth Plot 
    for ii in range(len(mdccm)):
        if mdccm[ii] >= 0.6:
            sc = axs[1].scatter(t[ii],baz[ii], c=tempfmax, edgecolors='k', lw=0.3, cmap=cm)
            sc.set_clim(cax)
   

axs[1].set_ylabel('Backazimuth [degrees]')
axs[1].xaxis_date()

# Add the colorbar.
cbot = axs[1].get_position().y0
ctop = axs[1].get_position().y1
cbaxes = fig.add_axes([0.92, cbot, 0.02, ctop-cbot])
hc = plt.colorbar(sc, cax=cbaxes)
hc.set_label('Frequency [Hz]')  

# Save figure
fig.savefig(save_dir + 'LeastSquaresButPMCC_scatter', dpi=150)
    
'''



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


if type == 'linear':
    fig1.savefig(save_dir + 'MCCM_example_least_squares_linear_1.png', dpi=150)
    fig2.savefig(save_dir + 'MCCM_example_least_squares_linear_2.png', dpi=150)
    fig3.savefig(save_dir + 'MCCM_example_least_squares_linear_3.png', dpi=150)
    fig4.savefig(save_dir + 'MCCM_example_least_squares_linear_4.png', dpi=150)
    fig5.savefig(save_dir + 'MCCM_example_least_squares_linear_5.png', dpi=150)
    fig6.savefig(save_dir + 'MCCM_example_least_squares_linear_6.png', dpi=150)
    fig7.savefig(save_dir + 'MCCM_example_least_squares_linear_7.png', dpi=150)
    fig8.savefig(save_dir + 'MCCM_example_least_squares_linear_8.png', dpi=150)
    fig9.savefig(save_dir + 'MCCM_example_least_squares_linear_9.png', dpi=150)
elif type == 'log':
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
