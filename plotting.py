import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import pylab as pl
from matplotlib.patches import Rectangle
import matplotlib.colorbar as cbar
import time
from copy import deepcopy
from collections import Counter


dpi_num = 300
fonts = 14                                 
rcParams.update({'font.size': fonts})


def broadband_filter_response_plot(w, h, FMIN, FMAX, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE):
    '''
    Plots the filter frequency response for broadband least-squares processing
    Args:
        w: The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the range [0, pi) (radians/sample) [ndarray]
        h: The frequency response, as complex numbers. [ndarray]
        FMIN: Minimum frequency [float] [Hz]
        FMAX: Maximum frequency [float] [Hz]
        FILTER_TYPE: filter type [string]
        FILTER_ORDER: filter order [integer]
        FILTER_RIPPLE: filter ripple (if Chebyshev I filter) [float]
    Returns:
        fig: Figure handle (:class:`~matplotlib.figure.Figure`)
    '''
    fig = plt.figure(figsize=(8,5), dpi=dpi_num)
    gs = gridspec.GridSpec(1,1)

    ax0 = plt.subplot(gs[0,0]) 
    ax0.semilogx(w, 20 * np.log10(abs(h)))
    ax0.axvline(x=FMIN, color='k', ls='--')
    ax0.axvline(x=FMAX, color='k', ls='--')
    ax0.set_ylabel('Amplitude [dB]', fontsize=fonts+2, fontweight='bold')
    ax0.set_xlabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')
    #ax0.set_xlim(0,10)
    ax0.set_ylim(-5,0.1)
    ax0.text(0.02, 0.05, 'Filter Type = ' + FILTER_TYPE, transform=ax0.transAxes)
    ax0.text(0.02, 0.1, 'Filter Order = ' + str(FILTER_ORDER), transform=ax0.transAxes)
    if FILTER_TYPE == 'cheby1':
        ax0.text(0.02, 0.15, 'Ripple = ' + str(FILTER_RIPPLE), transform=ax0.transAxes)
    plt.tight_layout()                 

    return fig


def broadband_plot(st, vel_array, baz_array, mdccm_array, t_array, MDCCM_THRESH, ALPHA, stdict, sig_tau):
    '''
    Plots the array processing results for broadband least-squares processing 
    Args:
        st: Filtered data. Assumes response has been removed. (:class:`~obspy.core.stream.Stream`)
        vel_array: array of trace velocity processing results
        baz_array: array of backazimuth processing results
        mdccm_array: array of MdCCM processing results
        t_array: array of times for processing results
        MDCCM_THRESH: Threshold value of MdCCM for plotting; Must be between 0 and 1 [float]
        ALPHA: Use least-squares or LTS processing 
        stdict: dictionary with dropped elements for LTS [dictionary]
        sig_tau: sigma tau processing results
    Returns:
        fig: Figure handle (:class:`~matplotlib.figure.Figure`)
    '''
    
    cm = 'YlGnBu'
    cax = [0,1.0]

    fig = plt.figure(figsize=(15,15), dpi=dpi_num)
    gs = gridspec.GridSpec(5,2, width_ratios=[3,0.1], height_ratios=[1,1,1,1,1])

    # Pressure plot (broadband bandpass filtered data)
    ax0 = plt.subplot(gs[0,0])  # Pressure Plot
    timevec = st[0].times('matplotlib') # Time vector for plotting
    ax0.plot(timevec, st[0], 'k') 
    ax0.set_ylabel('Pressure [Pa]', fontsize=fonts+2, fontweight='bold') 
    ax0.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax0.set_title('a)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax0.xaxis_date()
    ax0.set_xlim(timevec[1], timevec[-1])

    # MdCCM
    ax1 = plt.subplot(gs[1,0])  
    sc = ax1.scatter(t_array, mdccm_array, c=mdccm_array, edgecolors='k', lw=0.3, cmap=cm)
    sc.set_clim(cax)
    ax1.set_ylabel('MdCCM', fontsize=fonts+2, fontweight='bold')  
    ax1.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax1.set_title('b)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax1.xaxis_date()
    ax1.set_ylim(0,1)
    ax1.set_xlim(t_array[0],t_array[-1])
    ax1.plot([t_array[0], t_array[-1]], [MDCCM_THRESH, MDCCM_THRESH], 'k--')

    # Backazimuth 
    ax2 = plt.subplot(gs[2,0])  
    sc = ax2.scatter(t_array, baz_array, c=mdccm_array, edgecolors='k', lw=0.3, cmap=cm)
    sc.set_clim(cax)
    ax2.set_ylabel('Backazimuth [deg]', fontsize=fonts+2, fontweight='bold')  
    ax2.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax2.set_title('c)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax2.xaxis_date()
    ax2.set_ylim(0,360)
    ax2.set_xlim(t_array[0],t_array[-1])

    # Trace Velocity 
    ax3 = plt.subplot(gs[3,0])  
    sc = ax3.scatter(t_array, vel_array, c=mdccm_array, edgecolors='k', lw=0.3, cmap=cm)
    sc.set_clim(cax)
    ax3.set_ylabel('Trace Velocity [km/s]', fontsize=fonts+2, fontweight='bold')  
    ax3.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax3.set_title('d)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax3.xaxis_date()
    ax3.set_ylim(0.2,0.5)
    ax3.set_xlim(t_array[0],t_array[-1])

    # Sigma Tau or Dropped Elements
    ax4 = plt.subplot(gs[4,0])  
    if ALPHA == 1.0:
        # Plot Sigma Tau
        sc = ax4.scatter(t_array, sig_tau, c=mdccm_array, edgecolors='k', lw=0.3, cmap=cm)
        sc.set_clim(cax)
        ax4.set_ylim(-0.5,5)
        ax4.set_ylabel('Sigma Tau ('r'$\sigma_\tau$)', fontsize=fonts, fontweight='bold')

        # Add the MdCCM colorbar
        cbaxes = plt.subplot(gs[1:5,1]) 
        hc = plt.colorbar(sc, cax=cbaxes)
        hc.set_label('MdCCM', fontsize=fonts, fontweight='bold')


    elif ALPHA < 1.0:
        # Plot LTS Dropped Stations
        ndict = deepcopy(stdict)
        n = ndict['size']
        ndict.pop('size', None)
        tstamps = list(ndict.keys())
        tstampsfloat = [float(ii) for ii in tstamps]

        # Set the second colormap for station pairs.
        cm2 = plt.get_cmap('binary', (n-1))
        initplot = np.empty(len(t_array))
        initplot.fill(1)

        ax4.scatter(np.array([t_array[0], t_array[-1]]), np.array([0.01, 0.01]), c='w')
        ax4.axis('tight')
        ax4.set_ylabel('Element [#]', fontsize=fonts+2, fontweight='bold')
        ax4.set_ylim(0.5, n+0.5)

        # Loop through the stdict for each flag and plot
        for jj in range(len(tstamps)):
            z = Counter(list(ndict[tstamps[jj]]))
            keys, vals = z.keys(), z.values()
            keys, vals = np.array(list(keys)), np.array(list(vals))
            pts = np.tile(tstampsfloat[jj], len(keys))
            sc2 = ax4.scatter(pts, keys, c=vals, edgecolors='k',lw=0.1, cmap=cm2, vmin=0.5, vmax=n-0.5)

        cbaxes = plt.subplot(gs[4,1])  # Colorbar; Dropped stations
        hc = plt.colorbar(sc2, orientation="vertical", cax=cbaxes)
        hc.set_label('# of Flagged\nElement Pairs', fontsize=fonts+2, fontweight='bold')

        # Add the MdCCM colorbar
        cbaxes = plt.subplot(gs[1:4,1]) 
        hc = plt.colorbar(sc, cax=cbaxes)
        hc.set_label('MdCCM', fontsize=fonts, fontweight='bold')

    ax4.set_title('e)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax4.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax4.xaxis_date()
    ax4.set_xlim(t_array[0],t_array[-1])  


    plt.tight_layout()
    return fig



def narrow_band_processing_parameters_plot(rij, FREQ_BAND_TYPE, freqlist, WINLEN_list, NBANDS, FMIN, FMAX, w_array, h_array, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE):
    '''
    Plots the processing parameters for narrow-band least-squares processing 
    Args:
        rij: Coordinates of sensors as eastings & northings in a ``(2, N)`` array [km]
        FREQ_BAND_TYPE: indicates linear or logarithmic spacing for frequency bands
        freqlist: List of frequency bounds for narrow-band processing
        WINLEN_list: list of window lengths for each narrow frequency band
        NBANDS: number of frequency bands [integer]
        FMIN: Minimum frequency [float] [Hz]
        FMAX: Maximum frequency [float] [Hz]
        w_array: The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the range [0, pi) (radians/sample) [ndarray]
        h_array: The frequency response, as complex numbers. [ndarray]
        FILTER_TYPE: filter type [string]
        FILTER_ORDER: filter order [integer]
        FILTER_RIPPLE: filter ripple (if Chebyshev I filter) [float]
    Returns:
        fig: Figure handle (:class:`~matplotlib.figure.Figure`)
    '''
    height = []
    for ii in range(NBANDS):
        if FREQ_BAND_TYPE == '2_octave_over':
            height.append(freqlist[ii+2]- freqlist[ii])
        else:   
            height.append(freqlist[ii+1]- freqlist[ii])

    fig = plt.figure(figsize=(10,10), dpi=dpi_num)
    gs = gridspec.GridSpec(2,2)

    ax0 = plt.subplot(gs[0,0]) 
    ax0.scatter(rij[0], rij[1]) 
    ax0.set_xlabel('X [km]', fontsize=fonts+2, fontweight='bold')
    ax0.set_ylabel('Y [km]', fontsize=fonts+2, fontweight='bold')
    ax0.axis('square')
    ax0.grid()
    ax0.set_title('a) Array Geometry', loc='left', fontsize=fonts+2, fontweight='bold')

    ax1 = plt.subplot(gs[0,1]) 
    if FREQ_BAND_TYPE == '2_octave_over':
        ax1.barh(freqlist[:-2], WINLEN_list, height=height, align='edge', color='grey', edgecolor='k', alpha=0.25)
    else:
        ax1.barh(freqlist[:-1], WINLEN_list, height=height, align='edge', color='grey', edgecolor='k', alpha=0.5)
    
    if FREQ_BAND_TYPE == 'linear':
        ax1.set_ylim(-0.1,FMAX+1)
    else:
        plt.yscale('log')
        if FMAX < 10:
            ax1.set_ylim(-0.1,FMAX+2)
        elif FMAX >=10:
            ax1.set_ylim(-0.1,FMAX+10)


    ax1.set_xlabel('Window Length [s]',fontsize=fonts+2, fontweight='bold')
    ax1.set_ylabel('Frequency [Hz]',fontsize=fonts+2, fontweight='bold')
    ax1.set_title('b) Window Length', loc='left', fontsize=fonts+2, fontweight='bold')
    ax1.text(0.02, 0.95, '# of Bands = ' + str(NBANDS), transform=ax1.transAxes, horizontalalignment='left', fontsize=fonts-2)
    ax1.text(0.98, 0.95, 'FMIN = ' + str(round(FMIN, 2)) + ', FMAX = ' + str(round(FMAX, 2)), transform=ax1.transAxes, horizontalalignment='right', fontsize=fonts-2)
    

    ax2 = plt.subplot(gs[1,0:2]) 
    for ii in range(NBANDS):
        temp_w = w_array[ii,:-1]
        temp_h = h_array[ii,:-1]
        if FREQ_BAND_TYPE == 'linear':
            ax2.plot(temp_w, 20 * np.log10(abs(temp_h)))
        else:
            ax2.semilogx(temp_w, 20 * np.log10(abs(temp_h)))
        ax2.axvline(x=freqlist[ii], ymax=0.9, color='k', ls='--')
    ax2.axvline(x=freqlist[-1], ymax=0.9, color='k', ls='--')
    ax2.set_ylabel('Amplitude [dB]', fontsize=fonts+2, fontweight='bold')
    ax2.set_xlabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')
    ax2.set_xlim(FMIN-0.01,FMAX+1)
    ax2.set_ylim(-3,0.4)
    ax2.set_title('c) Narrow Band Filters', loc='left', fontsize=fonts+2, fontweight='bold')
    ax2.text(0.02, 0.95, 'Filter Type = ' + FILTER_TYPE, transform=ax2.transAxes, horizontalalignment='left', fontsize=fonts-2)
    ax2.text(0.98, 0.95, 'Filter Order = ' + str(FILTER_ORDER), transform=ax2.transAxes, horizontalalignment='right', fontsize=fonts-2)
    if FILTER_TYPE == 'cheby1':
        ax2.text(0.5, 0.95, 'Ripple = ' + str(FILTER_RIPPLE), transform=ax2.transAxes, horizontalalignment='center', fontsize=fonts-2)
    plt.tight_layout()
    return fig






def narrow_band_plot(FMIN, FMAX, st, NBANDS, freqlist, FREQ_BAND_TYPE, vel_array, baz_array, mdccm_array, t_array, num_compute_list, MDCCM_THRESH):
    '''
    Plots the results for narrow-band least-squares processing (no sigma tau or dropped stations)
    Args:
        FMIN: Minimum frequency [float] [Hz]
        FMAX: Maximum frequency [float] [Hz]
        st: Filtered data. Assumes response has been removed. (:class:`~obspy.core.stream.Stream`)
        NBANDS: number of frequency bands [integer]
        freqlist: List of frequency bounds for narrow-band processing
        FREQ_BAND_TYPE: indicates linear or logarithmic spacing for frequency bands
        vel_array: array of trace velocity processing results
        baz_array: array of backazimuth processing results
        mdccm_array: array of MdCCM processing results
        t_array: array of times for processing results
        num_compute_list: list of number of windows for each frequency band array processing
        MDCCM_THRESH: Threshold value of MdCCM for plotting; Must be between 0 and 1 [float]
    Returns:
        fig: Figure handle (:class:`~matplotlib.figure.Figure`)
    '''
    timevec = st[0].times('matplotlib') # Time vector for plotting
    cm = 'turbo'
    cm_mdccm = 'YlGnBu'
    cax = (FMIN, FMAX)

    fig = plt.figure(figsize=(15,20), dpi=dpi_num)
    gs = gridspec.GridSpec(6,2, width_ratios=[3,0.1], height_ratios=[1,1,1,1,1,1])

    # Pressure plot (broadband bandpass filtered data)
    ax0 = plt.subplot(gs[0,0])  # Pressure Plot
    ax0.plot(timevec, st[0], 'k') # plots pressure for the first band

    # Initialize other plots
    ax1 = plt.subplot(gs[1,0])  # MdCCM Plot
    ax2 = plt.subplot(gs[2,0])  # Backazimuth Plot
    ax3 = plt.subplot(gs[3,0])  # Trace Velocity Plot
    ax4 = plt.subplot(gs[4,0])  # Scatter Plot
    ax5 = plt.subplot(gs[5,0])  # Scatter Plot Trace Velocity

    for ii in range(NBANDS): 
        # Check if overlapping bands
        if FREQ_BAND_TYPE == '2_octave_over':
            tempfmin = freqlist[ii]
            tempfmax = freqlist[ii+2]
        # All others
        else:
            tempfmin = freqlist[ii]
            tempfmax = freqlist[ii+1]
        height_temp = tempfmax - tempfmin # height of frequency rectangles
        tempfavg = tempfmin + (height_temp/2)        # center point of the frequency interval

        # Gather array processing results for this narrow frequency band
        vel_temp = vel_array[ii,:]
        baz_temp = baz_array[ii,:]
        mdccm_temp = mdccm_array[ii,:]
        t_temp = t_array[ii,:]

        # Trim each vector to ignore NAN and zero values
        vel_float = vel_temp[:num_compute_list[ii]]
        baz_float = baz_temp[:num_compute_list[ii]]
        mdccm_float = mdccm_temp[:num_compute_list[ii]]
        t_float = t_temp[:num_compute_list[ii]]

        # Initialize colorbars
        normal_baz = pl.Normalize(0, 360)
        colors_baz = pl.cm.turbo(normal_baz(baz_float))

        for jj in range(len(t_float)-1):
            if vel_float[jj] >= 0.5:
                vel_float[jj] = 0.51
            elif vel_float[jj] <= 0.2:
                vel_float[jj] = 0.19
        normal_vel = pl.Normalize(0.2,0.5)
        colors_vel = pl.cm.turbo(normal_vel(vel_float))

        normal_mdccm = pl.Normalize(0.,1.0)
        colors_mdccm = pl.cm.YlGnBu(normal_mdccm(mdccm_float))


        # Find indices where mdccm_float >= MDCCM_THRESH
        mdccm_good_idx = [jj for jj,v in enumerate(mdccm_float) if v > MDCCM_THRESH]
        # Trim array to only have the indices where mdccm_float >= MDCCM_THRESH
        vel_good = [vel_float[jj] for jj in mdccm_good_idx]
        baz_good = [baz_float[jj] for jj in mdccm_good_idx]
        t_good = [t_float[jj] for jj in mdccm_good_idx]

       
        # Plot the scatter points
        tempfavg_array = np.repeat(tempfavg, len(t_good))
        # Scatter plot
        sc = ax4.scatter(t_good, baz_good, c=tempfavg_array, edgecolors='k', lw=0.3, cmap=cm)
        sc.set_clim(cax)

        # Scatter plot
        sc_vel = ax5.scatter(t_good, vel_good, c=tempfavg_array, edgecolors='k', lw=0.3, cmap=cm)
        sc_vel.set_clim(cax)



        # Loop through each narrow-band results vector and plot rectangles/scatter points
        for jj in range(len(t_float)-1):
            width_temp = t_float[jj+1] - t_float[jj]
            if mdccm_float[jj] >= MDCCM_THRESH: 
                x_temp = t_float[jj]
                y_temp = tempfmin

                # MdCCM Plot 
                rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_mdccm[jj])
                ax1.add_patch(rect)

                # Backazimuth plot 
                rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_baz[jj])
                ax2.add_patch(rect)

                # Trace Velocity Plot 
                rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_vel[jj])
                ax3.add_patch(rect)


        # MdCCM Loop through each narrow-band results vector and plot rectangles/scatter points
        for jj in range(len(t_float)-1):
            width_temp = t_float[jj+1] - t_float[jj]
            if mdccm_float[jj] < MDCCM_THRESH: 
                x_temp = t_float[jj]
                y_temp = tempfmin
                # MdCCM Plot 
                rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_mdccm[jj], alpha=0.5)
                ax1.add_patch(rect)




    #####################
    ### Add colorbars ###
    #####################

    # Add colorbar to mdccm plot
    cax = plt.subplot(gs[1,1]) 
    cb1 = cbar.ColorbarBase(cax, cmap=pl.cm.YlGnBu,norm=normal_mdccm,orientation='vertical') 
    cax.set_ylabel('MdCCM', fontsize=fonts+2, fontweight='bold')

    # Add colorbar to backazimuth plot
    cax = plt.subplot(gs[2,1]) 
    cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.turbo,norm=normal_baz,orientation='vertical', ticks=[0,90,180,270,360]) 
    cax.set_ylabel('Backazimuth [deg]', fontsize=fonts+2, fontweight='bold')

    # Add colorbar to trace velocity plot
    cax = plt.subplot(gs[3,1]) 
    cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.turbo,norm=normal_vel,orientation='vertical') 
    cax.set_ylabel('Trace Velocity [km/s]', fontsize=fonts+2, fontweight='bold')

    # Add colorbar to scatter plot of backazimuth and trace velocity
    cbaxes = plt.subplot(gs[4:6,1]) 
    if 'sc' in locals():
        hc = plt.colorbar(sc, cax=cbaxes,orientation='vertical') 
    cbaxes.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')



    ###################
    ### Format axes ###
    ###################

    # Pressure plot 
    ax0.xaxis_date()
    ax0.set_xlim(timevec[1], timevec[-1])
    ax0.set_ylabel('Pressure [Pa]', fontsize=fonts+2, fontweight='bold') 
    ax0.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax0.set_title('a)', loc='left', fontsize=fonts+2, fontweight='bold')

    # MdCCM Plot
    ax1.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
    ax1.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax1.set_title('b)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax1.xaxis_date()
    ax1.set_ylim(FMIN,FMAX)
    ax1.set_xlim(t_float[0],t_float[-1])

    # Backazimuth Plot
    ax2.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
    ax2.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax2.set_title('c)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax2.xaxis_date()
    ax2.set_ylim(FMIN,FMAX)
    ax2.set_xlim(t_float[0],t_float[-1])

    # Trace Velocity Plot
    ax3.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
    ax3.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax3.set_title('d)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax3.xaxis_date()
    ax3.set_ylim(FMIN,FMAX)
    ax3.set_xlim(t_float[0],t_float[-1])

    # Scatter Plot
    ax4.set_ylabel('Backazimuth [deg]', fontsize=fonts+2, fontweight='bold')  
    ax4.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax4.set_title('e)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax4.xaxis_date()
    ax4.set_ylim(0,360)
    ax4.set_xlim(t_float[0],t_float[-1])

    # Scatter Plot
    ax5.set_ylabel('Trace Velocity [km/s]', fontsize=fonts+2, fontweight='bold')  
    ax5.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax5.set_title('f)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax5.xaxis_date()
    ax5.set_ylim(0.2,0.5)
    ax5.set_xlim(t_float[0],t_float[-1])

    plt.tight_layout()
    return fig



def narrow_band_stau_plot(FMIN, FMAX, st, NBANDS, freqlist, FREQ_BAND_TYPE, vel_array, baz_array, mdccm_array, t_array, sig_tau_array, num_compute_list, MDCCM_THRESH, ALPHA):
    '''
    Plots the results for narrow-band least-squares processing with sigma tau
    Args:
        FMIN: Minimum frequency [float] [Hz]
        FMAX: Maximum frequency [float] [Hz]
        st: Filtered data. Assumes response has been removed. (:class:`~obspy.core.stream.Stream`)
        NBANDS: number of frequency bands [integer]
        freqlist: List of frequency bounds for narrow-band processing
        FREQ_BAND_TYPE: indicates linear or logarithmic spacing for frequency bands
        vel_array: array of trace velocity processing results
        baz_array: array of backazimuth processing results
        mdccm_array: array of MdCCM processing results
        t_array: array of times for processing results
        sig_tau_array: array of sigma tau processing results
        num_compute_list: list of number of windows for each frequency band array processing
        MDCCM_THRESH: Threshold value of MdCCM for plotting; Must be between 0 and 1 [float]
        ALPHA: Use ordinary least-squares or LTS processing 
    Returns:
        fig: Figure handle (:class:`~matplotlib.figure.Figure`)
    '''
    timevec = st[0].times('matplotlib') # Time vector for plotting
    cm = 'turbo'
    cm_mdccm = 'YlGnBu'
    cax = (FMIN, FMAX)

    fig = plt.figure(figsize=(15,20), dpi=dpi_num)
    gs = gridspec.GridSpec(8,2, width_ratios=[3,0.1], height_ratios=[1,1,1,1,1,1,1,1])

    # Pressure plot (broadband bandpass filtered data)
    ax0 = plt.subplot(gs[0,0])  # Pressure Plot
    ax0.plot(timevec, st[0], 'k') # plots pressure for the first band

    # Initialize other plots
    ax1 = plt.subplot(gs[1,0])  # MdCCM Plot
    ax2 = plt.subplot(gs[2,0])  # Sigma Tau Plot
    ax3 = plt.subplot(gs[3,0])  # Backazimuth Plot
    ax4 = plt.subplot(gs[4,0])  # Trace Velocity Plot
    ax5 = plt.subplot(gs[5,0])  # Scatter Plot; Sigma Tau
    ax6 = plt.subplot(gs[6,0])  # Scatter Plot; Backazimuth
    ax7 = plt.subplot(gs[7,0])  # Scatter Plot Trace Velocity


    for ii in range(NBANDS): 
        # Check if overlapping bands
        if FREQ_BAND_TYPE == '2_octave_over':
            tempfmin = freqlist[ii]
            tempfmax = freqlist[ii+2]
        # All others
        else:
            tempfmin = freqlist[ii]
            tempfmax = freqlist[ii+1]
        height_temp = tempfmax - tempfmin # height of frequency rectangles
        tempfavg = tempfmin + (height_temp/2)        # center point of the frequency interval

        # Gather array processing results for this narrow frequency band
        vel_temp = vel_array[ii,:]
        baz_temp = baz_array[ii,:]
        mdccm_temp = mdccm_array[ii,:]
        t_temp = t_array[ii,:]
        sig_tau_temp = sig_tau_array[ii,:]

        # Trim each vector to ignore NAN and zero values
        vel_float = vel_temp[:num_compute_list[ii]]
        baz_float = baz_temp[:num_compute_list[ii]]
        mdccm_float = mdccm_temp[:num_compute_list[ii]]
        t_float = t_temp[:num_compute_list[ii]]
        sig_tau_float = sig_tau_temp[:num_compute_list[ii]]

        # Initialize colorbars
        normal_baz = pl.Normalize(0, 360)
        colors_baz = pl.cm.jet(normal_baz(baz_float))

        for jj in range(len(t_float)-1):
            if vel_float[jj] >= 0.5:
                vel_float[jj] = 0.51
            elif vel_float[jj] <= 0.2:
                vel_float[jj] = 0.19
        normal_vel = pl.Normalize(0.2,0.5)
        colors_vel = pl.cm.jet(normal_vel(vel_float))

        normal_mdccm = pl.Normalize(0.,1.0)
        colors_mdccm = pl.cm.YlGnBu(normal_mdccm(mdccm_float))

        for jj in range(len(t_float)-1):
            if sig_tau_float[jj] >= 5:
                sig_tau_float[jj] = 5.1

        normal_sig_tau = pl.Normalize(0.,5.0)
        colors_sig_tau = pl.cm.YlGnBu_r(normal_sig_tau(sig_tau_float))


        # Find indices where mdccm_float >= MDCCM_THRESH
        mdccm_good_idx = [jj for jj,v in enumerate(mdccm_float) if v > MDCCM_THRESH]
        # Trim array to only have the indices where mdccm_float >= MDCCM_THRESH
        vel_good = [vel_float[jj] for jj in mdccm_good_idx]
        baz_good = [baz_float[jj] for jj in mdccm_good_idx]
        t_good = [t_float[jj] for jj in mdccm_good_idx]
        sig_tau_good = [sig_tau_float[jj] for jj in mdccm_good_idx]

       
        # Plot the scatter points
        tempfavg_array = np.repeat(tempfavg, len(t_good))

        # Scatter plot; Sigma Tau
        if ALPHA == 1.0:
            sc_sig = ax5.scatter(t_good, sig_tau_good, c=tempfavg_array, edgecolors='k', lw=0.3, cmap=cm)
            sc_sig.set_clim(cax)
        elif ALPHA < 1.0:
            print('You ran LTS with ALPHA = ' + str(ALPHA) + '. It would be better to use "narrow_band_lts_plot" and "narrow_band_lts_dropped_station_plot".')

        # Scatter plot
        sc = ax6.scatter(t_good, baz_good, c=tempfavg_array, edgecolors='k', lw=0.3, cmap=cm)
        sc.set_clim(cax)

        # Scatter plot
        sc_vel = ax7.scatter(t_good, vel_good, c=tempfavg_array, edgecolors='k', lw=0.3, cmap=cm)
        sc_vel.set_clim(cax)


        # Loop through each narrow-band results vector and plot rectangles/scatter points
        for jj in range(len(t_float)-1):
            width_temp = t_float[jj+1] - t_float[jj]
            if mdccm_float[jj] >= MDCCM_THRESH: 
                x_temp = t_float[jj]
                y_temp = tempfmin

                # MdCCM Plot 
                rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_mdccm[jj])
                ax1.add_patch(rect)

                # Sigma Tau Plot 
                if ALPHA ==1.0:
                    rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_sig_tau[jj])
                    ax2.add_patch(rect)

                # Backazimuth plot 
                rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_baz[jj])
                ax3.add_patch(rect)

                # Trace Velocity Plot 
                rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_vel[jj])
                ax4.add_patch(rect)



        # MdCCM Loop through each narrow-band results vector and plot rectangles/scatter points
        for jj in range(len(t_float)-1):
            width_temp = t_float[jj+1] - t_float[jj]
            if mdccm_float[jj] < MDCCM_THRESH: 
                x_temp = t_float[jj]
                y_temp = tempfmin
                # MdCCM Plot 
                rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_mdccm[jj], alpha=0.5)
                ax1.add_patch(rect)




    #####################
    ### Add colorbars ###
    #####################

    # Add colorbar to mdccm plot
    cax = plt.subplot(gs[1,1]) 
    cb1 = cbar.ColorbarBase(cax, cmap=pl.cm.YlGnBu,norm=normal_mdccm,orientation='vertical') 
    cax.set_ylabel('MdCCM', fontsize=fonts+2, fontweight='bold')

    # Add colorbar to sigma tau plot
    cax = plt.subplot(gs[2,1]) 
    cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.YlGnBu_r,norm=normal_sig_tau,orientation='vertical') 
    cax.set_ylabel('Sigma Tau\n('r'$\sigma_\tau$)', fontsize=fonts+2, fontweight='bold')

    # Add colorbar to backazimuth plot
    cax = plt.subplot(gs[3,1]) 
    cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.turbo,norm=normal_baz,orientation='vertical', ticks=[0,90,180,270,360]) 
    cax.set_ylabel('Backazimuth\n[deg]', fontsize=fonts+2, fontweight='bold')

    # Add colorbar to trace velocity plot
    cax = plt.subplot(gs[4,1]) 
    cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.turbo,norm=normal_vel,orientation='vertical') 
    cax.set_ylabel('Trace Velocity\n[km/s]', fontsize=fonts+2, fontweight='bold')

    # Add colorbar to scatter plot
    cbaxes = plt.subplot(gs[5:8,1]) 
    if 'sc' in locals():
        hc = plt.colorbar(sc, cax=cbaxes,orientation='vertical') 
    cbaxes.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')



    ###################
    ### Format axes ###
    ###################

    # Pressure plot 
    ax0.xaxis_date()
    ax0.set_xlim(timevec[1], timevec[-1])
    ax0.set_ylabel('Pressure [Pa]', fontsize=fonts+2, fontweight='bold') 
    ax0.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax0.set_title('a)', loc='left', fontsize=fonts+2, fontweight='bold')
    #ax0.set_xticklabels([])

    # MdCCM Plot
    ax1.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
    ax1.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax1.set_title('b)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax1.xaxis_date()
    ax1.set_ylim(FMIN,FMAX)
    ax1.set_xlim(t_float[0],t_float[-1])

    # Sigma Tau Plot
    ax2.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
    ax2.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax2.set_title('c)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax2.xaxis_date()
    ax2.set_ylim(FMIN,FMAX)
    ax2.set_xlim(t_float[0],t_float[-1])

    # Backazimuth Plot
    ax3.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
    ax3.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax3.set_title('d)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax3.xaxis_date()
    ax3.set_ylim(FMIN,FMAX)
    ax3.set_xlim(t_float[0],t_float[-1])

    # Trace Velocity Plot
    ax4.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
    ax4.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax4.set_title('e)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax4.xaxis_date()
    ax4.set_ylim(FMIN,FMAX)
    ax4.set_xlim(t_float[0],t_float[-1])

    # Scatter Plot
    ax5.set_ylabel('Sigma Tau\n('r'$\sigma_\tau$)', fontsize=fonts+2, fontweight='bold')  
    ax5.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax5.set_title('f)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax5.xaxis_date()
    ax5.set_ylim(-0.5,5)
    ax5.set_xlim(t_float[0],t_float[-1])

    # Scatter Plot
    ax6.set_ylabel('Backazimuth\n[deg]', fontsize=fonts+2, fontweight='bold')  
    ax6.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax6.set_title('g)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax6.xaxis_date()
    ax6.set_ylim(0,360)
    ax6.set_yticks([0,90,180,270,360])
    ax6.set_xlim(t_float[0],t_float[-1])

    # Scatter Plot
    ax7.set_ylabel('Trace Velocity\n[km/s]', fontsize=fonts+2, fontweight='bold')  
    ax7.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax7.set_title('h)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax7.xaxis_date()
    ax7.set_ylim(0.2,0.5)
    ax7.set_xlim(t_float[0],t_float[-1])


    plt.tight_layout()
    return fig







def narrow_band_lts_plot(FMIN, FMAX, st, NBANDS, freqlist, FREQ_BAND_TYPE, vel_array, baz_array, mdccm_array, t_array, stdict, num_compute_list, MDCCM_THRESH, ALPHA):
    '''
    Plots the results for narrow-band least-squares processing with overview of dropped stations
    Args:
        FMIN: Minimum frequency [float] [Hz]
        FMAX: Maximum frequency [float] [Hz]
        st: Filtered data. Assumes response has been removed. (:class:`~obspy.core.stream.Stream`)
        NBANDS: number of frequency bands [integer]
        freqlist: List of frequency bounds for narrow-band processing
        FREQ_BAND_TYPE: indicates linear or logarithmic spacing for frequency bands
        vel_array: array of trace velocity processing results
        baz_array: array of backazimuth processing results
        mdccm_array: array of MdCCM processing results
        t_array: array of times for processing results
        stdict: dictionary with dropped elements for LTS [dictionary]
        num_compute_list: list of number of windows for each frequency band array processing
        MDCCM_THRESH: Threshold value of MdCCM for plotting; Must be between 0 and 1 [float]
        ALPHA: Use ordinary least-squares or LTS processing 
    Returns:
        fig: Figure handle (:class:`~matplotlib.figure.Figure`)
    '''
    timevec = st[0].times('matplotlib') # Time vector for plotting
    cm = 'turbo'
    cm_mdccm = 'YlGnBu'
    cax = (FMIN, FMAX)

    fig = plt.figure(figsize=(15,20), dpi=dpi_num)
    gs = gridspec.GridSpec(7,2, width_ratios=[3,0.1], height_ratios=[1,1,1,1,1,1,1])

    # Pressure plot (broadband bandpass filtered data)
    ax0 = plt.subplot(gs[0,0])  # Pressure Plot
    ax0.plot(timevec, st[0], 'k') # plots pressure for the first band

    # Initialize other plots
    ax1 = plt.subplot(gs[1,0])  # MdCCM Plot
    ax2 = plt.subplot(gs[2,0])  # Backazimuth Plot
    ax3 = plt.subplot(gs[3,0])  # Trace Velocity Plot
    ax4 = plt.subplot(gs[4,0])  # Scatter Plot; Backazimuth
    ax5 = plt.subplot(gs[5,0])  # Scatter Plot Trace Velocity
    ax6 = plt.subplot(gs[6,0])  # Scatter Plot; Dropped stations


    for ii in range(NBANDS): 
        # Check if overlapping bands
        if FREQ_BAND_TYPE == '2_octave_over':
            tempfmin = freqlist[ii]
            tempfmax = freqlist[ii+2]
        # All others
        else:
            tempfmin = freqlist[ii]
            tempfmax = freqlist[ii+1]
        height_temp = tempfmax - tempfmin # height of frequency rectangles
        tempfavg = tempfmin + (height_temp/2)        # center point of the frequency interval

        # Gather array processing results for this narrow frequency band
        vel_temp = vel_array[ii,:]
        baz_temp = baz_array[ii,:]
        mdccm_temp = mdccm_array[ii,:]
        t_temp = t_array[ii,:]

        # Trim each vector to ignore NAN and zero values
        vel_float = vel_temp[:num_compute_list[ii]]
        baz_float = baz_temp[:num_compute_list[ii]]
        mdccm_float = mdccm_temp[:num_compute_list[ii]]
        t_float = t_temp[:num_compute_list[ii]]

        # Initialize colorbars
        normal_baz = pl.Normalize(0, 360)
        colors_baz = pl.cm.jet(normal_baz(baz_float))

        for jj in range(len(t_float)-1):
            if vel_float[jj] >= 0.5:
                vel_float[jj] = 0.51
            elif vel_float[jj] <= 0.2:
                vel_float[jj] = 0.19
        normal_vel = pl.Normalize(0.2,0.5)
        colors_vel = pl.cm.jet(normal_vel(vel_float))

        normal_mdccm = pl.Normalize(0.,1.0)
        colors_mdccm = pl.cm.YlGnBu(normal_mdccm(mdccm_float))



        # Find indices where mdccm_float >= MDCCM_THRESH
        mdccm_good_idx = [jj for jj,v in enumerate(mdccm_float) if v > MDCCM_THRESH]
        # Trim array to only have the indices where mdccm_float >= MDCCM_THRESH
        vel_good = [vel_float[jj] for jj in mdccm_good_idx]
        baz_good = [baz_float[jj] for jj in mdccm_good_idx]
        t_good = [t_float[jj] for jj in mdccm_good_idx]

       
        # Plot the scatter points
        tempfavg_array = np.repeat(tempfavg, len(t_good))


        # Scatter plot
        sc = ax4.scatter(t_good, baz_good, c=tempfavg_array, edgecolors='k', lw=0.3, cmap=cm)
        sc.set_clim(cax)

        # Scatter plot
        sc_vel = ax5.scatter(t_good, vel_good, c=tempfavg_array, edgecolors='k', lw=0.3, cmap=cm)
        sc_vel.set_clim(cax)


        # Loop through each narrow-band results vector and plot rectangles/scatter points
        for jj in range(len(t_float)-1):
            width_temp = t_float[jj+1] - t_float[jj]
            if mdccm_float[jj] >= MDCCM_THRESH: 
                x_temp = t_float[jj]
                y_temp = tempfmin

                # MdCCM Plot 
                rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_mdccm[jj])
                ax1.add_patch(rect)

                # Backazimuth plot 
                rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_baz[jj])
                ax2.add_patch(rect)

                # Trace Velocity Plot 
                rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_vel[jj])
                ax3.add_patch(rect)



        # MdCCM Loop through each narrow-band results vector and plot rectangles/scatter points
        for jj in range(len(t_float)-1):
            width_temp = t_float[jj+1] - t_float[jj]
            if mdccm_float[jj] < MDCCM_THRESH: 
                x_temp = t_float[jj]
                y_temp = tempfmin
                # MdCCM Plot 
                rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_mdccm[jj], alpha=0.5)
                ax1.add_patch(rect)


        # Plot dropped station pairs from LTS if given.
        ax6.set_ylabel('Element [#]', fontsize=fonts+2, fontweight='bold')
        ax6.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold')
        ax6.set_xlim(t_float[0], t_float[-1])
        ax6.xaxis_date()
        ax6.tick_params(axis='x', labelbottom='on')
        ax6.set_title('g)', loc='left', fontsize=fonts+2, fontweight='bold')
        if ALPHA == 1.0:
            print('You used ALPHA = 1.0. It would be better to use "narrow_band_stau_plot".')
        elif ALPHA < 1.0: 
            ndict = deepcopy(stdict)
            band_num = str(ii+1).zfill(2)
            temp_dict = {}
            for key in ndict:
                if key != 'size':
                    if key[0:2] == band_num:
                        new_key = key[3:]
                        temp_dict[new_key] = ndict[key]
                elif key == 'size':
                    temp_dict[key] = ndict[key] 

            n = temp_dict['size']
            temp_dict.pop('size', None)
            tstamps = list(temp_dict.keys())
            tstampsfloat = [float(ii) for ii in tstamps]

            # Set the second colormap for station pairs.
            cm2 = plt.get_cmap('binary', (n-1))
            initplot = np.empty(len(t_float))
            initplot.fill(1)

            ax6.scatter(np.array([t_float[0], t_float[-1]]), np.array([0.01, 0.01]), c='w')
            ax6.axis('tight')
            ax6.set_ylim(0.5, n+0.5)
            ax6.set_xlim(t_float[0], t_float[-1])
        

            # 'tstampsfloat' values are rounded to 7 places after the decimal because it was saved as a str in dict
            # round 't_float' to the same so the values can be compared for MdCCM check
            t_float_round = []
            for jj in range(len(t_float)):
                t_float_round.append(float(format(t_float[jj],'.7f')))

            # Loop through the stdict for each flag and plot
            for jj in range(len(tstamps)):
                # Check if the MdCCM is surpassed for this time; plot only if it is
                ind = t_float_round.index(tstampsfloat[jj])
                if mdccm_float[ind] >= MDCCM_THRESH:
                    z = Counter(list(temp_dict[tstamps[jj]]))
                    keys, vals = z.keys(), z.values()
                    keys, vals = np.array(list(keys)), np.array(list(vals))
                    pts = np.tile(tstampsfloat[jj], len(keys))
                    sc2 = ax6.scatter(pts, keys, c=vals, edgecolors='k', lw=0.1, cmap=cm2, vmin=0.5, vmax=n-0.5)

    if ALPHA < 1.0:
        # Add the colorbar for station pairs.
        ax7 = plt.subplot(gs[6,1])  # Colorbar; Dropped stations
        plt.colorbar(sc2, orientation="vertical", cax=ax7)
        ax7.set_ylabel('# of Flagged\nElement Pairs', fontsize=fonts+2, fontweight='bold')


    #####################
    ### Add colorbars ###
    #####################

    # Add colorbar to mdccm plot
    cax = plt.subplot(gs[1,1]) 
    cb1 = cbar.ColorbarBase(cax, cmap=pl.cm.YlGnBu,norm=normal_mdccm,orientation='vertical') 
    cax.set_ylabel('MdCCM', fontsize=fonts+2, fontweight='bold')


    # Add colorbar to backazimuth plot
    cax = plt.subplot(gs[2,1]) 
    cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.turbo,norm=normal_baz,orientation='vertical', ticks=[0,90,180,270,360]) 
    cax.set_ylabel('Backazimuth\n[deg]', fontsize=fonts+2, fontweight='bold')

    # Add colorbar to trace velocity plot
    cax = plt.subplot(gs[3,1]) 
    cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.turbo,norm=normal_vel,orientation='vertical') 
    cax.set_ylabel('Trace Velocity\n[km/s]', fontsize=fonts+2, fontweight='bold')

    # Add colorbar to scatter plot
    cbaxes = plt.subplot(gs[4:6,1]) 
    if 'sc' in locals():
        hc = plt.colorbar(sc, cax=cbaxes,orientation='vertical') 
    cbaxes.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')



    ###################
    ### Format axes ###
    ###################

    # Pressure plot 
    ax0.xaxis_date()
    ax0.set_xlim(timevec[1], timevec[-1])
    ax0.set_ylabel('Pressure [Pa]', fontsize=fonts+2, fontweight='bold') 
    ax0.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax0.set_title('a)', loc='left', fontsize=fonts+2, fontweight='bold')
    #ax0.set_xticklabels([])

    # MdCCM Plot
    ax1.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
    ax1.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax1.set_title('b)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax1.xaxis_date()
    ax1.set_ylim(FMIN,FMAX)
    ax1.set_xlim(t_float[0],t_float[-1])


    # Backazimuth Plot
    ax2.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
    ax2.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax2.set_title('c)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax2.xaxis_date()
    ax2.set_ylim(FMIN,FMAX)
    ax2.set_xlim(t_float[0],t_float[-1])

    # Trace Velocity Plot
    ax3.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
    ax3.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax3.set_title('d)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax3.xaxis_date()
    ax3.set_ylim(FMIN,FMAX)
    ax3.set_xlim(t_float[0],t_float[-1])


    # Scatter Plot
    ax4.set_ylabel('Backazimuth\n[deg]', fontsize=fonts+2, fontweight='bold')  
    ax4.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax4.set_title('e)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax4.xaxis_date()
    ax4.set_ylim(0,360)
    ax4.set_yticks([0,90,180,270,360])
    ax4.set_xlim(t_float[0],t_float[-1])

    # Scatter Plot
    ax5.set_ylabel('Trace Velocity\n[km/s]', fontsize=fonts+2, fontweight='bold')  
    ax5.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold') 
    ax5.set_title('f)', loc='left', fontsize=fonts+2, fontweight='bold')
    ax5.xaxis_date()
    ax5.set_ylim(0.2,0.5)
    ax5.set_xlim(t_float[0],t_float[-1])


    plt.tight_layout()
    return fig






def narrow_band_lts_dropped_station_plot(FMIN, FMAX, st, NBANDS, freqlist, FREQ_BAND_TYPE, vel_array, baz_array, mdccm_array, t_array, stdict, num_compute_list, MDCCM_THRESH):
    '''
    Plots the dropped stations results for narrow-band least-squares processing
    Args:
        FMIN: Minimum frequency [float] [Hz]
        FMAX: Maximum frequency [float] [Hz]
        st: Filtered data. Assumes response has been removed. (:class:`~obspy.core.stream.Stream`)
        NBANDS: number of frequency bands [integer]
        freqlist: List of frequency bounds for narrow-band processing
        vel_array: array of trace velocity processing results
        baz_array: array of backazimuth processing results
        mdccm_array: array of MdCCM processing results
        t_array: array of times for processing results
        stdict: dictionary with dropped elements for LTS [dictionary]
        num_compute_list: list of number of windows for each frequency band array processing
        MDCCM_THRESH: Threshold value of MdCCM for plotting; Must be between 0 and 1 [float]
    Returns:
        fig: Figure handle (:class:`~matplotlib.figure.Figure`)
    '''
    timevec = st[0].times('matplotlib') # Time vector for plotting
    cm = 'turbo'
    cm_mdccm = 'YlGnBu'
    cax = (FMIN, FMAX)

    # Find number of elements in array
    num_sta = stdict['size']

    fig = plt.figure(figsize=(15,20), dpi=dpi_num)
    gs = gridspec.GridSpec(num_sta,2, width_ratios=[3,0.1])

    for ii in range(NBANDS):
        # Check if overlapping bands
        if FREQ_BAND_TYPE == '2_octave_over':
            tempfmin = freqlist[ii]
            tempfmax = freqlist[ii+2]
        # All others
        else:
            tempfmin = freqlist[ii]
            tempfmax = freqlist[ii+1]
        height_temp = tempfmax - tempfmin # height of frequency rectangles
        tempfavg = tempfmin + (height_temp/2)        # center point of the frequency interval

        # Gather array processing results for this narrow frequency band
        t_temp = t_array[ii,:]
        mdccm_temp = mdccm_array[ii,:]

        # Trim each vector to ignore NAN and zero values
        t_float = t_temp[:num_compute_list[ii]]
        mdccm_float = mdccm_temp[:num_compute_list[ii]]

        # Dropped station pairs from LTS for this specific band 
        ndict = deepcopy(stdict)
        band_num = str(ii+1).zfill(2)
        temp_dict = {}
        for key in ndict:
            if key != 'size':
                if key[0:2] == band_num:
                    new_key = key[3:]
                    temp_dict[new_key] = ndict[key]
            elif key == 'size':
                temp_dict[key] = ndict[key] 

        n = temp_dict['size']
        temp_dict.pop('size', None)
        tstamps = list(temp_dict.keys())
        tstampsfloat = [float(jj) for jj in tstamps]

        # Set the second colormap for station pairs.
        cm2 = plt.get_cmap('binary', (n-1))
        initplot = np.empty(len(t_float))
        initplot.fill(1)

        # Rectangle colormap
        normal_element = pl.Normalize(0.5, n-0.5)

        # Plot dropped station pairs from LTS 
        for kk in range(num_sta):
            ax = plt.subplot(gs[kk,0])
            ax.scatter(np.array([t_float[0], t_float[-1]]), np.array([0.01, 0.01]), c='w')
            # Format axes
            ax.set_xlabel('Time [UTC]', fontsize=fonts+2, fontweight='bold')
            ax.set_xlim(t_float[0], t_float[-1])
            ax.xaxis_date()
            ax.tick_params(axis='x', labelbottom='on')
            ax.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')
            ax.set_ylim(FMIN, FMAX)
            ax.set_title('Element ' + str(kk+1), loc='left', fontsize=fonts+2, fontweight='bold')

        # 'tstampsfloat' values are rounded to 7 places after the decimal because it was saved as a str in dict
        # round 't_float' to the same so the values can be compared for MdCCM check
        t_float_round = []
        for jj in range(len(t_float)):
            t_float_round.append(float(format(t_float[jj],'.7f')))

        # Loop through the stdict for each flag and plot
        for jj in range(len(tstamps)):
            # Check if the MdCCM is surpassed for this time; plot only if it is
            ind = t_float_round.index(tstampsfloat[jj])
            if mdccm_float[ind] >= MDCCM_THRESH:
                z = Counter(list(temp_dict[tstamps[jj]]))
                keys, vals = z.keys(), z.values()
                keys, vals = np.array(list(keys)), np.array(list(vals))
                pts = np.tile(tstampsfloat[jj], len(keys))
                for ll in range(len(keys)):
                    subpanel_num = keys[ll] - 1
                    ax = plt.subplot(gs[subpanel_num,0])
                    # Plot by band number
                    if t_float[ind] == t_float[-1]:
                        width_temp = t_float[ind] - t_float[ind-1]
                    else:
                        width_temp = t_float[ind+1] - t_float[ind]
                    x_temp = tstampsfloat[jj]
                    y_temp = tempfmin
                    # My normalization seems to move the color up one block so subtract 1 for now...
                    rect = Rectangle((x_temp, y_temp), width_temp, height_temp, facecolor=cm2(vals[ll]-1), edgecolor='k', linewidth=0.1)
                    sc2 = ax.add_patch(rect)

    # Add the colorbar for station pairs.
    axc = plt.subplot(gs[0:num_sta+1,1])  # Colorbar; Dropped stations
    cb2 = cbar.ColorbarBase(axc, cmap=cm2,norm=normal_element) 
    axc.set_ylabel('# of Flagged Element Pairs', fontsize=fonts+2, fontweight='bold')


    plt.tight_layout()
    return fig








def baz_freq_plot(FMIN, FMAX, NBANDS, freqlist, vel_array, baz_array, mdccm_array, t_array, num_compute_list, MDCCM_THRESH):
    '''
    Plots the backazimuth through time colored by frequency for narrow-band least-squares processing
    Optimized for weeks/months array processing
    Args:
        FMIN: Minimum frequency [float] [Hz]
        FMAX: Maximum frequency [float] [Hz]
        NBANDS: number of frequency bands [integer]
        freqlist: List of frequency bounds for narrow-band processing
        vel_array: array of trace velocity processing results
        baz_array: array of backazimuth processing results
        mdccm_array: array of MdCCM processing results
        t_array: array of times for processing results
        num_compute_list: list of number of windows for each frequency band array processing
        MDCCM_THRESH: Threshold value of MdCCM for plotting; Must be between 0 and 1 [float]
    Returns:
        fig: Figure handle (:class:`~matplotlib.figure.Figure`)
    '''
    cm = 'jet'
    cax = (FMIN, FMAX)

    fig = plt.figure(figsize=(15,7), dpi=dpi_num)
    gs = gridspec.GridSpec(1,2, width_ratios=[3,0.1], height_ratios=[1])

    # Initialize other plots
    ax1 = plt.subplot(gs[0,0])  # Scatter Plot


    for ii in range(NBANDS):

        # Frequency band info
        tempfmin = freqlist[ii]
        tempfmax = freqlist[ii+1]
        height_temp = tempfmax - tempfmin # height of frequency rectangles
        tempfavg = tempfmin + (height_temp/2)        # center point of the frequency interval

        # Gather array processing results for this narrow frequency band
        vel_temp = vel_array[ii,:]
        baz_temp = baz_array[ii,:]
        mdccm_temp = mdccm_array[ii,:]
        t_temp = t_array[ii,:]

        # Trim each vector to ignore NAN and zero values
        vel_float = vel_temp[:num_compute_list[ii]]
        baz_float = baz_temp[:num_compute_list[ii]]
        mdccm_float = mdccm_temp[:num_compute_list[ii]]
        t_float = t_temp[:num_compute_list[ii]]

        
        # Find indices where mdccm_float >= MDCCM_THRESH
        mdccm_good_idx = [jj for jj,v in enumerate(mdccm_float) if v > MDCCM_THRESH]
        # Trim array to only have the indices where mdccm_float >= MDCCM_THRESH
        vel_good_temp = [vel_float[jj] for jj in mdccm_good_idx]
        baz_good_temp = [baz_float[jj] for jj in mdccm_good_idx]
        t_good_temp = [t_float[jj] for jj in mdccm_good_idx]


        # Find indices where 0.25 < vel <0.45
        vel_good_idx = [jj for jj,v in enumerate(vel_good_temp) if (v > 0.25 and v < 0.45)]
        # Trim array to only have the indices where 0.25 < vel <0.45
        baz_good = [baz_good_temp[jj] for jj in vel_good_idx]
        t_good = [t_good_temp[jj] for jj in vel_good_idx]

        # Plot the scatter points
        tempfavg_array = np.repeat(tempfavg, len(t_good))
        sc = ax1.scatter(t_good, baz_good, s=5, c=tempfavg_array, edgecolors='none', cmap=cm)
        sc.set_clim(cax)


    #####################
    ### Add colorbars ###
    #####################
    # Add colorbar to scatter plot
    cbaxes = plt.subplot(gs[0,1]) 
    hc = plt.colorbar(sc, cax=cbaxes,orientation='vertical') 
    cbaxes.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')


    ###################
    ### Format axes ###
    ###################
    # Scatter Plot
    ax1.set_ylabel('Backazimuth [deg]', fontsize=fonts+2, fontweight='bold')  
    ax1.set_xlabel('Time', fontsize=fonts+2, fontweight='bold') 
    ax1.xaxis_date()
    ax1.set_ylim(0,360)
    ax1.set_xlim(t_float[0],t_float[-1])


    plt.tight_layout()
    return fig

