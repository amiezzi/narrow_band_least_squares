import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import pylab as pl
from matplotlib.patches import Rectangle
import matplotlib.colorbar as cbar


dpi_num = 300
fonts = 14                                 
rcParams.update({'font.size': fonts})


def broad_filter_response_plot(w, h, FMIN, FMAX, filter_type, filter_order, filter_ripple):
	'''
	Plots the filter frequency response for standard least squares processing
	Args:
		w: The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the range [0, pi) (radians/sample) [ndarray]
		h: The frequency response, as complex numbers. [ndarray]
		FMIN: Minimum frequency [float] [Hz]
		FMAX: Maximum frequency [float] [Hz]
		filter_type: filter type [string]
		filter_order: filter order [integer]
		filter_ripple: filter ripple (if Chebyshev I filter) [float]
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
	ax0.text(0.02, 0.05, 'Filter Type = ' + filter_type, transform=ax0.transAxes)
	ax0.text(0.02, 0.1, 'Filter Order = ' + str(filter_order), transform=ax0.transAxes)
	if filter_type == 'cheby1':
	    ax0.text(0.02, 0.15, 'Ripple = ' + str(filter_ripple), transform=ax0.transAxes)
	plt.tight_layout()
	return fig



def processing_parameters_plot(rij, freqlist, WINLEN_list, nbands, FMIN, FMAX, w_array, h_array, filter_type, filter_order, filter_ripple):
	'''
	Plots the processing parameters for narrow band least squares processing
	Args:
		rij: Coordinates of sensors as eastings & northings in a ``(2, N)`` array [km]
		freqlist: List of frequency bounds for narrow band processing
		WINLEN_list: list of window lengths for each narrow frequency band
		nbands: number of frequency bands [integer]
		w_array: The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the range [0, pi) (radians/sample) [ndarray]
		h_array: The frequency response, as complex numbers. [ndarray]
		FMIN: Minimum frequency [float] [Hz]
		FMAX: Maximum frequency [float] [Hz]
		filter_type: filter type [string]
		filter_order: filter order [integer]
		filter_ripple: filter ripple (if Chebyshev I filter) [float]
	Returns:
		fig: Figure handle (:class:`~matplotlib.figure.Figure`)
	'''
	height = []
	for ii in range(nbands):
		height.append(freqlist[ii+1]- freqlist[ii])

	fig = plt.figure(figsize=(10,10), dpi=dpi_num)
	gs = gridspec.GridSpec(2,2)

	ax0 = plt.subplot(gs[0,0])
	ax0 = plt.subplot(gs[0,0]) 
	ax0.scatter(rij[0], rij[1]) 
	ax0.set_xlabel('X [km]', fontsize=fonts+2, fontweight='bold')
	ax0.set_ylabel('Y [km]', fontsize=fonts+2, fontweight='bold')
	ax0.axis('square')
	ax0.grid()
	ax0.set_title('a) Array Geometry', loc='left', fontsize=fonts+2, fontweight='bold')

	ax1 = plt.subplot(gs[0,1]) 
	ax1.barh(freqlist[:-1], WINLEN_list, height=height, align='edge', color='grey', edgecolor='k')
	#ax0.scatter(freqlist[:-1], WINLEN_list)
	ax1.set_xlabel('Window Length [s]',fontsize=fonts+2, fontweight='bold')
	ax1.set_ylabel('Frequency [Hz]',fontsize=fonts+2, fontweight='bold')
	ax1.set_title('b) Window Length', loc='left', fontsize=fonts+2, fontweight='bold')
	ax1.text(0.02, 0.95, '# of Bands = ' + str(nbands), transform=ax1.transAxes, horizontalalignment='left', fontsize=fonts-2)
	ax1.text(0.98, 0.95, 'FMIN = ' + str(FMIN) + ', FMAX = ' + str(FMAX), transform=ax1.transAxes, horizontalalignment='right', fontsize=fonts-2)
	ax1.set_ylim(-0.1,FMAX+1)

	ax2 = plt.subplot(gs[1,0:2]) 
	for ii in range(nbands):
	    temp_w = w_array[ii,:-1]
	    temp_h = h_array[ii,:-1]
	    ax2.semilogx(temp_w, 20 * np.log10(abs(temp_h)))
	    ax2.axvline(x=freqlist[ii], ymax=0.9, color='k', ls='--')
	ax2.axvline(x=freqlist[-1], ymax=0.9, color='k', ls='--')
	ax2.set_ylabel('Amplitude [dB]', fontsize=fonts+2, fontweight='bold')
	ax2.set_xlabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')
	ax2.set_xlim(FMIN-0.01,FMAX+1)
	ax2.set_ylim(-3,0.4)
	ax2.set_title('c) Narrow Band Filters', loc='left', fontsize=fonts+2, fontweight='bold')
	ax2.text(0.02, 0.95, 'Filter Type = ' + filter_type, transform=ax2.transAxes, horizontalalignment='left', fontsize=fonts-2)
	ax2.text(0.98, 0.95, 'Filter Order = ' + str(filter_order), transform=ax2.transAxes, horizontalalignment='right', fontsize=fonts-2)
	if filter_type == 'cheby1':
	    ax2.text(0.5, 0.95, 'Ripple = ' + str(filter_ripple), transform=ax2.transAxes, horizontalalignment='center', fontsize=fonts-2)
	plt.tight_layout()
	return fig






def pmcc_like_plot(FMIN, FMAX, st, nbands, freqlist, vel_array, baz_array, mdccm_array, t_array, num_compute_list, mdccm_thresh):
	'''
	Plots the results for narrow band least squares processing
	Args:
		FMIN: Minimum frequency [float] [Hz]
		FMAX: Maximum frequency [float] [Hz]
		st: Filtered data. Assumes response has been removed. (:class:`~obspy.core.stream.Stream`)
		nbands: number of frequency bands [integer]
		freqlist: List of frequency bounds for narrow band processing
		vel_array: array of trace velocity processing results
		baz_array: array of backazimuth processing results
		mdccm_array: array of MdCCM processing results
		t_array: array of times for processing results
		num_compute_list: list of number of windows for each frequency band array processing
		mdccm_thresh: Threshold value of MdCCM for plotting; Must be between 0 and 1 [float]
	Returns:
		fig: Figure handle (:class:`~matplotlib.figure.Figure`)
	'''
	timevec = st[0].times('matplotlib')	# Time vector for plotting
	cm = 'jet'
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

	for ii in range(nbands):

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

	    # Initialize colorbars
	    #normal = pl.Normalize(baz_float.min(), baz_float.max())
	    normal_baz = pl.Normalize(0, 360)
	    colors_baz = pl.cm.jet(normal_baz(baz_float))

	    #normal_vel = pl.Normalize(vel_float.min(), vel_float.max()) # This re-normalizing for each narrow frequency band (not right)
	    normal_vel = pl.Normalize(0.2,0.5)
	    colors_vel = pl.cm.jet(normal_vel(vel_float))

	    #normal_vel = pl.Normalize(vel_float.min(), vel_float.max()) # This re-normalizing for each narrow frequency band (not right)
	    normal_mdccm = pl.Normalize(0.,1.0)
	    colors_mdccm = pl.cm.jet(normal_mdccm(mdccm_float))


	    # Loop through each narrow band results vector and plot rectangles/scatter points
	    for jj in range(len(t_float)-1):
	        width_temp = t_float[jj+1] - t_float[jj]
	        if mdccm_float[jj] >= mdccm_thresh: 
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

	            # Scatter plot
	            sc = ax4.scatter(t_float[jj], baz_float[jj], c=tempfavg, edgecolors='k', lw=0.3, cmap=cm)
	            sc.set_clim(cax)

	            # Scatter plot
	            sc_vel = ax5.scatter(t_float[jj], vel_float[jj], c=tempfavg, edgecolors='k', lw=0.3, cmap=cm)
	            sc_vel.set_clim(cax)

	    # MdCCM Loop through each narrow band results vector and plot rectangles/scatter points
	    for jj in range(len(t_float)-1):
	        width_temp = t_float[jj+1] - t_float[jj]
	        if mdccm_float[jj] < mdccm_thresh: 
	            x_temp = t_float[jj]
	            y_temp = tempfmin
	            # MdCCM Plot 
	            rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_mdccm[jj], alpha=0.5)
	            ax1.add_patch(rect)


	#####################
	### Add colorbars ###
	#####################

	# Add colorbar to trace velocity plot
	cax = plt.subplot(gs[1,1]) 
	cb1 = cbar.ColorbarBase(cax, cmap=pl.cm.jet,norm=normal_mdccm,orientation='vertical') 
	cax.set_ylabel('MdCCM', fontsize=fonts+2, fontweight='bold')

	# Add colorbar to backazimuth plot
	cax = plt.subplot(gs[2,1]) 
	cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.jet,norm=normal_baz,orientation='vertical', ticks=[0,90,180,270,360]) 
	cax.set_ylabel('Backazimuth [deg]', fontsize=fonts+2, fontweight='bold')

	# Add colorbar to trace velocity plot
	cax = plt.subplot(gs[3,1]) 
	cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.jet,norm=normal_vel,orientation='vertical') 
	cax.set_ylabel('Trace Velocity [km/s]', fontsize=fonts+2, fontweight='bold')

	# Add colorbar to scatter plot
	cbaxes = plt.subplot(gs[4,1]) 
	hc = plt.colorbar(sc, cax=cbaxes,orientation='vertical') 
	cbaxes.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')

	# Add colorbar to scatter plot for trace velocity
	cbaxes_vel = plt.subplot(gs[5,1]) 
	hc_vel = plt.colorbar(sc_vel, cax=cbaxes_vel,orientation='vertical') 
	cbaxes_vel.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')


	###################
	### Format axes ###
	###################
	# Pressure plot 
	ax0.xaxis_date()
	ax0.set_xlim(timevec[0], timevec[-1])
	ax0.set_ylabel('Pressure [Pa]', fontsize=fonts+2, fontweight='bold') 
	ax0.set_xlabel('Time', fontsize=fonts+2, fontweight='bold') 
	ax0.set_title('a)', loc='left', fontsize=fonts+2, fontweight='bold')

	# MdCCM Plot
	ax1.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
	ax1.set_xlabel('Time', fontsize=fonts+2, fontweight='bold') 
	ax1.set_title('b)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax1.xaxis_date()
	ax1.set_ylim(FMIN,FMAX)
	ax1.set_xlim(t_float[0],t_float[-1])

	# Backazimuth Plot
	ax2.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
	ax2.set_xlabel('Time', fontsize=fonts+2, fontweight='bold') 
	ax2.set_title('c)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax2.xaxis_date()
	ax2.set_ylim(FMIN,FMAX)
	ax2.set_xlim(t_float[0],t_float[-1])

	# Trace Velocity Plot
	ax3.set_ylabel('Frequency [Hz]', fontsize=fonts+2, fontweight='bold')  
	ax3.set_xlabel('Time', fontsize=fonts+2, fontweight='bold') 
	ax3.set_title('d)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax3.xaxis_date()
	ax3.set_ylim(FMIN,FMAX)
	ax3.set_xlim(t_float[0],t_float[-1])

	# Scatter Plot
	ax4.set_ylabel('Backazimuth [deg]', fontsize=fonts+2, fontweight='bold')  
	ax4.set_xlabel('Time', fontsize=fonts+2, fontweight='bold') 
	ax4.set_title('e)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax4.xaxis_date()
	ax4.set_ylim(0,360)
	ax4.set_xlim(t_float[0],t_float[-1])

	# Scatter Plot
	ax5.set_ylabel('Trace Velocity [km/s]', fontsize=fonts+2, fontweight='bold')  
	ax5.set_xlabel('Time', fontsize=fonts+2, fontweight='bold') 
	ax5.set_title('f)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax5.xaxis_date()
	ax5.set_ylim(0.25,0.45)
	ax5.set_xlim(t_float[0],t_float[-1])

	plt.tight_layout()
	return fig



