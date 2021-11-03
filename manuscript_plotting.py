import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib import rcParams
import pylab as pl
from matplotlib.patches import Rectangle
import matplotlib.colorbar as cbar


dpi_num = 300
fonts = 14                                 
rcParams.update({'font.size': fonts})


def array_processing_combined_plot(FMIN, FMAX, st_broad, t_broad, mdccm_broad, vel_broad, baz_broad, sigma_tau_broad, nbands, freqlist, vel_array, baz_array, mdccm_array, t_array, num_compute_list, mdccm_thresh, true_baz):
	'''
	Plots the results for standard and narrow band least squares processing
	Args:
		FMIN: Minimum frequency [float] [Hz]
		FMAX: Maximum frequency [float] [Hz]
		st_broad: Filtered data. Assumes response has been removed. (:class:`~obspy.core.stream.Stream`)
		t_broad: Array processing time vector.
        mdccm_broad: Array of median cross-correlation maxima.
        vel_broad: Array of trace velocity estimates.
        baz_broad: Array of back-azimuth estimates.
        sigma_tau_broad: Array of :math:`sigma_tau` values. If provided, will plot
            the values on a separate subplot.
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
	cm = 'BuPu'
	cm = 'jet'
	#cm = 'RdYlBu_r'
	cax_narrow = (FMIN, FMAX)
	cax_broad = (0.2, 1) # Colorbar/y-axis limits for MdCCM

	# Specify the time vector for plotting the trace.
	tvec = st_broad[0].times('matplotlib')


	fig = plt.figure(figsize=(20,15), dpi=dpi_num)
	gs = gridspec.GridSpec(6,5, width_ratios=[4,0.1,0.5,4, 0.1], height_ratios=[1,1,1,1,1,1])

	#################################
	### Standard Array Processing ###
	#################################
	# Pressure Plot
	ax00 = plt.subplot(gs[0,0])
	ax00.plot(tvec, st_broad[0].data, 'k')
	ax00.set_ylabel('Pressure\n[Pa]', fontsize=fonts, fontweight='bold')
	ax00.xaxis_date()
	ax00.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	ax00.set_xlim(tvec[1], tvec[-1])
	ax00.set_title('(a)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax00.set_xlabel('Time', fontsize=fonts, fontweight='bold') 

	'''
	# Spectrogram
	ax10 = plt.subplot(gs[1,0])
	ax10.set_title('b)', loc='left', fontsize=fonts+2, fontweight='bold')
	'''

	# MdCCM Plot
	ax20 = plt.subplot(gs[1,0])
	sc = ax20.scatter(t_broad, mdccm_broad, c=mdccm_broad,edgecolors='k', lw=0.3, cmap=cm)
	ax20.plot([t_broad[0], t_broad[-1]], [mdccm_thresh, mdccm_thresh], 'k--')
	ax20.set_xlim(t_broad[0], t_broad[-1])
	ax20.xaxis_date()
	ax20.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	ax20.set_ylim(cax_broad)
	sc.set_clim(cax_broad)
	ax20.set_ylabel('MdCCM', fontsize=fonts, fontweight='bold')
	ax20.set_title('(b)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax20.set_xlabel('Time', fontsize=fonts, fontweight='bold') 

	'''
	# Plot sigma_tau if given.
	ax30 = plt.subplot(gs[3,0])
	if sigma_tau_broad is not None:
		if np.isnan(np.sum(sigma_tau_broad)):
			print(r'Sigma_tau values are NaN!')
			ax30.scatter(np.array([t_broad_broad[0], t_broad[-1]]),
								np.array([0.01, 0.01]), c='w')
		else:
			sc = ax30.scatter(t_broad, sigma_tau_broad, c=mdccm_broad,
								edgecolors='k', lw=0.3, cmap=cm)
		ax30.set_xlim(t_broad[0], t_broad[-1])
		ax30.xaxis_date()
		sc.set_clim(cax_broad)
		ax30.set_ylabel(r'$\sigma_\tau$', fontsize=fonts, fontweight='bold')
		ax30.set_title('d)', loc='left', fontsize=fonts+2, fontweight='bold')

	'''


	# Plot the back-azimuth.
	ax40 = plt.subplot(gs[3,0])
	if true_baz > 0:
		ax40.plot([t_broad[0], t_broad[-1]], [true_baz, true_baz], 'k--')
	sc = ax40.scatter(t_broad, baz_broad, c=mdccm_broad, edgecolors='k', lw=0.3, cmap=cm)
	ax40.set_ylim(0, 360)
	ax40.set_xlim(t_broad[0], t_broad[-1])
	ax40.xaxis_date()
	ax40.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	sc.set_clim(cax_broad)
	ax40.set_ylabel('Backazimuth\n[deg]', fontsize=fonts, fontweight='bold')
	ax40.set_title('(c)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax40.set_xlabel('Time', fontsize=fonts, fontweight='bold') 

	# Plot the trace/apparent velocity.
	ax50 = plt.subplot(gs[5,0])
	sc = ax50.scatter(t_broad, vel_broad, c=mdccm_broad, edgecolors='k', lw=0.3, cmap=cm)
	ax50.set_ylim(0.25, 0.45)
	ax50.set_xlim(t_broad[0], t_broad[-1])
	ax50.xaxis_date()
	ax50.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	sc.set_clim(cax_broad)
	ax50.set_ylabel('Trace Velocity\n[km/s]', fontsize=fonts, fontweight='bold')
	ax50.set_title('(d)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax50.set_xlabel('Time', fontsize=fonts, fontweight='bold') 


	# Add the MdCCM colorbar.
	cbaxes = plt.subplot(gs[1,1]) 
	hc = plt.colorbar(sc, cax=cbaxes)
	hc.set_label('MdCCM', fontsize=fonts, fontweight='bold')

	# Add the MdCCM colorbar.
	cbaxes = plt.subplot(gs[3,1]) 
	hc = plt.colorbar(sc, cax=cbaxes)
	hc.set_label('MdCCM', fontsize=fonts, fontweight='bold')

	# Add the MdCCM colorbar
	cbaxes = plt.subplot(gs[5,1]) 
	hc = plt.colorbar(sc, cax=cbaxes)
	hc.set_label('MdCCM', fontsize=fonts, fontweight='bold')




    #################################
	### Narrow Band Least Squares ###
	#################################
	cm = 'jet'
	# Pressure plot (broadband bandpass filtered data)
	ax0 = plt.subplot(gs[0,3])  # Pressure Plot
	ax0.plot(tvec, st_broad[0], 'k') # plots pressure for the first band

	# Initialize other plots
	ax1 = plt.subplot(gs[1,3])  # MdCCM Plot
	ax2 = plt.subplot(gs[2,3])  # Backazimuth Plot
	ax3 = plt.subplot(gs[3,3])  # Scatter Plot
	ax4 = plt.subplot(gs[4,3])  # Trace Velocity Plot
	ax5 = plt.subplot(gs[5,3])  # Scatter Plot Trace Velocity

	if true_baz > 0:
		ax3.plot([t_broad[0], t_broad[-1]], [true_baz, true_baz], 'k--')


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
	    #colors_mdccm = pl.cm.BuPu(normal_mdccm(mdccm_float))
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

	            # Scatter plot
	            sc = ax3.scatter(t_float[jj], baz_float[jj], c=tempfavg, edgecolors='k', lw=0.3, cmap=cm)
	            sc.set_clim(cax_narrow)

	            # Trace Velocity Plot 
	            rect = Rectangle((x_temp, y_temp), width_temp, height_temp, color=colors_vel[jj])
	            ax4.add_patch(rect)

	            # Scatter plot
	            sc_vel = ax5.scatter(t_float[jj], vel_float[jj], c=tempfavg, edgecolors='k', lw=0.3, cmap=cm)
	            sc_vel.set_clim(cax_narrow)

	            



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
	cax = plt.subplot(gs[1,4]) 
	#cb1 = cbar.ColorbarBase(cax, cmap=pl.cm.BuPu,norm=normal_mdccm,orientation='vertical') 
	cb1 = cbar.ColorbarBase(cax, cmap=pl.cm.jet,norm=normal_mdccm,orientation='vertical') 
	cax.set_ylabel('MdCCM', fontsize=fonts, fontweight='bold')

	# Add colorbar to backazimuth plot
	cax = plt.subplot(gs[2,4]) 
	cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.jet,norm=normal_baz,orientation='vertical', ticks=[0,90,180,270,360]) 
	cax.set_ylabel('Backazimuth\n[deg]', fontsize=fonts, fontweight='bold')

	# Add colorbar to scatter plot
	cbaxes = plt.subplot(gs[3,4]) 
	hc = plt.colorbar(sc, cax=cbaxes,orientation='vertical') 
	cbaxes.set_ylabel('Frequency [Hz]', fontsize=fonts, fontweight='bold')

	# Add colorbar to trace velocity plot
	cax = plt.subplot(gs[4,4]) 
	cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.jet,norm=normal_vel,orientation='vertical') 
	cax.set_ylabel('Trace Velocity\n[km/s]', fontsize=fonts, fontweight='bold')

	# Add colorbar to scatter plot for trace velocity
	cbaxes_vel = plt.subplot(gs[5,4]) 
	hc_vel = plt.colorbar(sc_vel, cax=cbaxes_vel,orientation='vertical') 
	cbaxes_vel.set_ylabel('Frequency [Hz]', fontsize=fonts, fontweight='bold')




	###################
	### Format axes ###
	###################
	# Pressure plot 
	ax0.xaxis_date()
	ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	ax0.set_xlim(tvec[1], tvec[-1])
	ax0.set_ylabel('Pressure\n[Pa]', fontsize=fonts, fontweight='bold') 
	ax0.set_xlabel('Time', fontsize=fonts, fontweight='bold') 
	ax0.set_title('(e)', loc='left', fontsize=fonts+2, fontweight='bold')

	# MdCCM Plot
	ax1.set_ylabel('Frequency\n[Hz]', fontsize=fonts, fontweight='bold')  
	ax1.set_xlabel('Time', fontsize=fonts, fontweight='bold') 
	ax1.set_title('(f)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax1.xaxis_date()
	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	ax1.set_ylim(FMIN,FMAX)
	ax1.set_xlim(t_float[0],t_float[-1])

	# Backazimuth Plot
	ax2.set_ylabel('Frequency\n[Hz]', fontsize=fonts, fontweight='bold')  
	ax2.set_xlabel('Time', fontsize=fonts, fontweight='bold') 
	ax2.set_title('(g)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax2.xaxis_date()
	ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	ax2.set_ylim(FMIN,FMAX)
	ax2.set_xlim(t_float[0],t_float[-1])

	# Scatter Plot
	ax3.set_ylabel('Backazimuth\n[deg]', fontsize=fonts, fontweight='bold')  
	ax3.set_xlabel('Time', fontsize=fonts, fontweight='bold') 
	ax3.set_title('(h)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax3.xaxis_date()
	ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	ax3.set_ylim(0,360)
	ax3.set_xlim(t_float[0],t_float[-1])


	# Trace Velocity Plot
	ax4.set_ylabel('Frequency\n[Hz]', fontsize=fonts, fontweight='bold')  
	ax4.set_xlabel('Time', fontsize=fonts, fontweight='bold') 
	ax4.set_title('(i)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax4.xaxis_date()
	ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	ax4.set_ylim(FMIN,FMAX)
	ax4.set_xlim(t_float[0],t_float[-1])

	# Scatter Plot
	ax5.set_ylabel('Trace Velocity\n[km/s]', fontsize=fonts, fontweight='bold')  
	ax5.set_xlabel('Time', fontsize=fonts, fontweight='bold') 
	ax5.set_title('(j)', loc='left', fontsize=fonts+2, fontweight='bold')
	ax5.xaxis_date()
	ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	ax5.set_ylim(0.25,0.45)
	ax5.set_xlim(t_float[0],t_float[-1])



	#plt.tight_layout()
	gs.tight_layout(fig, w_pad=-2)
	return fig
