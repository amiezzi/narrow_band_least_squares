
import numpy as np
from helpers import filter_data, make_float
import multiprocessing
from scipy import signal
from lts_array import ltsva 




def narrow_band_least_squares(WINLEN_list, WINOVER, ALPHA, st, rij, nbands, w, h, freqlist, freq_resp_list, filter_type, filter_order, filter_ripple):
	'''
	Runs narrow band least squares processing
	Args:
		WINLEN_list: list of window length for narrow band processing
		WINOVER: window overlap [float]
		ALPHA: Use ordinary least squares processing (not trimmed least squares)
		st: array data (:class:`~obspy.core.stream.Stream`)
		rij: array coordinates
		nbands: number of frequency bands [integer]
		w: The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the range [0, pi) (radians/sample) [ndarray]
		h: The frequency response, as complex numbers. [ndarray]
		freqlist: list of narrow frequency band limits
		freq_resp_list: list for computing filter frequency response
		filter_type: filter type [string]
		filter_order: filter order [integer]
		filter_ripple: filter ripple (if Chebyshev I filter) [float]
	Returns:
		vel_array: numpy array with trace velocity results 
		baz_array: numpy array with backazimuth results 
		mdccm_array: numpy array with mdccm results 
		t_array: numpy array with times for array processing results 
		num_compute_list: length for processing reults in each frequency band
		w_array: The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the range [0, pi) (radians/sample) [ndarray]
		h_array: The frequency response, as complex numbers. [ndarray]
		
	'''
	###############################
	### Initialize Numpy Arrays ###
	###############################
	max_WINLEN = WINLEN_list[-1]
	sampinc = int((1-WINOVER)*max_WINLEN)
	npts = len(st[0].data)
	its = np.arange(0,npts,sampinc)
	nits = len(its)-1
	Fs = st[0].stats.sampling_rate
	vector_len = int(nits/Fs)

	# Initialize arrays to be as large as the number of windows for the highest frequency band
	vel_array = np.empty((nbands,vector_len))
	baz_array = np.empty((nbands,vector_len))
	mdccm_array = np.empty((nbands,vector_len))
	t_array = np.empty((nbands,vector_len))

	# Initialize Frequency response arrays
	w_array = np.empty((nbands,len(w)), dtype = 'complex_')
	h_array = np.empty((nbands,len(h)), dtype = 'complex_')

	'''
	# Parallel Processing
	num_cores = multiprocessing.cpu_count()
	print(num_cores)
	'''

	########################################
	### Run Narrow Band Array Processing ###
	########################################
	num_compute_list = []
	for ii in range(nbands): 
		tempfmin = freqlist[ii]
		tempfmax = freqlist[ii+1]

		tempst_filter, Fs, sos = filter_data(st, filter_type, tempfmin, tempfmax, filter_order, filter_ripple)
		w, h = signal.sosfreqz(sos,freq_resp_list,fs=Fs)
		w_array[ii,:] = w
		h_array[ii,:] = h


	    # Run Array Processing 
		vel, baz, t, mdccm, stdict, sig_tau = ltsva(tempst_filter, rij, WINLEN_list[ii], WINOVER, ALPHA)

	    # Convert array processing output to numpy array of floats
		vel_float = make_float(vel)
		baz_float = make_float(baz)
		mdccm_float = make_float(mdccm)
		t_float = make_float(t)


	    ####################################
	    ### Save Array Processing Output ###
	    ####################################
		vel_array[ii,:len(vel_float)] = vel_float
		baz_array[ii,:len(baz_float)] = baz_float
		mdccm_array[ii,:len(mdccm_float)] = mdccm_float
		t_array[ii,:len(t_float)] = t_float
		num_compute_list.append(len(vel_float))

	return vel_array, baz_array, mdccm_array, t_array, num_compute_list, w_array, h_array


