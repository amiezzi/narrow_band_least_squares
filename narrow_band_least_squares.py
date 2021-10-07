
import numpy as np
from helpers import filter_data, make_float
from scipy import signal
from lts_array import ltsva 
import multiprocessing
from joblib import Parallel, delayed


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


	########################################
	### Run Narrow Band Array Processing ###
	########################################
	num_compute_list = []
	#num_compute_list = np.zeros((nbands))
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
		#num_compute_list[ii]=len(vel_float)

	return vel_array, baz_array, mdccm_array, t_array, num_compute_list, w_array, h_array






def narrow_band_loop(ii, freqlist, freq_resp_list, st, filter_type, filter_order, filter_ripple, rij, WINLEN_list, WINOVER, ALPHA, vector_len):
	'''
	Loop designed for narrow band least squares processing parallelization
	Args:
		ii: index for narrow frequency band to be used
		freqlist: list of narrow frequency band limits
		freq_resp_list: list for computing filter frequency response
		st: array data (:class:`~obspy.core.stream.Stream`)
		filter_type: filter type [string]
		filter_order: filter order [integer]
		filter_ripple: filter ripple (if Chebyshev I filter) [float]
		rij: array coordinates
		WINLEN_list: list of window length for narrow band processing
		WINOVER: window overlap [float]
		ALPHA: Use ordinary least squares processing (not trimmed least squares)
		vector_len: number of windows for the highest frequency band
	Returns:
		vel_float: trace velocity results for that frequency band
		baz_float: backazimuth results for that frequency band
		mdccm_float: mdccm results for that frequency band
		t_float: time results for that frequency band
		num_compute: number of windows computed for that narrow frquency band
		w_temp: The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the range [0, pi) (radians/sample) [ndarray]
		h_temp: The frequency response, as complex numbers. [ndarray]	
	'''
	tempfmin = freqlist[ii]
	tempfmax = freqlist[ii+1]

	tempst_filter, Fs, sos = filter_data(st, filter_type, tempfmin, tempfmax, filter_order, filter_ripple)
	w_temp, h_temp = signal.sosfreqz(sos,freq_resp_list,fs=Fs)

    # Run Array Processing 
	vel, baz, t, mdccm, stdict, sig_tau = ltsva(tempst_filter, rij, WINLEN_list[ii], WINOVER, ALPHA)

    # Convert array processing output to numpy array of floats
	vel_float = make_float(vel)
	baz_float = make_float(baz)
	mdccm_float = make_float(mdccm)
	t_float = make_float(t)


    ###################################
    ### Pad Array Processing Output ###
    ###################################
	num_compute = np.array(len(vel_float))
	vel_float = np.pad(vel_float, (0,vector_len-num_compute))
	baz_float = np.pad(baz_float, (0,vector_len-num_compute))
	mdccm_float = np.pad(mdccm_float, (0,vector_len-num_compute))
	t_float = np.pad(t_float, (0,vector_len-num_compute))


	return vel_float, baz_float, mdccm_float, t_float, num_compute, w_temp, h_temp




def narrow_band_least_squares_parallel(WINLEN_list, WINOVER, ALPHA, st, rij, nbands, w, h, freqlist, freq_resp_list, filter_type, filter_order, filter_ripple):
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
	vel_array = np.zeros((nbands,vector_len))
	#print(vel_array.shape)
	baz_array = np.zeros((nbands,vector_len))
	mdccm_array = np.zeros((nbands,vector_len))
	t_array = np.zeros((nbands,vector_len))

	# Initialize Frequency response arrays
	w_array = np.zeros((nbands,len(w)), dtype = 'complex_')
	h_array = np.zeros((nbands,len(h)), dtype = 'complex_')

	#num_compute_list = np.zeros((nbands))
	num_compute_list = []
	
	########################################
	### Run Narrow Band Array Processing ###
	########################################
	# Parallel Processing
	#num_cores = int(multiprocessing.cpu_count()/2)
	#print(num_cores)
	results = Parallel(n_jobs=-1)(delayed(narrow_band_loop)(ii, freqlist, freq_resp_list, st, filter_type, filter_order, filter_ripple, rij, WINLEN_list, WINOVER, ALPHA, vector_len) for ii in range(nbands))


	###################################################################
	### Transform results of parallelization to numpy array outputs ###
	###################################################################
	for jj in range(nbands):
		vel_array[jj,:] = results[jj][0]
		baz_array[jj,:] = results[jj][1]
		mdccm_array[jj,:] = results[jj][2]
		t_array[jj,:] = results[jj][3]
		num_compute_list.append(int(results[jj][4]))
		w_array[jj,:] = results[jj][5]
		h_array[jj,:] = results[jj][6]


	return vel_array, baz_array, mdccm_array, t_array, num_compute_list, w_array, h_array
	


