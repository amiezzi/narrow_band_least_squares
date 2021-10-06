import numpy as np
import math as math
from scipy import signal 



def get_freqlist(FMIN, FMAX, freq_band_type, nbands):
	'''
	Obtains list of narrow frequency band limits
	Args:
		FMIN: Minimum frequency [float] [Hz]
		FMAX: Maximum frequency [float] [Hz]
		freq_band_type: `linear' or 'log' for frequency band width [string]
		nbands: number of frequency bands [integer]
	Returns:
		freqlist: list of narrow frequency band limits
	'''
	freqrange = FMAX - FMIN

	if freq_band_type == 'linear':
		freqinterval = freqrange / nbands
		freqlist = np.arange(FMIN, FMAX+freqinterval, freqinterval)
	elif freq_band_type == 'log':
		FMINL = math.log(FMIN, 10)
		FMAXL = math.log(FMAX, 10)
		freqlist = np.logspace(FMINL, FMAXL, num = nbands+1)

	return freqlist



def get_winlenlist(window_length, nbands, WINLEN, WINLEN_1, WINLEN_X):
	'''
	Obtains list of window length for narrow band processing
	Args:
		window_length: 'constant' or 'adaptive' [string]
		nbands: number of frequency bands [integer]
		WINLEN: window length [s]; used if window_length = 'constant' AND if window_length = '1/f' (because of broadband processing)
		WINLEN_1: window length for band 1 (lowest frequency) [s]; only used if window_length = 'adaptive'
		WINLEN_X: window length for band X (highest frequency) [s]; only used if window_length = 'adaptive'
	Returns:
		WINLEN_list: list of window length for narrow band processing
	'''
	if window_length == 'constant':
		WINLEN_list = [] 
		for ii in range(nbands):
			WINLEN_list.append(WINLEN)
	elif window_length == 'adaptive':
		# varies linearly with period
		WINLEN_list = np.linspace(WINLEN_1, WINLEN_X, num=nbands)
		WINLEN_list = [int(item) for item in WINLEN_list]

	return WINLEN_list



def filter_data(st, filter_type, FMIN, FMAX, filter_order, filter_ripple):
	'''
	Filter and taper the data
	Args:
		st: array data (:class:`~obspy.core.stream.Stream`)
		filter_type: filter type; 'butter', 'cheby1' [string]
		FMIN: Minimum frequency [float] [Hz]
		FMAX: Maximum frequency [float] [Hz]
		filter_order: filter order [integer]
		filter_ripple: filter ripple (if Chebyshev I filter) [float]
	Returns:
		stf: Filtered array data (:class:`~obspy.core.stream.Stream`)
		Fs: Sampling Frequency [Hz] [float]
		sos: second-order sections of filter

	'''
	stf = st.copy()
	Fs = stf[0].stats.sampling_rate
	if filter_type == 'butter': 
		stf.filter('bandpass', freqmin = FMIN, freqmax = FMAX, corners=filter_order, zerophase = True)
		sos = signal.iirfilter(filter_order, [FMIN, FMAX], btype='band',ftype='butter', fs=Fs, output='sos')    
	elif filter_type == 'cheby1': 
		Wn = [FMIN, FMAX]
		sos = signal.iirfilter(filter_order, [FMIN, FMAX], rp=filter_ripple, btype='band', analog=False, ftype='cheby1', fs=Fs,output='sos')
		for ii in range(len(st)):
			# Put signal in numpy array
			temp_array = stf[ii].data
			# Filter
			filtered = signal.sosfilt(sos, temp_array)
			# transform signal back to st
			stf[ii].data = filtered

	stf.taper(max_percentage=0.01)    # Taper the waveforms

	return stf, Fs, sos



def make_float(input):
	'''
	Convert array processing output to numpy array of floats
	Args:
		input: array processing output
	Returns:
		float_array: array of floats
	'''
	float_list = []
	for jj in range(len(input)):
		float_list.append(float(input[jj]))
	float_array = np.array(float_list)

	return float_array


def write_txtfile(save_dir, vel_array, baz_array, mdccm_array, t_array, freqlist, num_compute_list):
	'''
	Write array processing results to txt file
	Args:
		save_dir: directory in which to save output file
		vel_array: numpy array with trace velocity results 
		baz_array: numpy array with backazimuth results 
		mdccm_array: numpy array with mdccm results 
		t_array: numpy array with times for array processing results 
		freqlist: list of narrow frequency band limits
		num_compute_list: length for processing reults in each frequency band
	Returns:
		None
	'''
	f = open(save_dir + 'narrow_band_processing_results.txt', 'w')
	f.write('Fmin \t Fmax \t Time \t Trace_vel \t Backaz \t MdCCM \n')
	for ii in range(len(freqlist)-1):
		print((num_compute_list[ii]))
		for jj in range(num_compute_list[ii]):
			f.write(str(freqlist[ii]) + '\t' + str(freqlist[ii+1]) + '\t' + str(t_array[ii,jj]) + '\t' + str(vel_array[ii,jj]) + '\t' + str(baz_array[ii,jj]) + '\t' + str(mdccm_array[ii,jj]) + '\n')
	f.close()


