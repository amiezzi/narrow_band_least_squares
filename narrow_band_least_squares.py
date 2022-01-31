
import numpy as np
from helpers import filter_data, make_float
from scipy import signal
from lts_array import ltsva 


def narrow_band_least_squares(WINLEN_list, WINOVER, ALPHA, st, rij, NBANDS, w, h, freqlist, FREQ_BAND_TYPE, freq_resp_list, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE):
    '''
    Runs narrow-band least-squares processing (not paralellized)
    Args:
        WINLEN_list: list of window length for narrow-band processing
        WINOVER: window overlap [float]
        ALPHA: Use ordinary least-squares or LTS processing 
        st: array data (:class:`~obspy.core.stream.Stream`)
        rij: array coordinates
        NBANDS: number of frequency bands [integer]
        w: The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the range [0, pi) (radians/sample) [ndarray]
        h: The frequency response, as complex numbers. [ndarray]
        freqlist: list of narrow frequency band limits
        FREQ_BAND_TYPE: `linear' or 'log' for frequency band width [string]
        freq_resp_list: list for computing filter frequency response
        FILTER_TYPE: filter type [string]
        FILTER_ORDER: filter order [integer]
        FILTER_RIPPLE: filter ripple (if Chebyshev I filter) [float]
    Returns:
        vel_array: numpy array with trace velocity results 
        baz_array: numpy array with backazimuth results 
        mdccm_array: numpy array with mdccm results 
        t_array: numpy array with times for array processing results 
        stdict_all: dictionary with dropped elements for LTS [dictionary]
        sig_tau_array: numpy array of sigma tau values for array processing results 
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
    vel_array = np.empty((NBANDS,vector_len))
    baz_array = np.empty((NBANDS,vector_len))
    mdccm_array = np.empty((NBANDS,vector_len))
    sig_tau_array = np.empty((NBANDS,vector_len))
    t_array = np.empty((NBANDS,vector_len))
    stdict_all = {}

    # Initialize Frequency response arrays
    w_array = np.empty((NBANDS,len(w)), dtype = 'complex_')
    h_array = np.empty((NBANDS,len(h)), dtype = 'complex_')


    ########################################
    ### Run Narrow-Band Array Processing ###
    ########################################
    num_compute_list = []

    for ii in range(NBANDS): 
        # Check if overlapping bands
        if FREQ_BAND_TYPE == '2_octave_over':
            tempfmin = freqlist[ii]
            tempfmax = freqlist[ii+2]
        # All others
        else:
            tempfmin = freqlist[ii]
            tempfmax = freqlist[ii+1]

        tempst_filter, Fs, sos = filter_data(st, FILTER_TYPE, tempfmin, tempfmax, FILTER_ORDER, FILTER_RIPPLE)
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
        sig_tau_float = make_float(sig_tau)


        ####################################
        ### Save Array Processing Output ###
        ####################################
        vel_array[ii,:len(vel_float)] = vel_float
        baz_array[ii,:len(baz_float)] = baz_float
        mdccm_array[ii,:len(mdccm_float)] = mdccm_float
        t_array[ii,:len(t_float)] = t_float
        num_compute_list.append(len(vel_float))

        if ALPHA == 1.0:
            sig_tau_float = make_float(sig_tau)
            sig_tau_array[ii,:len(sig_tau_float)] = sig_tau_float
            stdict_all = None
        elif ALPHA < 1.0:
            sigma_tau_float = None
            # Append band number to each key in the drop stations dictionary
            temp_dict = {}
            for key in stdict:
                if key != 'size':
                    new_key = str(ii+1).zfill(2) + '_' + key
                    temp_dict[new_key] = stdict[key]
                elif key == 'size':
                    temp_dict[key] = stdict[key]
            stdict_all = {**stdict_all, **temp_dict}


    return vel_array, baz_array, mdccm_array, t_array, stdict_all, sig_tau_array, num_compute_list, w_array, h_array






def narrow_band_loop(ii, freqlist, FREQ_BAND_TYPE, freq_resp_list, st, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE, rij, WINLEN_list, WINOVER, ALPHA, vector_len):
    '''
    Loop designed for narrow-band least-squares processing parallelization
    Args:
        ii: index for narrow frequency band to be used
        freqlist: list of narrow frequency band limits
        FREQ_BAND_TYPE; `linear' or 'log' for frequency band width [string]
        freq_resp_list: list for computing filter frequency response
        st: array data (:class:`~obspy.core.stream.Stream`)
        FILTER_TYPE: filter type [string]
        FILTER_ORDER: filter order [integer]
        FILTER_RIPPLE: filter ripple (if Chebyshev I filter) [float]
        rij: array coordinates
        WINLEN_list: list of window length for narrow-band processing
        WINOVER: window overlap [float]
        ALPHA: Use ordinary least-squares processing (not trimmed least-squares)
        vector_len: number of windows for the highest frequency band
    Returns:
        vel_float: trace velocity results for that frequency band
        baz_float: backazimuth results for that frequency band
        mdccm_float: mdccm results for that frequency band
        t_float: time results for that frequency band
        stdict_times: times for dropped elements for that frequency band (portion of 'stdict')
        stdict_elements: dropped elements for that frequency band (portion of 'stdict')
        sig_tau_float: sigma tau results for that frequency band
        num_compute: number of windows computed for that narrow frquency band
        w_temp: The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the range [0, pi) (radians/sample) [ndarray]
        h_temp: The frequency response, as complex numbers. [ndarray]   
    '''
    if FREQ_BAND_TYPE == '2_octave_over':
        tempfmin = freqlist[ii]
        tempfmax = freqlist[ii+2]
    # All others
    else:
        tempfmin = freqlist[ii]
        tempfmax = freqlist[ii+1]

    tempst_filter, Fs, sos = filter_data(st, FILTER_TYPE, tempfmin, tempfmax, FILTER_ORDER, FILTER_RIPPLE)
    w_temp, h_temp = signal.sosfreqz(sos,freq_resp_list,fs=Fs)

    # Run Array Processing 
    vel, baz, t, mdccm, stdict, sig_tau = ltsva(tempst_filter, rij, WINLEN_list[ii], WINOVER, ALPHA)


    # Convert array processing output to numpy array of floats
    vel_float = make_float(vel)
    baz_float = make_float(baz)
    mdccm_float = make_float(mdccm)
    t_float = make_float(t)
    sig_tau_float = make_float(sig_tau)


    ###################################
    ### Pad Array Processing Output ###
    ###################################
    num_compute = np.array(len(vel_float))
    vel_float = np.pad(vel_float, (0,vector_len-num_compute))
    baz_float = np.pad(baz_float, (0,vector_len-num_compute))
    mdccm_float = np.pad(mdccm_float, (0,vector_len-num_compute))
    t_float = np.pad(t_float, (0,vector_len-num_compute))
    sig_tau_float = np.pad(sig_tau_float, (0,vector_len-num_compute))

    if ALPHA == 1.0:
        stdict_times = None
        stdict_elements = None

    # Deal with 'stdict'
    # Convert to numpy arrays so it can be be parallelized
    elif ALPHA < 1.0:
        temp_data = list(stdict.items())
        temp_array = np.array(temp_data, dtype=object)
        stdict_times = temp_array[:,0]
        stdict_elements = temp_array[:,1]



    return vel_float, baz_float, mdccm_float, t_float, stdict_times, stdict_elements, sig_tau_float, num_compute, w_temp, h_temp




def narrow_band_least_squares_parallel(WINLEN_list, WINOVER, ALPHA, st, rij, NBANDS, w, h, freqlist, FREQ_BAND_TYPE, freq_resp_list, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE):
    from joblib import Parallel, delayed
    '''
    Runs narrow-band least-squares processing in parallel
    Args:
        WINLEN_list: list of window length for narrow-band processing
        WINOVER: window overlap [float]
        ALPHA: Use ordinary least-squares or LTS processing 
        st: array data (:class:`~obspy.core.stream.Stream`)
        rij: array coordinates
        NBANDS: number of frequency bands [integer]
        w: The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the range [0, pi) (radians/sample) [ndarray]
        h: The frequency response, as complex numbers. [ndarray]
        freqlist: list of narrow frequency band limits
        FREQ_BAND_TYPE: `linear' or 'log' for frequency band width [string]
        freq_resp_list: list for computing filter frequency response
        FILTER_TYPE: filter type [string]
        FILTER_ORDER: filter order [integer]
        FILTER_RIPPLE: filter ripple (if Chebyshev I filter) [float]
    Returns:
        vel_array: numpy array with trace velocity results 
        baz_array: numpy array with backazimuth results 
        mdccm_array: numpy array with mdccm results 
        t_array: numpy array with times for array processing results 
        stdict_all: dictionary with dropped elements for LTS [dictionary]
        sig_tau_array: numpy array of sigma tau values for array processing results 
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
    vel_array = np.zeros((NBANDS,vector_len))
    baz_array = np.zeros((NBANDS,vector_len))
    mdccm_array = np.zeros((NBANDS,vector_len))
    t_array = np.zeros((NBANDS,vector_len))
    sig_tau_array = np.zeros((NBANDS,vector_len))
    stdict_all = {}

    # Initialize Frequency response arrays
    w_array = np.zeros((NBANDS,len(w)), dtype = 'complex_')
    h_array = np.zeros((NBANDS,len(h)), dtype = 'complex_')

    num_compute_list = []
    
    ########################################
    ### Run Narrow-Band Array Processing ###
    ########################################
    # Parallel Processing
    results = Parallel(n_jobs=-1)(delayed(narrow_band_loop)(ii, freqlist, FREQ_BAND_TYPE, freq_resp_list, st, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE, rij, WINLEN_list, WINOVER, ALPHA, vector_len) for ii in range(NBANDS))


    ###################################################################
    ### Transform results of parallelization to numpy array outputs ###
    ###################################################################
    for jj in range(NBANDS):
        vel_array[jj,:] = results[jj][0]
        baz_array[jj,:] = results[jj][1]
        mdccm_array[jj,:] = results[jj][2]
        t_array[jj,:] = results[jj][3]

        if ALPHA == 1.0:
            sig_tau_array[jj,:] = results[jj][6]
            stdict_all = None
        elif ALPHA < 1.0:
            sigma_tau_float = None
            # Put 'stdict' back together
            stdict_times = results[jj][4]
            stdict_elements = results[jj][5]
            stdict = {}
            for A, B in zip(stdict_times, stdict_elements):
                stdict[A] = B
            # Append band number to each key in the drop stations dictionary
            temp_dict = {}
            for key in stdict:
                if key != 'size':
                    new_key = str(jj+1).zfill(2) + '_' + key
                    temp_dict[new_key] = stdict[key]
                elif key == 'size':
                    temp_dict[key] = stdict[key]
            stdict_all = {**stdict_all, **temp_dict}

        num_compute_list.append(int(results[jj][7]))
        w_array[jj,:] = results[jj][8]
        h_array[jj,:] = results[jj][9]


    return vel_array, baz_array, mdccm_array, t_array, stdict_all, sig_tau_array, num_compute_list, w_array, h_array
    


