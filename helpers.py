import numpy as np
import math as math
from scipy import signal 
from obspy.geodetics.base import calc_vincenty_inverse



def get_freqlist(FMIN, FMAX, FREQ_BAND_TYPE, NBANDS):
    '''
    Obtains list of narrow frequency band limits
    Args:
        FMIN: Minimum frequency [float] [Hz]
        FMAX: Maximum frequency [float] [Hz]
        FREQ_BAND_TYPE: `linear' or 'log' for frequency band width [string]
        NBANDS: number of frequency bands [integer]
    Returns:
        freqlist: list of narrow frequency band limits
        nbands_calc: updated NBANDS (applicable for 'octave' types)
        FMAX_calc: updated FMAX (applicable for 'octave' types)
    '''
    freqrange = FMAX - FMIN

    if FREQ_BAND_TYPE == 'linear':
        freqinterval = freqrange / NBANDS
        freqlist = np.arange(FMIN, FMAX+freqinterval, freqinterval)
        nbands_calc = NBANDS
        FMAX_calc = FMAX

    elif FREQ_BAND_TYPE == 'log':
        FMINL = math.log(FMIN, 10)
        FMAXL = math.log(FMAX, 10)
        freqlist = np.logspace(FMINL, FMAXL, num = NBANDS+1)
        nbands_calc = NBANDS
        FMAX_calc = FMAX

    elif FREQ_BAND_TYPE == 'octave':
        # octave width; upper frequency (f2) is twice the lower frequency (f1)
        freqlist=[FMIN,]
        while 2*freqlist[-1]<=FMAX:
            freqlist.append(2*freqlist[-1])
        # double check NBANDS
        nbands_calc = int(len(freqlist)) -1
        FMAX_calc = freqlist[-1]

    elif FREQ_BAND_TYPE == '2_octave_over':
        # e.g. used in Green and Bowers (2010), JGR
        # two-octave bands that overlap by 1 octave
        # upper frequency (f2) is 4 times the lower frequency (f1)
        freqlist = [FMIN,]
        while 2*freqlist[-1]<=FMAX:
            freqlist.append(2*freqlist[-1])
        # double check NBANDS
        nbands_calc = int(len(freqlist)) -2
        FMAX_calc = freqlist[-1]

    elif FREQ_BAND_TYPE == 'onethird_octave':
        # e.g. Garces (2013), Inframatics
        # a one third octave is when the upper band edge (f2) is the lower band edge (f1) times the cubed root of 2
        freqlist=[FMIN,]
        while freqlist[-1]* (2** (1./3.)) <=FMAX:
            freqlist.append(freqlist[-1]* (2** (1./3.)))
        # double check NBANDS
        nbands_calc = int(len(freqlist)) -1
        FMAX_calc = freqlist[-1]

    elif FREQ_BAND_TYPE == 'octave_linear':
        # octave width from FMIN to the switch frequency, then linear until FMAX
        switch_freq = 2
        freqlist=[FMIN,]
        while 2*freqlist[-1]<=switch_freq:
            freqlist.append(2*freqlist[-1])
        temp_nbands = NBANDS - len(freqlist)
        freqinterval = (FMAX-freqlist[-1]) / temp_nbands
        freqlist = freqlist + list(np.arange(freqlist[-1], FMAX+freqinterval, freqinterval))
        # double check NBANDS
        nbands_calc = int(len(freqlist)) -1
        FMAX_calc = FMAX

    return freqlist, nbands_calc, FMAX_calc



def get_winlenlist(WINDOW_LENGTH_TYPE, NBANDS, WINLEN, WINLEN_1, WINLEN_X):
    '''
    Obtains list of window length for narrow-band processing
    Args:
        WINDOW_LENGTH_TYPE: 'constant' or 'adaptive' [string]
        NBANDS: number of frequency bands [integer]
        WINLEN: window length [s]; used if WINDOW_LENGTH_TYPE = 'constant' AND if WINDOW_LENGTH_TYPE = '1/f' (because of broadband processing)
        WINLEN_1: window length for band 1 (lowest frequency) [s]; only used if WINDOW_LENGTH_TYPE = 'adaptive'
        WINLEN_X: window length for band X (highest frequency) [s]; only used if WINDOW_LENGTH_TYPE = 'adaptive'
    Returns:
        WINLEN_list: list of window length for narrow-band processing
    '''
    if WINDOW_LENGTH_TYPE == 'constant':
        WINLEN_list = [] 
        for ii in range(NBANDS):
            WINLEN_list.append(WINLEN)
    elif WINDOW_LENGTH_TYPE == 'adaptive':
        # varies linearly with period
        WINLEN_list = np.linspace(WINLEN_1, WINLEN_X, num=NBANDS)
        WINLEN_list = [int(item) for item in WINLEN_list]

    return WINLEN_list



def filter_data(st, FILTER_TYPE, FMIN, FMAX, FILTER_ORDER, FILTER_RIPPLE):
    '''
    Filter and taper the data
    Args:
        st: array data (:class:`~obspy.core.stream.Stream`)
        FILTER_TYPE: filter type; 'butter', 'cheby1' [string]
        FMIN: Minimum frequency [float] [Hz]
        FMAX: Maximum frequency [float] [Hz]
        FILTER_ORDER: filter order [integer]
        FILTER_RIPPLE: filter ripple (if Chebyshev I filter) [float]
    Returns:
        stf: Filtered array data (:class:`~obspy.core.stream.Stream`)
        Fs: Sampling Frequency [Hz] [float]
        sos: second-order sections of filter

    '''
    stf = st.copy()
    Fs = stf[0].stats.sampling_rate
    if FILTER_TYPE == 'butter': 
        stf.filter('bandpass', freqmin = FMIN, freqmax = FMAX, corners=FILTER_ORDER, zerophase = True)
        sos = signal.iirfilter(FILTER_ORDER, [FMIN, FMAX], btype='band',ftype='butter', fs=Fs, output='sos')    
    elif FILTER_TYPE == 'cheby1': 
        sos = signal.iirfilter(FILTER_ORDER, [FMIN, FMAX], rp=FILTER_RIPPLE, btype='band', analog=False, ftype='cheby1', fs=Fs,output='sos')
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


def write_txtfile(save_dir, fname,  vel_array, baz_array, mdccm_array, t_array, freqlist, num_compute_list):
    '''
    Write array processing results to txt file
    Args:
        save_dir: directory in which to save output file
        fname: output filename root
        vel_array: numpy array with trace velocity results 
        baz_array: numpy array with backazimuth results 
        mdccm_array: numpy array with mdccm results 
        t_array: numpy array with times for array processing results 
        freqlist: list of narrow frequency band limits
        num_compute_list: length for processing reults in each frequency band
    Returns:
        None
    '''
    f = open(save_dir + fname + '.txt', 'w')
    f.write('Fmin \t Fmax \t Time \t Trace_vel \t Backaz \t MdCCM \n')
    for ii in range(len(freqlist)-1):
        print((num_compute_list[ii]))
        for jj in range(num_compute_list[ii]):
            f.write(str(freqlist[ii]) + '\t' + str(freqlist[ii+1]) + '\t' + str(t_array[ii,jj]) + '\t' + str(vel_array[ii,jj]) + '\t' + str(baz_array[ii,jj]) + '\t' + str(mdccm_array[ii,jj]) + '\n')
    f.close()


def read_txtfile(save_dir, fname):
    '''
    Read array processing results from txt file
    Args:
        save_dir: directory in which to save output file
        fname: output filename root
    Returns:
        vel_array: numpy array with trace velocity results 
        baz_array: numpy array with backazimuth results 
        mdccm_array: numpy array with mdccm results 
        t_array: numpy array with times for array processing results 
        freqlist: list of narrow frequency band limits
        num_compute_list: length for processing reults in each frequency band
    '''
    temp_file = np.genfromtxt(save_dir + fname + '.txt', skip_header=1, dtype='float')

    # Create freqlist
    fmin_list = temp_file[:,0]
    fmax_temp = temp_file[-1,1]
    unique_freq, idx = np.unique(fmin_list, return_index=True)
    freqlist = np.append(unique_freq, fmax_temp)
    idx = np.append(idx,len(fmin_list))
    num_compute_list = np.diff(idx)
    FMIN = fmin_list[0]
    FMAX = fmax_temp

    # Setup arrays
    vector_len = len(fmin_list) - idx[-2]
    nbands = len(freqlist)-1
    vel_array = np.empty((nbands,vector_len))
    baz_array = np.empty((nbands,vector_len))
    mdccm_array = np.empty((nbands,vector_len))
    t_array = np.empty((nbands,vector_len))

    # Fill in the arrays
    t_list = temp_file[:,2]
    vel_list = temp_file[:,3]
    baz_list = temp_file[:,4]
    mdccm_list = temp_file[:,5]


    for ii in range(nbands):
        temp_start_idx = idx[ii]
        temp_end_idx = idx[ii+1]
        temp_len = temp_end_idx - temp_start_idx
        vel_array[ii,:temp_len] = vel_list[temp_start_idx:temp_end_idx]
        baz_array[ii,:temp_len] = baz_list[temp_start_idx:temp_end_idx]
        mdccm_array[ii,:temp_len] = mdccm_list[temp_start_idx:temp_end_idx]
        t_array[ii,:temp_len] = t_list[temp_start_idx:temp_end_idx]

    return vel_array, baz_array, mdccm_array, t_array, freqlist, num_compute_list, nbands, FMIN, FMAX



def get_rij(latlist, lonlist, nchans):
    """ 
    Calculate element locations (r_ij) from latitude and longitude.

    Return the projected geographic positions
    in X-Y (Cartesian) coordinates. Points are calculated
    with the Vincenty inverse and will have a zero-mean.

    Args:
        latlist (list): A list of latitude points.
        lonlist (list): A list of longitude points.

    Returns:
        (array):
        ``rij``: A numpy array with the first row corresponding to
        cartesian "X" - coordinates and the second row
        corresponding to cartesian "Y" - coordinates.

    Modified from lts_array by Jordan Bishop

    """

    # Check that the lat-lon arrays are the same size.
    if (len(latlist) != nchans) or (len(lonlist) != nchans):
        raise ValueError('Mismatch between the number of stream channels and the latitude or longitude list length.') # noqa

    # Pre-allocate "x" and "y" arrays.
    xnew = np.zeros((nchans, ))
    ynew = np.zeros((nchans, ))

    for jj in range(1, nchans):
        # Obspy defaults to the WGS84 ellipsoid.
        delta, az, _ = calc_vincenty_inverse(
            latlist[0], lonlist[0], latlist[jj], lonlist[jj])
        # Convert azimuth to degrees from North
        az = (450 - az) % 360
        xnew[jj] = delta/1000 * np.cos(az*np.pi/180)
        ynew[jj] = delta/1000 * np.sin(az*np.pi/180)

    # Remove the mean.
    xnew -= np.mean(xnew)
    ynew -= np.mean(ynew)

    rij = np.array([xnew.tolist(), ynew.tolist()])

    return rij