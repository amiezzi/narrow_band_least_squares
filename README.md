# narrow_band_least_squares
Uses the open source [uafgeotools/array_processing](https://github.com/uafgeotools/array_processing) in multiple sequential narrow frequency bands.
 
## Dependencies
- Follow the instructions to install [uafgeotools/array_processing](https://github.com/uafgeotools/array_processing), which will create the new conda environment `uafinfra`. 
- [Obspy](https://docs.obspy.org/) is included in this environment
- If you would like to use the parallelized version, you must also install [joblib](https://joblib.readthedocs.io/en/latest/) in the `uafinfra` conda environment. For example, using conda:
```
>> conda install -c anaconda joblib
```

## Installation
In Terminal, navigate to the directory you wish to install then download the repository by running the following:
```
>> git clone https://github.com/amiezzi/narrow_band_least_squares.git
```
This will create a folder named `narrow_band_least_squares`. 

## Usage
To use the code, you will need to activate the conda environment from [uafgeotools/array_processing](https://github.com/uafgeotools/array_processing) by 
```
>> conda activate uafinfra
```

Run an example script by 
```
>> python example.py
```

## Associated Publications
Please cite the following for any use of this repository, which also has more information on the algorithm and example applications:

Iezzi, A.M., Matoza R.S., Bishop, J.W., Bhetanabhotla, S., and Fee, D. (*accepted*), Narrow-Band Least-Squares Infrasound Array Processing, *submitted to Seismological Research Letters: Electronic Seismologist*.

Please also cite the following paper, as this repository uses their open-source least-squares code:

Bishop, J. W., Fee, D., and Szuberla, C. A. L. (2020).  Improved infrasound array processing with robust estimators. *Geophysical Journal International*, 221(3):2058–2074.


## Contact Information
For any questions, please contact Alex Iezzi (amiezzi@ucsb.edu).

