# narrow_band_least_squares
Uses the open source [uafgeotools/lts_array](https://github.com/uafgeotools/lts_array) in multiple sequential narrow frequency bands.

## Dependencies
- [conda](https://docs.conda.io/en/latest/)
- Required: [uafgeotools/lts_array](https://github.com/uafgeotools/lts_array)
- Optional: [uafgeotools/waveform_collection](https://github.com/uafgeotools/waveform_collection) if you would like to gather waveforms with that repository. 
- These two repositories are included here as [git submodules](https://www.atlassian.com/git/tutorials/git-submodule), so a copy of them are installed in this repository.

## Installation
In Terminal, navigate to the directory you wish to install then download the repository by running the following:
```
>> git clone --recurse-submodules https://github.com/amiezzi/narrow_band_least_squares.git
```
This will create a folder named `narrow_band_least_squares`. 

```
>> cd narrow_band_least_squares
>> conda env create -f environment.yml
```
This will create a conda environment named `nbls`. 

## Usage
To use the code, you will need to activate the conda environment by 
```
>> conda activate nbls
```

Run an example script in the `narrow_band_least_squares` directory by 
```
>> python example.py
```

## Associated Publications
Please cite the following for any use of this repository, which also has more information on the algorithm and example applications:

Iezzi, A.M., Matoza R.S., Bishop, J.W., Bhetanabhotla, S., and Fee, D. (2022), Narrow-Band Least-Squares Infrasound Array Processing, *Seismological Research Letters: Electronic Seismologist*. [https://doi.org/10.1785/0220220042](https://pubs.geoscienceworld.org/ssa/srl/article/doi/10.1785/0220220042/614310/Narrow-Band-Least-Squares-Infrasound-Array)

Please also cite the following paper, as this repository uses their open-source least-squares code:

Bishop, J. W., Fee, D., and Szuberla, C. A. L. (2020).  Improved infrasound array processing with robust estimators. *Geophysical Journal International*, 221(3):2058–2074. [https://doi.org/10.1093/gji/ggaa110](https://academic.oup.com/gji/article/221/3/2058/5800991?login=true)


## Contact Information
For any questions, please contact Alex Iezzi (amiezzi@alaska.edu).

