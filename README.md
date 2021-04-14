# dMaps_SSH
Application of the deltaMaps method to identify sea surface height anomaly domains. deltaMaps for Python is avaialable [on GitHub](https://github.com/FabriFalasca/py-dMaps).

# Notes
## Adjustment knobs
- Data preparation
  - resolution
  - seasonality removal (done by CDO)
  - trend removal (done by CDO)
  - [crop (done by CDO and determined by satellite orbits)]
  - gaussian filter
    - sigma
    - truncate
- deltaMaps
  - number of random samples for delta calculation -> heuristic
  - alpha/q significance levels (should better stay on default values)
  - k (neighborhood size) -> heuristic
  - tau_max: only important for network inference

## TBD
- [x] check removal of seasonality and trend. Better use non-linear detrending method?
- [ ] check gaussian filter settings and plot results
- [ ] apply heuristic to calculate number of random samples. Visualize with boxplot
- [ ] run dMaps again for different values of k and determine best k
- [ ] clear up code and push
- [ ] update readme workflow to current version

# Workflow
## Preparation of environment and general remarks
For this project I used Python (Miniconda) on Ubuntu on my Windows 10 machine (Windows Subsystem for Linux, WSL). [This article](https://medium.com/@macasaetjohn/setting-up-a-spyder-environment-with-wsl-bb83716a44f3) explains how to install Ubuntu on a Windows 10 computer, install Miniconda and Spyder (which is not trivial if you want to run it from the Ubuntu Bash. However, this will make the preparation of the data easier).
Additionally, the following packages are required:
- numpy
- scipy
- scikit-learn
- netCDF4
- matplotlib
- cartopy

## Download and preprocessing of data
To make it easier to download and preprocess the data, use prep_data.py. This code
1. downloads monthly sea level anomaly data from AVISO using wget
2. unzips the data
3. concatenates the monthly nc-files into a single nc file with a time dimension using cdo
4. resamples the nc-files to 1 or 2 ° resolution (if specified)
5. removes the seasonal cycle and a linear trend from the data
6. subsets the data to a specified region (if specified)

## Run deltaMaps
Use run_dMaps.py to create the required configuration file with all parameters and then run the deltaMaps algorithm.

## Visualize output
plot_results.py plots maps of the local homogeneity field, the seeds locations and the domains that have been identified.

## Alternative way to download and preprocess the data
Data con be obtained and prepared using the Ubuntu bash, wget and cdo. This does effectively the same as automated in prep_data.py. I can only think of one situation where it may be useful to use the bash manually: If the internet connection is bad and the download is interrupted, the Python code will have a problem. If the wget command is executed manually in the bash, it can be started again when the internet connection is better. Anyway, I want to keep the instructions below for possible future use.

### Download of the data
Maps of monthly mean sea level anomalies are available at [AVISO](https://www.aviso.altimetry.fr/en/data/products/sea-surface-height-products/global/gridded-sea-level-anomalies-mean-and-climatology.html) for the period 1993-2020. It can be downloaded from [their ftp hub](ftp://ftp-access.aviso.altimetry.fr/climatology/global/delayed-time/monthly_mean/). Since there is one zipped netCDF4-file per month, I suggest to use a batch download, e.g. by using wget on a Linux system.

```
wget -c -r -np -nH -A nc.gz --user <username> --password <password> --cut-dirs=1 -erobots=off --no-check-certificate ftp://ftp-access.aviso.altimetry.fr/climatology/global/delayed-time/monthly_mean/
```
The downloaded files are still zipped but can easily be unzipped using the unzip function in the prep_data.py code.

### Preprocessing of the Sea Level Anomaly data
First, I [installed CDO on my Ubuntu subsystem on my Windows 10 computer](https://code.mpimet.mpg.de/projects/cdo/wiki/Win32). With CDO our ndetCDF-files can be easily prepared for deltaMaps. Next, I carried out the following commands in my Ubuntu bash:

- Concatenate all individual nc-files into a new one which contains a time variable and the data of each month
```
cdo mergetime *.nc AVISO_dt_global_allsat_msla_h_1993-2020.nc
```
- Remove the seasonal cycle
```
cdo -L -ymonsub AVISO_dt_global_allsat_msla_h_1993-2020.nc -ymonmean AVISO_dt_global_allsat_msla_h_1993-2020.nc AVISO_dt_global_allsat_msla_h_1993-2020_s.nc
```
- Remove linear trend
```
cdo detrend AVISO_dt_global_allsat_msla_h_1993-2020_s.nc AVISO_dt_global_allsat_msla_h_1993-2020_s_t.nc
```
