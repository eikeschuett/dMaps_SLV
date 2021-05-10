# dMaps_SLV
Application of deltaMaps method to identify sea surface veriability domains. deltaMaps for Python is avaialable [on GitHub](https://github.com/FabriFalasca/py-dMaps).

A functional network is infered with deltaMaps and a causal network with PCMCI (included in the package [tigramite](https://github.com/jakobrunge/tigramite).

# Notes

## TBD
- [ ] update readme
- [x] plot PCMCI network with lower p value
- [ ] interprete network
- [ ] update report
- [ ] read paper
- [ ] update bash scripts for data preparation

# Workflow

## Preparation of environment and general remarks
For this project I used Python (Miniconda) on Ubuntu on my Windows 10 machine (Windows Subsystem for Linux, WSL). [This article](https://medium.com/@macasaetjohn/setting-up-a-spyder-environment-with-wsl-bb83716a44f3) explains how to install Ubuntu on a Windows 10 computer, install Miniconda and Spyder (which is not trivial if you want to run it from the Ubuntu Bash. However, this will make the preparation of the data easier).
Additionally, the following python packages are required:
- numpy
- scipy
- scikit-learn
- pandas
- netCDF4
- matplotlib
- cartopy
- cmocean
- networkx
- (tigramite for causal discovery)

Alternatively, you can clone my Anaconda environment. This may be useful since I have not been using the newest versions of all packages in order to get Spyder running from the Ubuntu WSL. The file env.txt contains the required package information and the conda command required to clone the environment (in the second line).

Additionally, you need to have the python version of deltaMaps locally on your computer (get it [here](https://github.com/FabriFalasca/py-dMaps) ).

The prepparation of the dataset is mainly based on [Climate Data Operators (CDO)](https://code.mpimet.mpg.de/projects/cdo). Instructions on the installation on Windows can be found [here](https://code.mpimet.mpg.de/projects/cdo/wiki/Win32). I chose to install CDO in my Ubuntu environment, because it's dead simple to install it with the native package manager (apt). In the Ubuntu bash run these two commands:
```
sudo apt-get install cdo
```
```
sudo apt-get upgrade
```

## Prepare data, run deltaMaps and visualize the results
dMaps_utils.py contains all functions required to download and preprocess the AVISO (or any other) dataset, run deltaMaps, visualize the results and apply the proposed heuristic by [Falasca et al. 2019](https://doi.org/10.1029/2019MS001654) to identify the optimal value for the neighborhood size (k). In dMaps_utils.py all functions are defined and a code example is provided.

### Download and preprocessing of data
To make it easier to download and preprocess the data, use the following functions:
1. aviso_download(): downloads monthly sea level anomaly data from AVISO using wget
2. unzip(): unzips the data
3. concat_nc_files(): concatenates the monthly nc-files into a single nc file with a time dimension using cdo
4. nc_prep(): resamples the nc-files to 1 or 2 ° resolution (if specified), removes the seasonal cycle and a linear trend from the data and subsets the data to a specified region (if specified)
5. nc_gaus_filter(): applies a gaussian filter to the dataset in order to spatially smooth the dataset. With this the influence of disturbing mesoscale eddies on the result is reduced.

### Run deltaMaps
1. create_config_file(): Creates a config file with all required information to run deltaMaps (e.g. input and output filenames/paths, required parameters, etc.)
2. run_dMaps(): Applies the deltaMaps with the settings specified in the previous step.

### Visualize output
plot_dMaps_output(): Plots a map of the local homogeneity and/or domains and/or domain strength. If needed, it can also plot the locations of the seeds on the maps.
plot_network(): Plots the functional network infered by deltaMaps on a projected map.

### Finding the best value for K
Finding the best value for the neighborhood size (K) (i.e. the K nearest neighbors around a grid cell that initially form a region) is a crucual step when using deltaMaps. If K is too small, the local homogeneity field will be noisy, if its too large, the local homogeneity may be oversmoothed and candidates for domains may remain undeteced. According to [Falasca et al. 2019](https://doi.org/10.1029/2019MS001654), the best K-value can be found by comparing the differences between several runs of deltaMaps with different values for k using the Normalize Mutual Information (NMI). This is done in
1. calc_nmi_matrix(), which returns a matrix of the NMIs for each combination of deltaMaps-run for the different k-values.
2. plot_nmi_matrix() plots the NMI matrix.

K should be choosen so that the NMI is large for this K and for neighboring values of K. This avoids that the K is sensitive to fluctuations around the chosen K value. It should also be at least 4 so that neighbors in all directions of a grid cell are considered.

### Alternative way to download and preprocess the data
Data con be obtained and prepared using the Ubuntu bash, wget and cdo. This does effectively the same as automated in the functions above. I can only think of one situation where it may be useful to use the bash manually: If the internet connection is bad and the download is interrupted, the Python code will have a problem. If the wget command is executed manually in the bash, it can be started again when the internet connection is better. Anyway, I want to keep the instructions below for possible future use.

#### Download of the data
Maps of monthly mean sea level anomalies are available at [AVISO](https://www.aviso.altimetry.fr/en/data/products/sea-surface-height-products/global/gridded-sea-level-anomalies-mean-and-climatology.html) for the period 1993-2020. It can be downloaded from [their ftp hub](ftp://ftp-access.aviso.altimetry.fr/climatology/global/delayed-time/monthly_mean/). Since there is one zipped netCDF4-file per month, I suggest to use a batch download, e.g. by using wget on a Linux system.

```
wget -c -r -np -nH -A nc.gz --user <username> --password <password> --cut-dirs=1 -erobots=off --no-check-certificate ftp://ftp-access.aviso.altimetry.fr/climatology/global/delayed-time/monthly_mean/
```
The downloaded files are still zipped but can easily be unzipped using the unzip function in the prep_data.py code.

#### Preprocessing of the Sea Level Anomaly data
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

## PCMCI

Since deltaMaps provides only a functional network (based on simple cross-correlations), the PCMCI method from the package [tigramite](https://github.com/jakobrunge/tigramite) can be used to infer a causal network and analyse the network. This is done in "run_tigramite_on_dMaps.py". This code provides a function that prepares the deltaMaps output for PCMCI and applies this algorithm. Then it plots the network on a projected map and maps of the Average Causal Effect & Average Causal Susceptibility as well as the Average Mediated Causal Effect. A code example is provided.

## Results
Data and plots are available in the /results-directory.
