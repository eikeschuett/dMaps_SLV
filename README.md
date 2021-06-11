# Sea Level Variability Domains and Their Linkages
This repository contains all codes, results and a report on my project on sea level variability (SLV) with special focus on the South East Pacific. 

Accurate projections of sea level rise are highly important for decision makers and the population in the low-lying coastal zone (Nicholls and Cazenave 2010). However, due to the complex causes for the variability of sea level, reliable regional predictions remain a challenge. Understanding local and regional factors that contribute to this variability will improve the understanding and predictability of sea level (Church et al. 2010, Han et al. 2019). Sea level variations can be caused by many different factors, e.g. local and remote variability of atmospheric or oceanic circulations causing wind stress anomalies or changes in temperature or salinity (Meyssignac et al. 2017). 

In this project, I use a dataset of satellite-derived monthly mean sea level anomaly from [AVISO](https://www.aviso.altimetry.fr/en/data/products/sea-surface-height-products/global.html) and the dimensionality reduction algorithm [deltaMaps](https://github.com/FabriFalasca/py-dMaps) (Fountalis et al. 2018, Falasca et al. 2019) to derive regions with similar SLV dynamics, so called domains. The sea level signals of the different domains are then fed into the causal inference algorithm [PCMCI](https://github.com/jakobrunge/tigramite) (Runge et al. 2019) to identify the causal network between the domains. A subset of the causal network in the Southeast Pacific is then analysed using current velocity data from the CMEMS' [GLORYS reprocessing](https://resources.marine.copernicus.eu/?option=com_csw&view=details&product_id=GLOBAL_REANALYSIS_PHY_001_030) and wind and sea level pressure data from ERA5 provided by ECMWF (Hersbach et al. 2020). The results highlight the importance of the El Niño-Southern Oscillation (ENSO) on both atmospheric and oceanic processes and suggest that SLV patterns in the Southeast Pacific are sensitive to the "flavour" of ENSO. However, this data driven approach can not accurately determine the physical processes at work and some questions remain unanswered. An approach that couples observational data and modelling may help to overcome this constraint.

This repository contains the final report, where I discuss the methods and results in more detail. Furthermore, I included all codes and necessary information to reproduce my results in this repository (see below).

This was my internship project during my time in the sea level group at [NIOZ](https://www.nioz.nl/en). 

### Literature
- Church, John A.; Aarup, Thorkild; Woodworth, Philip L.; Wilson, W. Stanley; Nicholls, Robert J.; Rayner, Ralph et al. (2010): Sea - Level Rise and Variability. Synthesis and Outlook for the Future. In: John A. Church, Philip L. Woodworth, Thorkild Aarup und W. Stanley Wilson (Hg.): Understanding Sea-Level Rise and Variability. Oxford, UK: Wiley-Blackwell, S. 402–419.

- Falasca, Fabrizio; Bracco, Annalisa; Nenes, Athanasios; Fountalis, Ilias (2019): Dimensionality Reduction and Network Inference for Climate Data Using δ ‐MAPS. Application to the CESM Large Ensemble Sea Surface Temperature. In: J. Adv. Model. Earth Syst. 11 (6), S. 1479–1515. DOI: 10.1029/2019MS001654.

- Fountalis, Ilias; Dovrolis, Constantine; Bracco, Annalisa; Dilkina, Bistra; Keilholz, Shella (2018): δ-MAPS. From spatio-temporal data to a weighted and lagged network between functional domains. In: Applied network science 3 (1), S. 21. DOI: 10.1007/s41109-018-0078-z.

- Han, Weiqing; Stammer, Detlef; Thompson, Philip; Ezer, Tal; Palanisamy, Hindu; Zhang, Xuebin et al. (2019): Impacts of Basin-Scale Climate Modes on Coastal Sea Level. A Review. In: Surveys in geophysics 40 (6), S. 1493–1541. DOI: 10.1007/s10712-019-09562-8.

- Hersbach, Hans; Bell, Bill; Berrisford, Paul; Hirahara, Shoji; Horányi, András; Muñoz‐Sabater, Joaquín et al. (2020): The ERA5 global reanalysis. In: Q.J.R. Meteorol. Soc. 146 (730), S. 1999–2049. DOI: 10.1002/qj.3803.

- Meyssignac, B.; Piecuch, C. G.; Merchant, C. J.; Racault, M.-F.; Palanisamy, H.; MacIntosh, C. et al. (2017): Causes of the Regional Variability in Observed Sea Level, Sea Surface Temperature and Ocean Colour Over the Period 1993–2011. In: Surv Geophys 38 (1), S. 187–215. DOI: 10.1007/s10712-016-9383-1.

- Nicholls, Robert J.; Cazenave, Anny (2010): Sea-level rise and its impact on coastal zones. In: Science (New York, N.Y.) 328 (5985), S. 1517–1520. DOI: 10.1126/science.1185782.

- Runge, Jakob; Nowack, Peer; Kretschmer, Marlene; Flaxman, Seth; Sejdinovic, Dino (2019): Detecting and quantifying causal associations in large nonlinear time series datasets. In: Science advances 5 (11), eaau4996. DOI: 10.1126/sciadv.aau4996.

# Notes

## TBD
- [ ] update readme
  - [x]  data preparation (bash scripts are included) and availability of climate indices!
  - [x]  environments yml files
  - [x]  structure of directories and required additional packages which are not available through conda (i.e. airsea and py-dMaps)
  - [x]  animation
  - [ ]  dconsider to delete stuff at the bottom
- [ ] write and upload report
- [ ] prepare presentation for Thusday
- [ ] 

# Workflow
## Overview
This study is divided into three main parts: 
1. Dimensionality reduction with [deltaMaps](https://github.com/FabriFalasca/py-dMaps)
2. Inference of a causal network with [PCMCI](https://github.com/jakobrunge/tigramite)
3. Data analysis and interpretation of the results


## Preparation of environment and general remarks
In this project, I used data from different sources and code from different other repositories. All data I used is freely available on the internet (on some pages its required to create an account). Because of legal concerns and the size of the datasets, not all of the data I used is included in this repo. Instead, I included download instructions and scripts to prepare the data in the same way I did it. I recommend to put these files into a ```data```-directory in the parent directory of this repo (see directory tree below). For the packages [airsea](https://github.com/pyoceans/python-airsea) and [deltaMaps](https://github.com/FabriFalasca/py-dMaps) no installation through conda or pip is available so far. Therefore I cloned them into the parent directory of this repo (see directory tree below). In all scripts I used the parent directory or this repo as working directory and imported the packages from there. Other approaches to properly install them or use the ```sys.path.append()```-function work as well, but may require some adjustments in the filepaths in the code.

```bash
├── airsea (from: https://github.com/pyoceans/python-airsea)
├── data
│   ├── climate_indices
│   │   ├── nino_34_anomaly.txt
│   ├── CMEMS
│   ├── ERA5
│   │   ├── ERA5_wind_pressure_2_deg.nc
│   │   ├── ERA5_wind_pressure_05_anomalies_2_deg.nc
├── dMaps_SLV (this repository)
├── py-dMaps (from https://github.com/FabriFalasca/py-dMaps)
```
### Preparation of Python environments
I used two different environments for this project. For the first part (deltaMaps) I worked in Python (Miniconda) on Ubuntu on my Windows 10 machine (Windows Subsystem for Linux, WSL), which made the download and preprocessing of the SLA dataset from AVISO much easier. Later I switched to Python (Anaconda) directly on Windows, because this was a bit easier. If I remember correctly, this was because of conflicts between the required packages which were requrired for deltaMaps and tigramite, so that I would have needed to use two environments anyway. Below are more detailed instructions on how to clone my Python environments.

#### deltaMaps environment on Ubuntu
For this project I used Python (Miniconda) on Ubuntu on my Windows 10 machine (Windows Subsystem for Linux, WSL). [This article](https://medium.com/@macasaetjohn/setting-up-a-spyder-environment-with-wsl-bb83716a44f3) explains how to install Ubuntu on a Windows 10 computer, install Miniconda and Spyder (which is not trivial if you want to run it from the Ubuntu Bash. However, this will make the preparation of the AVISO data easier).

Some additional packages are required. I included ``deltaMaps_env.yml`` in the repo, so that you can clone my environment using this command in your Ubuntu bash:
```
conda env create -f deltaMaps_env.yml
```
Additionally, you need to have the python version of deltaMaps locally on your computer (get it [here](https://github.com/FabriFalasca/py-dMaps)) and place it in the parent directory of this repo (see directory tree above).

The prepparation of the dataset is mainly based on [Climate Data Operators (CDO)](https://code.mpimet.mpg.de/projects/cdo). Instructions on the installation on Windows can be found [here](https://code.mpimet.mpg.de/projects/cdo/wiki/Win32). I chose to install CDO in my Ubuntu environment, because it's dead simple to install it with the native package manager (apt). In the Ubuntu bash run these two commands:
```
sudo apt-get install cdo
```
```
sudo apt-get upgrade
```
#### networkx environment for Tigramtie on Windows
To run the PCMCI algorithm from the tigramite package (and to properly plot the results) I created a new environment in Anaconda on Windows. You can clone this environment with this command
```
conda env create -f networkx_env.yml
```
The data analysis was also done in this environment. To calculate windstress, I used the package [airsea](https://github.com/pyoceans/python-airsea), which needs to be installed in the same way as py-dMaps. For correlation analysis I downlaoded data for the [Nino3.4 index from NOAA](https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino34/) and saved it as ``parent_dir/data/climate_indices/nino_34_anomaly.txt`` (see directory tree above).

Now that the preparation is done, we can (almost) start to run the codes!

## Preparation of SLA data and dimensionality reduction with deltaMaps
``dMaps_utils.py`` contains many (helper-) functions required to download and preprocess the AVISO (or any other) dataset, run deltaMaps, visualize the results and apply the proposed heuristic by [Falasca et al. 2019](https://doi.org/10.1029/2019MS001654) to identify the optimal value for the neighborhood size (k).

To download and preprocess the data, open ``dMaps_utils.py`` and scroll towards the ``Download and prepare dataset`` section (~line 1305). In this section, you need to check that all file paths are correct and you enter your AVISO credentials. Then run this file in you ``dMaps``-environment on Ubuntu. By default, the final output file will be called ``AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc``.

For convinience, i now switched to my ``networkx``-environment on Windows, but I think you should be able to run deltaMaps from the environment on Ubuntu. Anyway, to run deltaMaps on the prepared SLA dataset, have a look into ``run_dMaps_on_SLA.py``. Again you should check that all path and filenames are correct. There are two pitfalls: 1. The repository ``dMaps_SLV`` (from where ``dMaps_utils`` is imported) is a subdirectory of the working directory. If this is different in your case, you need to sys.append the directory containing ``dMaps_utils`` to import it correctly. 2: The function ``dMaps.run_dMaps`` (~line 37) requires the (relative) path the ``run_delta_maps.py``, which is located in your clone of the deltaMaps-repo (``py-Dmaps``).

Finding the best value for the neighborhood size (K) (i.e. the K nearest neighbors around a grid cell that initially form a region) is a crucual step when using deltaMaps. If K is too small, the local homogeneity field will be noisy, if its too large, the local homogeneity may be oversmoothed and candidates for domains may remain undeteced. According to [Falasca et al. 2019](https://doi.org/10.1029/2019MS001654), the best K-value can be found by comparing the differences between several runs of deltaMaps with different values for k using the Normalize Mutual Information (NMI). For this reason, deltaMaps will by default run for k's between 4 and 20 and the NMI-matrix will be plotted at the end. K should be choosen so that the NMI is large for this K and for neighboring values of K. This avoids that the K is sensitive to fluctuations around the chosen K value. It should also be at least 4 so that neighbors in all directions of a grid cell are considered. In my case, k=11 appears to be the best choice. Therefore, the results from the deltaMas run with k=11 will be used for the remainder of the study.

## Inference of a causal network with the PCMCI-algorithm from the tigramite package
``run_tigramite_on_dMaps.py`` contains some helper functions to run tigramite on the results of deltaMaps (i.e. SLA dataset with reduces dimensionality). Again, you should check that the working directory (~ line 665) and all other filepaths are correct. Some of the functions import ``dMaps_utils.py`` and ``network_analysis_utils.py``. Therefore I suggest to use the parent directory of the repo as working directory. Otherwise you will have to fix this with a sys.append. 

PCMCI can take some time to process. The result of the algorithm will be saved as a pickle-file. Additionally, the network will be plotted and saved. Optionally, the plotting-function supports to plot only edges to/from a specific node.

## Analysis of the semi-enclosed circular network in the South-East Pacific Ocean
I based my analysis of the network mainly on the literature and on the wind (10m u- & v-component of wind) and mean sea level pressure data from the ERA5 reprocessing, which provides monthly averages. I downloaded it for the period 1993-2021 directly from [their website](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form) and preprocessed it with cdo in the Ubuntu bash. For this I used the commands in ``/data/prep_ERA5_data.sh``. Additionally, flow data was taken from the [Global Ocean Physics Reanalysis (GLORYS12V1) product from CMEMS](https://resources.marine.copernicus.eu/?option=com_csw&view=details&product_id=GLOBAL_REANALYSIS_PHY_001_030) and prepared again in CDO according to ``/data/prep_cmems_data.sh``.

All the figures in the Report were produced with code provided in the [``/figures``](https://github.com/eikeschuett/dMaps_SLV/tree/main/figures) subdirectory of this repo. There you can also find the png-versions of each plot and supplementary plots. An in-depth discussion of all figures can be found in the [Report](wikipedia.org)

## Animations of SLA, SLPa and wind stress anomalies
An animation of the development of SLA, SLPa and wind stress anomalies in the South-Eastern Pacific can be created with ``/network/sla_wind_stress_animation_cmems_isobars_ERA5.py``. By default, it produces gifs at different framerates for both the original eddy-resolving AVISO resolution (1/4x1/4°) and the regridded ans smoothed 2x2° version that was used in this project. The results are avialable at ``[/figures/supplementary/animation](https://github.com/eikeschuett/dMaps_SLV/tree/main/figures/supplementary)``.














For possible later use
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
