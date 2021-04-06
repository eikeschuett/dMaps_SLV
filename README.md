# dMaps_SSH
Application of the deltaMaps method to identify sea surface height anomaly domains. deltaMaps for Python is avaialable [on GitHub](https://github.com/FabriFalasca/py-dMaps).

# Workflow
## Download Altimetry Data
Maps of monthly mean sea level anomalies are available at [AVISO](https://www.aviso.altimetry.fr/en/data/products/sea-surface-height-products/global/gridded-sea-level-anomalies-mean-and-climatology.html) for the period 1993-2020. It can be downloaded from [their ftp hub](ftp://ftp-access.aviso.altimetry.fr/climatology/global/delayed-time/monthly_mean/). Since there is one zipped netCDF4-file per month, I suggest to use a batch download, e.g. by using wget on a Linux system.

```
wget -c -r -np -nH -A nc.gz --user <username> --password <password> --cut-dirs=1 -erobots=off --no-check-certificate ftp://ftp-access.aviso.altimetry.fr/climatology/global/delayed-time/monthly_mean/
```
The downloaded files are still zipped but can easily be unzipped using the unzip.py code.

## Preprocessing of the Sea Level Anomaly data
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
