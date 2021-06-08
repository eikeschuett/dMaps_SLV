# this is a Ubuntu bash script to prepare the ERA5 data
# To run it navigate to the folder where the ERA5.nc-file is stroed. In my case:
# cd /mnt/h/Eigene\ Dateien/Studium/10.\ Semester/NIOZ/data/ERA5/

# Use cdo to regrid the nc file to a 0.5x0.5° resolution
#cdo -L remapbil,r720x360 -select,name=uo,vo,mlotst $eachfile ${eachfile:3:38}


# remove the seasonality from the data
cdo -L remapbil,r720x360 ERA5_wind_pressure.nc ERA5_wind_pressure_05_deg.nc
cdo -L -ymonsub ERA5_wind_pressure_05_deg.nc -ymonmean ERA5_wind_pressure_05_deg.nc ERA5_wind_pressure_05_anomalies.nc