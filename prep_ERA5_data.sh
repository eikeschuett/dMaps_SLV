# this is a Ubuntu bash script to prepare the ERA5 data
# To run it navigate to the folder where the ERA5.nc-file is stroed. In my case:
# cd /mnt/h/Eigene\ Dateien/Studium/10.\ Semester/NIOZ/data/ERA5/
# This bash script may not work properly, because of line-ending issues since this file was created on windows.
# If this is the case, you can simply copy and paste each line into the ubuntu bash.


# Use cdo to regrid the nc file to a 0.5x0.5° resolution
#cdo -L remapbil,r720x360 -select,name=uo,vo,mlotst $eachfile ${eachfile:3:38}


# Regrid the nc file to a 0.5x0.5° resolution.  For other resolutions adjust r{lonpixels}x{latpixles} to the required resolution
cdo -L remapbil,r720x360 ERA5_wind_pressure.nc ERA5_wind_pressure_05_deg.nc

# Regrid to a 2x2° resolution
cdo -L remapbil,r180x90 ERA5_wind_pressure.nc ERA5_wind_pressure_2_deg.nc

# remove the seasonality from the data and calculate anomalies
cdo -L -ymonsub ERA5_wind_pressure_05_deg.nc -ymonmean ERA5_wind_pressure_05_deg.nc ERA5_wind_pressure_05_anomalies.nc

# cdo -L remapbil,r360x180 ERA5_wind_pressure_05_anomalies.nc ERA5_wind_pressure_05_anomalies_1_deg.nc

# Regrid to 4 ° resolution (useful for some animations)
cdo -L remapbil,r90x45 ERA5_wind_pressure_2_deg.nc ERA5_wind_pressure_4_deg.nc

# Calculate the windspeed from u10 and v10 components for the 2° resolution
cdo chname,u10,ws -sqrt -add -sqr -selname,u10 ERA5_wind_pressure_2_deg.nc -sqr -selname,v10  ERA5_wind_pressure_2_deg.nc ERA5_wind_pressure_2_deg_windspeed.nc
