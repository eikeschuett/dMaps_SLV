

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")


def import_data(path_to_data, var):
    nc_fid = nc.Dataset(path_to_data, 'r')
    data = nc_fid.variables[var][:].filled(np.nan)
    
    lat = nc_fid.variables['lat'][:].filled(np.nan)
    lon = nc_fid.variables['lon'][:].filled(np.nan)
    
    time_var = nc_fid.variables['time']
    time = nc.num2date(time_var[:],time_var.units, 
                       only_use_cftime_datetimes=False, 
                       only_use_python_datetimes=True).filled(np.nan)
    return data, time, lat, lon
    

path_to_data = "data/AVISO/AVISO_MSLA_1993-2020_prep_2_deg.nc"
detrend, time, lat, lon = import_data(path_to_data=path_to_data, 
                            var='sla')

path_to_data = "data/AVISO/AVISO_MSLA_1993-2020_2_deg_crop.nc"
orig, time, _, _ = import_data(path_to_data=path_to_data, 
                            var='sla')


y = [60, 30, 55, 20, 35, 35, 10, 35]
x = [2, 1, -15, 40, 78, 105, 105, 135]

for i in range(len(y)):
    
    y_lat = y[i]
    x_lon = x[i]
    
    lat_i = lat[y_lat]
    lon_i = lon[x_lon]
    
    if lon_i > 180:
        lon_i = -lon_i+180
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(2,1,1, 
                             projection = ccrs.Robinson())
    
    ax.plot(lon[x_lon], lat[y_lat], 
            marker='+', 
            color='red', 
            markersize=5,
            transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.set_global()
    ax.coastlines('110m', alpha=1)
    plt.title("Lat: " + str(lat_i) + "°N LON: " + str(lon_i) + "°E")
    #plt.show()
    
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(time, detrend[:,y_lat,x_lon], linewidth=1, label='detrended')
    ax2.plot(time, orig[:,y_lat,x_lon], linewidth=0.5, label = 'original')
    ax2.set_ylabel("SLA [m]")
    ax2.set_xlabel("year")
    ax2.legend(loc='lower right',
               ncol=2)
    ax2.grid()
    # ax.set_xlim([datetime.datetime(1993, 1, 15, 0, 0), 
    #              datetime.datetime(2015, 1, 15, 0, 0)])
    
    path = "dMaps_SLV/misc/trend_analysis/"
    fname = "trend_" + str(lat_i) + "N_" + str(lon_i) + "E.jpg"
    
    plt.savefig(path+fname)