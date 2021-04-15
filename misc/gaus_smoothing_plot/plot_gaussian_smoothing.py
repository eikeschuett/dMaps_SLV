

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset
import numpy as np
import os
from matplotlib.axes import Axes
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh


os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/data/AVISO/")



def import_data(file):
    ds = Dataset(file, 'r')
    sla = ds.variables["sla"][:].filled(np.nan)
    lat = ds.variables["lat"][:].filled(np.nan)
    lon = ds.variables["lon"][:].filled(np.nan)
    return sla, lat, lon




files = ["AVISO_MSLA_1993-2020_prep_1_deg.nc", 
         "AVISO_MSLA_1993-2020_prep_1_deg_gaus.nc",
         "AVISO_MSLA_1993-2020_prep_2_deg.nc", 
         "AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"]

titles = ["1° w/o smoothing",
          "1° with smoothing",
          "2° w/o smoothing",
          "2° with smoothing"]

time = 60

projection = ccrs.PlateCarree(central_longitude=180)
axes_class = (GeoAxes,
             dict(map_projection=projection))


fig = plt.figure(dpi=300)

axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(2, 2),
                    axes_pad=0.5,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.2,
                    cbar_size='3%',
                    label_mode='') 

for i, ax in enumerate(axgr):

    ax.set_global()
    #ax.coastlines('110m', alpha=0.1)
    ax.add_feature(cfeature.LAND, facecolor='gray', alpha=0.5)
    
    sla, lat, lon = import_data(files[i])
        
    p = ax.pcolormesh(lon, lat, sla[time,:,:], transform=ccrs.PlateCarree(),
                             cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax.set_title(titles[i])

cbar = axgr.cbar_axes[0].colorbar(p)
#cbar.set_label('SLA')

plt.savefig("../../dMaps_SLV/misc/gaus_smoothing_plot/1_2_deg_smoothing.png",
            bbox_inches='tight')
