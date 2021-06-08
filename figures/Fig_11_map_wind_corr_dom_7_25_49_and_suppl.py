
import xarray as xr
import os
try:
    os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
except FileNotFoundError:
    os.chdir("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")    
from dMaps_SLV import dMaps_utils as dMaps
from dMaps_SLV.network import network_analysis_utils as nau
import numpy as np


def extent_to_dict(extents):
    pos_dict = []
    for extent in extents:
        pos_lat = [extent[3], extent[2], extent[2], extent[3], extent[3]]
        pos_lon = [extent[0], extent[0], extent[1], extent[1], extent[0]]
        
        pos_dict_temp = {"lat": pos_lat,
                         "lon": pos_lon}
        pos_dict.append(pos_dict_temp)
    return pos_dict

#%% get domain signal
dmaps_outpath = "dMaps_SLV/results/dMaps/res_2_k_11_gaus/"
# Import domain map
d_maps = np.load(dmaps_outpath + 'domain_identification/domain_maps.npy')
        
ncfile = "dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"
# Import SLA data and lat/lon data
sla = dMaps.importNetcdf(ncfile, "sla")
lat = dMaps.importNetcdf(ncfile, "lat")
lon = dMaps.importNetcdf(ncfile, "lon")
time = dMaps.importNetcdf(ncfile, "time")
        
        
# Produce SLA domain signals for each domain in d_maps and time step in sla
signals = dMaps.get_domain_signals(domains = d_maps,
                                   sla = sla, 
                                   lat = lat, 
                                   signal_type = "average")


#%% Import and produce wind stress data for several different regions for domain 25!

# Extent of the region of interest
# [lonmin, lonmax, latmin, latmax]
# extent = [-95, -80, -43, -33]
extent1 = [-125, -115, -26, -15]
extent2 = [-120, -110, -39, -30]
extent3 = [-110, -98, -41, -31]
extent4 = [-98, -88, -41, -31]
extent5 = [-103, -93, -26, -15]
extent6 = [-115, -103, -24, -15]
extent7 = [-114, -98, -31, -25]
extents_25 = [extent1, extent2, extent3, extent4, extent5, extent6, extent7]

nc_fname = "data/ERA5/ERA5_wind_pressure_05_anomalies_2_deg.nc"
fl_ds = xr.open_dataset(nc_fname)

domain_id = 25

out_dir = "dMaps_SLV/figures/supplementary/wind_stress_lagged_corr/domain_{domain}/".format(domain=domain_id)


nau.mean_areal_cross_corr(fl_ds, 
                          domain_id, 
                          signals, 
                          extents_25, 
                          "wind stress", 
                          out_dir)


#%% Import and produce wind stress data for several different regions for domain 49!

# Extent of the region of interest
# [lonmin, lonmax, latmin, latmax]
# extent = [-95, -80, -43, -33]
extent1 = [-95, -78, -40, -30]
extent2 = [-95, -78, -45, -40]
extent3 = [-95, -78, -52, -45]
extent4 = [-95, -70, -60, -52]
extent5 = [-80, -70, -30, -20]
extent6 = [-90, -80, -30, -20]
extent7 = [-78, -72, -52, -31]
extents_49 = [extent1, extent2, extent3, extent4, extent5, extent6, extent7]

nc_fname = "data/ERA5/ERA5_wind_pressure_05_anomalies_2_deg.nc"
# fl_ds = xr.open_dataset(nc_fname)

domain_id = 49

out_dir = "dMaps_SLV/figures/supplementary/wind_stress_lagged_corr/domain_{domain}/".format(domain=domain_id)

nau.mean_areal_cross_corr(fl_ds, 
                          domain_id, 
                          signals, 
                          extents_49, 
                          "wind stress", 
                          out_dir)



#%% Import and produce wind stress data for several different regions for domain 7!

# Extent of the region of interest
# [lonmin, lonmax, latmin, latmax]
# extent = [-95, -80, -43, -33]
extent1 = [-128, -108, -25, -17]
extent2 = [-108, -90, -25, -17]
extent3 = [-90, -82, -17, -11]
extent4 = [-108, -90, -11, -3]
extent5 = [-128, -108, -11, -3]
extent6 = [-135, -128, -17, -11]
extent7 = [-128, -90, -17, -11]
extents_7 = [extent1, extent2, extent3, extent4, extent5, extent6, extent7]

nc_fname = "data/ERA5/ERA5_wind_pressure_05_anomalies_2_deg.nc"
# fl_ds = xr.open_dataset(nc_fname)

domain_id = 7

out_dir = "dMaps_SLV/figures/supplementary/wind_stress_lagged_corr/domain_{domain}/".format(domain=domain_id)

nau.mean_areal_cross_corr(fl_ds, 
                          domain_id, 
                          signals, 
                          extents_7, 
                          "wind stress", 
                          out_dir)


#%% get domain map

dmaps_path = 'dMaps_SLV/results/dMaps/res_2_k_11_gaus/'
# Import domain maps
d_maps = np.load(dmaps_path + 'domain_identification/domain_maps.npy')
# Create array containing the number of each domain
domain_map = dMaps.get_domain_map(d_maps)

#%% plot one map with all data in it





import matplotlib.pyplot as plt
from cartopy import crs as ccrs

crs = ccrs.PlateCarree(central_longitude=180)

fig, ax =  plt.subplots(1,1,figsize=(12,8), dpi=300,
                        subplot_kw=dict(projection=crs))  


dMaps.plot_map(lat = lat, 
                lon = lon, 
                data = domain_map,
                seeds = None,
                title = "",
                cmap = 'prism',
                alpha = 0.3,
                show_colorbar=False,
                show_grid=True,
                outpath = None,
                labels = True,
                extent = [-130, -60, -65, -5],
                ax=ax) 

# Plot circle around mean position of SPSA
# Derived from ERA5_wind_pressure_2_deg.nc with CDO using the timmean-function
# cdo timmean ERA5_wind_pressure_2_deg.nc ERA5_wind_pressure_2_deg_mean.nc
ax.scatter( -92 , -31 , s = 2000 ,  facecolors='none', edgecolors='red', 
            linewidths=2,
            transform=ccrs.PlateCarree()) 

ax.text(-92, -31, "H", fontsize=30,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ccrs.PlateCarree())


col = ["royalblue", "orangered"]
for k, extent in enumerate([extents_49, extents_25]):
        
    pos_dict = extent_to_dict(extent)
    
    for i in range(len(pos_dict)):
        temp = pos_dict[i]
        ax.plot(temp["lon"], temp["lat"], c=col[k], alpha=0.5,
                transform=ccrs.Geodetic())
    
dom_49_arrows = [[-87,-23,2,-2], # Region 5
                 [-77,-23,2,-2], # Region 4
                 [-85,-33,0.5,-2], # Region 0
                 [-85, -41,-1,-1], # Region 1
                 [-84, -49, -2, 0], # Region 2    
                 [-80,  -55, -2, 0], # Region 3
                 [-80,  -58, -2, 0], # Region 3 second arrow
                 [-74, -37, -1, -2]] # Region 6

dom_25_arrows = [[-97,-20,-2,0], # Region 4
                 [-107, -20, -2, 0], # Region 5
                 [-120, -18, 0, -2], # Region 0
                 [-117, -33, 2, -2], # Region 1
                 [-106, -35, 2, -1], # Region 2
                 [-96, -35, 2, -0.5], # Region 3
                 [-108, -25.3, 0, -2]] # Region 6
    
for k, arrows in enumerate([dom_49_arrows, dom_25_arrows]):
    for arrow_i in arrows:
        plt.arrow(arrow_i[0], arrow_i[1], dx=arrow_i[2], dy=arrow_i[3], 
                  width=0.75, fc = col[k], ec='none',alpha=0.9,
                  transform=ccrs.PlateCarree()) 
plt.savefig("dMaps_SLV/figures/fig_11_dom_49_25_wind_corr.png", 
            bbox_inches='tight')
