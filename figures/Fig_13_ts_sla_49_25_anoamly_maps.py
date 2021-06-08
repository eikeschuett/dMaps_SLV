


import numpy as np
import os
try: 
    os.chdir("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")
    # sys.path.append("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")
except FileNotFoundError:
    os.chdir("G:/Eigene Dateien/Studium/10. Semester/NIOZ/")
    # sys.path.append("G:/Eigene Dateien/Studium/10. Semester/NIOZ/")
from dMaps_SLV import dMaps_utils as dMaps
from dMaps_SLV.network import network_analysis_utils as nau
from dMaps_SLV.network import sla_wind_stress_animation_cmems_isobars_ERA5 as anim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import xarray as xr
import pandas as pd
import datetime as dt




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

#%% get pressure signal
# Import nc file with ERA5 data (pressure anomalies)
nc_fname = "data/ERA5/ERA5_wind_pressure_05_anomalies_2_deg.nc"
pa_ds = xr.open_dataset(nc_fname)
pa_lat = pa_ds.coords["lat"].values
lat_mask = pa_lat[(pa_lat>=np.min(lat)) & (pa_lat <= np.max(lat))]
time_mask = pa_ds.time[(pa_ds.time>=pa_ds.time[0]) & (pa_ds.time<=pa_ds.time[325])]
pa_da = pa_ds["msl"].loc[dict(expver=1, lat=lat_mask, time=time_mask)]

pa_signals = dMaps.get_domain_signals(domains = d_maps, 
                                     sla = pa_da, 
                                     lat = lat_mask, 
                                     signal_type = 'average')



#%% prepare data for maps
extent = [-150, -50, -66, 10]

domain_ids = [5, 7, 18, 25, 43, 49, 59]
fpath = "dMaps_SLV/results/dMaps/res_2_k_11_gaus/domain_identification/domain_maps.npy"
domains = np.load(fpath)
domain_mask = domains[domain_ids, :, :]
lat_dom = dMaps.importNetcdf(ncfile, 'lat')
lon_dom = dMaps.importNetcdf(ncfile, 'lon')

nc_fname = "data/ERA5/ERA5_wind_pressure_05_anomalies_4_deg.nc"
ds = xr.open_dataset(nc_fname)
 
ncfile = "dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"
ds_sla = xr.open_dataset(ncfile)

#%% import enso data
fpath = "data/climate_indices/nino_34_anomaly.txt"
enso34, enso34_time = nau.prep_clim_index(fpath)


#%% plot timeline of SLA

import cartopy.crs as ccrs
fig = plt.figure(figsize=(7,11), dpi=300)

ax1 = plt.subplot2grid((7, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((7, 2), (1, 0), colspan=2)
ax3 = plt.subplot2grid((7, 2), (2, 0), rowspan=2, projection=ccrs.PlateCarree())
ax4 = plt.subplot2grid((7, 2), (2, 1), rowspan=2, sharex = ax3, sharey = ax3,
                       projection=ccrs.PlateCarree())
ax5 = plt.subplot2grid((7, 2), (4, 0), rowspan=2, sharex = ax3, sharey = ax3,
                       projection=ccrs.PlateCarree())
ax6 = plt.subplot2grid((7, 2), (4, 1), rowspan=2, sharex = ax3, sharey = ax3,
                       projection=ccrs.PlateCarree())

ax1.plot(time, signals[:,49], label="SLA Domain 49")
ax1.plot(time, signals[:,25], label="SLA Domain 25")
ax1.grid()
ax1.legend(loc=4, ncol=2)
# ax1.set_title("SLA")
ax1.set_ylabel("[m]")
ax1.set_ylim([-0.08,0.055])
ax1.axes.xaxis.set_ticklabels([])
ax1.text(time[1]-dt.timedelta(days=350), 0.03, "a", fontweight='bold',
             bbox=dict(facecolor='w', alpha=0.9, zorder=49))


#% plot ENSO timeline

ax2.plot(enso34_time, enso34, label="Niño 3.4 SST Index")
ax2.grid()
ax2.legend()
ax2.set_ylabel("[°C]")
ax2.set_ylim([-2,2.5])
ax2.legend(loc=4)
ax2.text(time[1]-dt.timedelta(days=350), 1.75, "b", fontweight='bold',
             bbox=dict(facecolor='w', alpha=0.9, zorder=49))

# insert labels for CP and EP El Ninos
# According to Wiedermann et al. 2016 Table 1
# https://www.frontiersin.org/articles/10.3389/fclim.2021.618548/full
cp = [24, # 1994/1995
      120, # 2002/2003
      144, # 2004/2005
      204, # 2009/2010
      264] # 2014/2015
ep = [60, # 1997/1998
      168, #2006/2007
      276] # 2015/2016

for i in cp :
    ax2.text(time[i-5],1.9,"CP")
for i in ep :
    ax2.text(time[i-5],1.9,"EP")


# Mark periods of similar behaviour in red
start_times = [0, 22, 113, 80, 134, 192]
end_times = [9, 29, 120, 102, 148, 206]
for i in range(len(start_times)):
    start = time[start_times[i]]
    end = time[end_times[i]]
    width = end-start
    ax1.add_patch(Rectangle((start, -1000), width, 2000, alpha=0.3, fc='r'))
    ax2.add_patch(Rectangle((start, -1000), width, 2000, alpha=0.3, fc='r'))


# Plot anomaly maps
timesteps = [5, 22, 116, 195]
label = ["c", "d", "e", "f"]

axes = [ax3, ax4, ax5, ax6]
for i in range(len(timesteps)):
    timestep = timesteps[i]
    
    ax1.axvline(x=time[timestep], color='k', linestyle='dashed')
    if i < 2:
        text_y = -0.055
    else:
        text_y = 0.03
    ax1.text(time[timestep-3], text_y, label[i], fontweight='normal',
             bbox=dict(facecolor='w', alpha=0.9, zorder=49))
    
    year = pd.to_datetime(time[timestep]).year
    month = pd.to_datetime(time[timestep]).month
        
        
    v_east, v_north, velocity, lon_mask, lat_mask = anim.arrow_data(ds, 
                                                                extent, 
                                                                timestep,
                                                                param = "wind stress")
        
    ap, ap_lon, ap_lat = anim.ap_data(ds, extent, timestep)
        
    sla, sla_lon, sla_lat = anim.sla_data(ds_sla, extent, timestep)
        
    anim.arrow_plot_currents(extent, lon=lon_mask, lat=lat_mask, 
                            v_east=v_east, v_north=v_north, 
                            velocity=velocity,
                            title="", vmax=0.025, 
                            domain_data=domain_mask, domain_lat=lat_dom, 
                            domain_lon=lon_dom, 
                            sla_data=sla, sla_lon=sla_lon, sla_lat=sla_lat,
                            sla_vmin=-0.09, sla_vmax=0.09,
                            ap_data=ap, ap_lat=ap_lat, ap_lon=ap_lon,
                            arrow_label='Arrow color: \nWind stress anomaly [N m$^{-2}$]',
                            anomaly = True,
                            timestep=timestep,
                            ax=axes[i], fig=fig, draw_grid_labels=False) 
    


    axes[i].text(-145, 2 ,"$\\bf{"+label[i]+"}$ "+ "({month:02.0f}-{year})".format(
                                                      month=month[0],
                                                      year=year[0]), 
                 transform=ccrs.PlateCarree(), 
                 #fontsize=20,
                 bbox=dict(facecolor='w', alpha=0.9, zorder=49), 
                 zorder=50)

g1 = ax3.gridlines(draw_labels=True)
g1.top_labels = False
g1.right_labels = False
g1.bottom_labels = False

g2 = ax5.gridlines(draw_labels=True)
g2.top_labels = False
g2.right_labels = False

g3 = ax6.gridlines(draw_labels=True)
g3.top_labels = False
g3.right_labels = False
g3.left_labels = False

    
plt.savefig("dMaps_SLV/figures/fig_13_dom25_49_ts_synoptics.png", 
            bbox_inches='tight')



