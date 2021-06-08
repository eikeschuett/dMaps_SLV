

import numpy as np
import os
try:
    os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
except FileNotFoundError:
    os.chdir("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")    
from dMaps_SLV import dMaps_utils as dMaps
from dMaps_SLV.network import network_analysis_utils as nau
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import xarray as xr


# import xarray as xr


extent1 = [-98, -80, -45, -31]
#  extent2 = [-80, -70, -20, -10]
extents = [extent1]#, extent1]
regions = ["between_49","peru"]

domain_id = 25

for i, extent in enumerate(extents):
    # print(extent)
    # Min/Max Extent to Lat/Lon-Dict
    pos_dict = []
    #for extent in extents:
    pos_lat = [extent[3], extent[2], extent[2], extent[3], extent[3]]
    pos_lon = [extent[0], extent[0], extent[1], extent[1], extent[0]]
    
    pos_dict_temp = {"lat": pos_lat,
                     "lon": pos_lon}
    pos_dict.append(pos_dict_temp)
    
    extent = [extent]        
            
    # Plot positions into the domain map
    geofile = 'dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc'
    lon = dMaps.importNetcdf(geofile,'lon')
    lat = dMaps.importNetcdf(geofile,'lat')
            
    dmaps_path = 'dMaps_SLV/results/dMaps/res_2_k_11_gaus/'
    # Import domain maps
    d_maps = np.load(dmaps_path + 'domain_identification/domain_maps.npy')
    # Create array containing the number of each domain
    domain_map = dMaps.get_domain_map(d_maps)
        
    
    #%% Get flow for extent
    
    nc_fname = "data/CMEMS/CMEMS_phy_030_uo_vo_mlotst_1993_2018_deseasoned_2_deg.nc"
    fl_ds = xr.open_dataset(nc_fname)
    
    
    v_east, v_north = nau.mean_areal_flow(ds=fl_ds, extents=extent, param="flow", 
                                          window=7, cutoff = 1./6.)
    
    
    #%% get domain signals
    dmaps_outpath = "dMaps_SLV/results/dMaps/res_2_k_11_gaus/"
    # Import domain map
    d_maps = np.load(dmaps_outpath + 'domain_identification/domain_maps.npy')
            
    ncfile = "dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"
    # Import SLA data and lat/lon data
    sla = dMaps.importNetcdf(ncfile, "sla")
    lat = dMaps.importNetcdf(ncfile, "lat")
    time = dMaps.importNetcdf(ncfile, "time")
            
            
    # Produce SLA domain signals for each domain in d_maps and time step in sla
    signals = dMaps.get_domain_signals(domains = d_maps,
                                       sla = sla, 
                                       lat = lat, 
                                       signal_type = "average")
    
    
    #%%
    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(nrows=2, ncols=4)
    
    ax1 = fig.add_subplot(gs[:, 0:2], projection=ccrs.PlateCarree())
    
                    
    dMaps.plot_map(lat = lat, 
                   lon = lon, 
                   data = domain_map,
                   seeds = None,
                   title = "",
                   cmap = 'prism',
                   alpha = 0.3,
                   show_colorbar=False,
                   show_grid=True,
                   outpath = None, # 'playground/plots/Network/flow/dom_25_pos',
                   labels = True,
                   extent = [-110, -70, -63, -7],
                   pos_dict = pos_dict,
                   draw_box = True,
                   ax = ax1)  
    
    
    #%%
    ax2 = fig.add_subplot(gs[0, 2:])
    
    nau.calc_ax_cross_corr(pd.Series(signals[:,domain_id]), v_north, time, ax=ax2,
                       lag_range=range(12,-13,-1),
                       out_fname = None,
                       return_crosscorr = False,
                       # data1_lead_label="SLA",
                       # data2_lead_label="Flow",
                       )
    ax2.text(0,0.8,"Nothward flow", fontweight='bold', fontsize=12)
    ax2.set_yticks([-1,-0.5, 0, 0.5, 1])
    ax2.set_xticklabels([])
    ax2.set_xlabel("")
    #%%
    
    ax3 = fig.add_subplot(gs[1, 2:])
    
    nau.calc_ax_cross_corr(pd.Series(signals[:,domain_id]), v_east, time, ax=ax3,
                       lag_range=range(12,-13,-1),
                       out_fname = None,
                       return_crosscorr = False,
                       # data1_lead_label="SLA",
                       # data2_lead_label="Flow",
                       )
    ax3.text(0,0.8,"Eastward flow", fontweight='bold', fontsize=12)
    ax3.text(1,-1.3,"Flow leading")
    ax3.text(18,-1.3,"SLA leading")
    ax3.set_yticks([-1,-0.5, 0, 0.5, 1])
    
    
    
    fig.subplots_adjust(wspace=0.4)
    
    out_fname = "dMaps_SLV/figures/fig_10_flow_between_49_25.png"
    
    plt.savefig(out_fname, bbox_inches = 'tight')