
# Plots laggeed-correlation between domain signal and average surface pressure
# anomaly in a domain

import numpy as np
import os
try:
    os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
except FileNotFoundError:
    os.chdir("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")    
from dMaps_SLV import dMaps_utils as dMaps
from dMaps_SLV.network import network_analysis_utils as nau
import pandas as pd
import xarray as xr


#%% get domain signal
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

#%% Import and produce domain pressure signals

# Import nc file with ERA5 data (wind anomalies)
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



#%% Pressure in domain 18 vs SLA in domains 25 and 7
domains_1 = [25, 7, 25, 49]
domains_2 = [18, 18, 59, 59]

fnames = ["fig_12_lag_corr_pa_18_sla_25.png",
          "fig_15_lag_corr_pa_18_sla_7.png",
          "supplementary/lag_corr_SLPa_SLA/lag_corr_pa_18_sla_25.png",
          "supplementary/lag_corr_SLPa_SLA/lag_corr_pa_18_sla_49.png"]


for i in range(len(domains_1)):
    domain_1 = domains_1[i]
    domain_2 = domains_2[i]

    data1 = pd.Series(signals[:,domain_1])
    data1_label = "Domain {domain} SLA".format(domain=domain_1)
        
    data2 = nau.low_pass_filter(data = pd.Series(pa_signals[:,domain_2]), 
                                window = 7, 
                                cutoff = 1./6.)
    
    data2_label = "Domain {domain} SLPa (7 month low-pass filtered)".format(domain=domain_2)
       
    # out_fname = "playground/plots/Network/sla_pressure_correlation/domain_{domain_1}_pa_{domain_2}_low_pass.png".format(domain_1=domain_1,
    #                                                                                                                     domain_2=domain_2)
    out_fname = "dMaps_SLV/figures/" + fnames[i]
    
    nau.calc_plot_cross_corr(data1, data2, data1_label, data2_label,
                             data1_ylabel="[m]",
                             data1_lead_label="Dom. {domain_1} SLA".format(domain_1=domain_1),
                             data2_ylabel="[Pa]",
                             data2_lead_label="Dom. {domain_2} SLPa".format(domain_2=domain_2),
                             time = time,
                             lag_range=range(12,-13,-1),
                             out_fname = out_fname,
                             leg_loc="lower right")