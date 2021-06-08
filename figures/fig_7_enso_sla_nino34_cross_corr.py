

import numpy as np
import os
try:
    os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
except FileNotFoundError:
    os.chdir("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")    
from dMaps_SLV import dMaps_utils as dMaps
from dMaps_SLV.network import network_analysis_utils as nau
import pandas as pd
# import xarray as xr

fpath = "data/climate_indices/nino_34_anomaly.txt"
enso34, enso34_time = nau.prep_clim_index(fpath)

#%% get domain signal
dmaps_outpath = "dMaps_SLV/results/dMaps/res_2_k_11_gaus/"
# Import domain map
d_maps = np.load(dmaps_outpath + 'domain_identification/domain_maps.npy')
        
ncfile = "dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"
# Import SLA data and lat/lon data
sla = dMaps.importNetcdf(ncfile, "sla")
lat = dMaps.importNetcdf(ncfile, "lat")
        
        
# Produce domain signals for each domain in d_maps and time step in sla
signals = dMaps.get_domain_signals(domains = d_maps,
                                   sla = sla, 
                                   lat = lat, 
                                   signal_type = "average")
    
#%% calc cross correlations and plot timeseries

domain_id = 5
data1 = pd.Series(enso34) #pd.Series(signals[:,49])
data1_label = "Niño3.4 index"
data2 = pd.Series(signals[:,domain_id])
data2_label = "Domain 5 mean SLA"

out_fname = "dMaps_SLV/figures/fig_7_enso_sla_alongshore_flow.png"
nau.calc_plot_cross_corr(data1, data2, data1_label, data2_label,
                     data1_ylabel="SSTA [°C]",
                     data2_ylabel="SLA [cm]",
                     data1_lead_label="SSTA",
                     data2_lead_label="SLA",
                     time = enso34_time,
                     lag_range=range(12,-13,-1),
                     out_fname = out_fname)


