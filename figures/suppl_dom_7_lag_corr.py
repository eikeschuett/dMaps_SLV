

# Plots laggeed-correlation between domain signal of domain 7, the domain 
# signal of domain 25 and the Nino3.4 index

import numpy as np
import os
try:
    os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
except FileNotFoundError:
    os.chdir("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")    
from dMaps_SLV import dMaps_utils as dMaps
from dMaps_SLV.network import network_analysis_utils as nau
import pandas as pd


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


#%% import enso data
fpath = "data/climate_indices/nino_34_anomaly.txt"
enso34, enso34_time = nau.prep_clim_index(fpath)

#%% cross correlation between the two domains

domain_1 = 25
domain_2 = 7

data_1 = pd.Series(signals[:,domain_1])
data_2 = pd.Series(signals[:,domain_2])

out_fname = "dMaps_SLV/figures/supplementary/lag_corr_domain_7/lag_corr_sla_dom_7_dom_25.png"

nau.calc_plot_cross_corr(data_1, data_2, 
                         "{dom1} SLA".format(dom1=domain_1), 
                         "{dom2} SLA".format(dom2=domain_2), 
                         data1_ylabel="[m]",
                         data1_lead_label="Dom. {domain_1} SLA".format(domain_1=domain_1),
                         data2_ylabel="[m]",
                         data2_lead_label="Dom. {domain_2} SLA".format(domain_2=domain_2),
                         time = time,
                         lag_range=range(12,-13,-1),
                         out_fname = out_fname)

#%% cross corr domain 7 with Nino34
domain_1 = 7

data_1 = pd.Series(signals[:,domain_1])
data_2 = enso34

out_fname = "dMaps_SLV/figures/supplementary/lag_corr_domain_7/lag_corr_sla_dom_7_nino34.png"

nau.calc_plot_cross_corr(data_1, data_2, "Domain 7 SLA", "Niño 3.4 SST Index",
                         data1_ylabel="[m]",
                         data1_lead_label="Dom. {domain_1} SLA".format(domain_1=domain_1),
                         data2_ylabel="[°C]",
                         data2_lead_label="Nino3.4",
                         time = time,
                         lag_range=range(12,-13,-1),
                         out_fname = out_fname)