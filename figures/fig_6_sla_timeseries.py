





import os
try:
    os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
except FileNotFoundError:
    os.chdir("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")    
from dMaps_SLV import dMaps_utils as dMaps
import numpy as np
import matplotlib.pyplot as plt


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


#%% plot timeseries data

def plot_timeseries(x, y, label, ax, show_yticklabels=False):
    
    props = dict(boxstyle='Square', facecolor='w', alpha=1)
    # place a text box in upper left in axes coords
    ax.text(0.01, 0.95, label, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    ax.plot(x, y)
    ax.grid()
    ax.hlines(0, time[0], time[-1], color='k')
    if show_yticklabels==False:
        ax.set_xticklabels([])
    ax.set_ylabel("[m]")
    ax.set_xlim([time[0], time[-1]])


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(12,10))

plot_timeseries(time, signals[:,5], "Domain 5", ax1)
plot_timeseries(time, signals[:,49], "Domain 49", ax2)
plot_timeseries(time, signals[:,25], "Domain 25", ax3)
plot_timeseries(time, signals[:,7], "Domain 7", ax4, show_yticklabels=True)

plt.savefig("dMaps_SLV/figures/fig_6_sla_timeseries_sep.png",
            bbox_inches = 'tight')