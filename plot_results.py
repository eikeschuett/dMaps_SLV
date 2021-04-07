#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:01:12 2021

@author: root
"""


import os
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs



#import netCDF4
from netCDF4 import Dataset
def importNetcdf(path,variable_name):
    nc_fid = Dataset(path, 'r')
    field = nc_fid.variables[variable_name][:]     
    return field 

def plot_map(lat, lon, data, title):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1,1,1, projection = ccrs.Robinson())
    ax.set_global()
    ax.coastlines('110m', alpha=0.1)
    
    filled_c = ax.contourf(lon, lat, data, transform=ccrs.PlateCarree())

    fig.colorbar(filled_c, orientation='horizontal')
    ax.set_title(title)

    plt.savefig("playground/plots/" + title + ".png")
    #plt.show()

os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
         
         
path = 'data/AVISO/AVISO_MSLA_1993-2020_detrend.nc'

lon = importNetcdf(path,'lon')
lat = importNetcdf(path,'lat')



    
    
#%% Plots of homogeneity fielf and seeds location    
    
# Path to the homogeneity field
path_homogeneity = 'playground/output/seed_identification/local_homogeneity_field.npy'
# Path to the seed positions
path_seeds = 'playground/output/seed_identification/seed_positions.npy'
# Import the homogeneity field
homogeneity_field = np.load(path_homogeneity)
# Import the seeds location
seeds = np.load(path_seeds)

plot_map(lat, lon, homogeneity_field, 'local homogeneity field')

plot_map(lat, lon, seeds, "seeds")



#%% Plot single domain map

# Path to the domain maps
path_domain_maps = 'playground/output/domain_identification/domain_maps.npy'
# Path to the domain ids
path_domain_ids = 'playground/output/domain_identification/domain_ids.npy'
# Import domain maps
d_maps = np.load(path_domain_maps)
# Import domain ids
d_ids = np.load(path_domain_ids)


domain_map = np.zeros((d_maps.shape[1], d_maps.shape[2]))
i = 1
for d in range(len(d_maps)):
    domain_map[d_maps[d] == 1] = i;
    i += 1

domain_map[domain_map==0] = np.nan

plot_map(lat, lon, domain_map, "Domain map")


