#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:22:51 2021

@author: root
"""

import sys
sys.path.append("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
from dMaps_SLV import dMaps_utils as dMaps
import os
os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")

infile = "data/AVISO/AVISO_MSLA_1993-2020_prep_1_deg_gaus.nc"

k_range = range(13,14)

for k in k_range: #range(20,0,-1):
    
    version = "res_1_k_" + str(k) + "_gaus"
    outdir = "playground/output/" + version
    
    path_to_config = "playground/"
    config_name = "config_sla_aviso_gaus"
                
    dMaps.create_config_file (path_to_config = path_to_config,
                                  config_name = config_name,
                                  path_to_file = infile,
                                  output_dir = outdir,
                                  var_name = "sla",
                                  lat_name='lat',lon_name='lon',
                                  delta_samples=10000,
                                  alpha=0.01,
                                  k=k,
                                  tau_max=1,
                                  q=0.05)
        
    dMaps.run_dMaps(config_file_name = path_to_config+config_name+".json",
                    dmaps_file_name = "py-dMaps/run_delta_maps.py")
    
    dMaps.plot_dMaps_output(geofile = 'data/AVISO/AVISO_MSLA_1993-2020_prep_1_deg.nc',
                                   fpath = outdir, 
                                   output = 'all',
                                   outpath = 'playground/plots/1_deg_gaus/' + 
                                                   version + "/") 

#%% NMI calculation to find best value for k
    
# # Path to the domain maps
# path = 'playground/output/'
# nmi_matrix = dMaps.calc_nmi_matrix(path = path, 
#                              res = 1, 
#                              k = k_range,
#                              path_end = '_gaus/domain_identification/domain_maps.npy')
    
# dMaps.plot_nmi_matrix(nmi_matrix=nmi_matrix,
#                       k = k_range,
#                       fname="playground/plots/1_deg_gaus/nmi_matrix_res_1_gaus.png")