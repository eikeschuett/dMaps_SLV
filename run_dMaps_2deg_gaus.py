#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:22:51 2021

@author: root
"""

import sys
try: 
    sys.path.append("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
except FileNotFoundError:
    sys.path.append("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")
from dMaps_SLV import dMaps_utils as dMaps
import os
try:
    os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
except FileNotFoundError:
    os.chdir("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")    
    # os.chdir("G:/Eigene Dateien/Studium/10. Semester/NIOZ/")

infile = "dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"

geofile = 'dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc'

k_range = [8] #range(12,13,1)

for k in k_range: #range(20,0,-1):
    
    version = "res_2_k_" + str(k) + "_gaus"
    outdir = "dMaps_SLV/data/dMaps/" + version + "/"
    
    path_to_config = outdir # "dMaps_SLV/data/dMaps/"
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
                                  tau_max=6,
                                  q=0.05)
        
    dMaps.run_dMaps(config_file_name = path_to_config+config_name+".json",
                    dmaps_file_name = "py-dMaps/run_delta_maps.py")
    
    dMaps.plot_dMaps_output(geofile = geofile,
                                   fpath = outdir, 
                                   output = 'all',
                                   show_seeds = 'homogeneity',
                                   outpath = 'dMaps_SLV/plots/dMaps/' + 
                                                   version + "/") 
    
    dMaps.plot_network(fpath = outdir, 
                       geofile = geofile, 
                       out_fpath = 'dMaps_SLV/plots/dMaps/' + 
                                                   version + "/network.png")

#%% NMI calculation to find best value for k
    
# Path to the domain maps
#path = 'playground/output/'
#nmi_matrix = dMaps.calc_nmi_matrix(path = path, 
#                             res = 2, 
#                             k = k_range,
#                             path_end = '_gaus_a_1/domain_identification/domain_maps.npy')
    
#dMaps.plot_nmi_matrix(nmi_matrix=nmi_matrix,
#                      k = k_range,
#                      fname="playground/plots/2_deg_gaus_a_1/nmi_matrix_res_2_gaus_a_1.png")
