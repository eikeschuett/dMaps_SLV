#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:49:11 2021

@author: root
"""

import sys
sys.path.append("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/py-dMaps")

from Utils import utils
from netCDF4 import Dataset
import numpy as np

path_to_data = "/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/data/AVISO/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"

nc_fid = Dataset(path_to_data, 'r')
data = nc_fid.variables['sla'][:]

results = np.zeros([14,51])
for i, delta_rand_samples in enumerate(np.arange(5000,16000,1000)):
    alpha = 0.01
    results[i,0] = delta_rand_samples
    for k in range(0,50):
        delta = utils.estimate_delta(data, delta_rand_samples, alpha);
        results[i,k+1] = delta
        print("Done with i "+ str(i) + " k " + str(k))
    
np.save("calc_delta_res2.npy", results)


