# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:45:21 2021

@author: eikes
"""

def create_config_file (path_to_config,config_name,
                        path_to_file,
                        output_dir,
                        var_name,
                        lat_name='lat',lon_name='lon',
                        delta_samples=10000,
                        alpha=0.01,
                        k=8,
                        tau_max=12,
                        q=0.05):
#------------------------------------------------------
    """ Create configuration file.
    
    Function that creates the configuraiton file with the necessary parameters 
    to run deltaMAPS.
    
    Given an input netcdf file, it (1) remaps to the desired resoltuon; 
    (2) selects the time period; 
    (3) removes seasonal cycle; 
    (4) removes linear trend; 
    (5) selects the desired lat lon box,   
    and creates an output file, in the same path as the input file
    
    Paramaters 
    ----------
    path_to_config : str
                  path to save the cofniguration file. Example: 'pydMAPS/configs/'
    config_name: str
                name of the configuration file. Example:'config_example'
    path_to_file : str
                  path to the input file. Example: '~/working_folder/file.nc'
    output_dir : str
          name of directory with the output that will be created. Example: 'outputs'
    var_name : str
          name of the variable of interest in the netcdf file (e.g., "sst")
    lat_name: str
        name of the variable containing latitudes (e.g., 'lat', "latitude")
    lon_name: str
        name of the variable containing longitudes (e.g., 'lon', "longitude")
    delta_samples: int
        random sample of pairs of timeseries to estimate delta. 
        Default delta_samples =10000
    alpha: int
        significance level for the domain identification algorithm
        Default alpha=0.01
    k: int
        number of nearest neighbors to each grid cell i. The nearest neighbors 
        are computed using the Haversine distance 
        (https://en.wikipedia.org/wiki/Haversine_formula)
        Default k=8
    tau_max: int
        it defines the range of lags used in the network inference 
        (i.e., for each pair of domains signals A and B, the code will test the 
         statistical significance of correlations in the lag range R \in [-tau_max,tau_max])
        Default: tau_max=12
    q: int
        False Discovery rate (FDR) parameter to test the significance of the 
        lag-correlations (e.g., q = 0.05 implies that (on average) only 5% of 
        the links identified  is expected to be a false positive).
        Default: q=0.05
        

    Returns
    -------
    no return
    
    
    Usage
    --------
    create_config_file (path_to_config='~/py-dMaps/configs/',
                    config_name='config_example',
                        path_to_file='data/sst_output.nc',
                        output_dir='output_example',
                        var_name='sst',
                        lat_name='lat',lon_name='lon',
                        delta_samples=10000,
                        alpha=0.01,
                        k=8,
                        tau_max=12,
                        q=0.05)

    """
    #--- Create control.txt file for EstimateTrend
    
    file=str(path_to_config+config_name+'.json')
    fp = open(file,'w')
    #fp = open("./estimatetrend.ctl", "w")
    fp.write("{\n")
    fp.write('"path_to_data":"{0:s}",\n'.format(path_to_file))
    fp.write('"output_directory":"{0:s}",\n'.format(output_dir))
    fp.write('"variable_name":"{0:s}",\n'.format(var_name))
    fp.write('"latitude_name":"{0:s}",\n'.format(lat_name))
    fp.write('"longitude_name":"{0:s}",\n'.format(lon_name))
    fp.write('"delta_rand_samples":'+str(delta_samples)+',\n')
    fp.write('"alpha":'+str(alpha)+',\n')
    fp.write('"k":'+str(k)+',\n')
    fp.write('"tau_max":'+str(tau_max)+',\n')
    fp.write('"q":'+str(q)+'\n')
    
    fp.write("}\n")
    fp.close()
    
    return 


if __name__ == "__main__":
    
    import os
    os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
    
    create_config_file (path_to_config = "Playground/",
                        config_name = "config_sla_aviso",
                        path_to_file = "data/AVISO/AVISO_MSLA_1993-2020_detrend.nc",
                        output_dir = "Playground/output",
                        var_name = "sla",
                        lat_name='lat',lon_name='lon',
                        delta_samples=10000,
                        alpha=0.01,
                        k=8,
                        tau_max=12,
                        q=0.05)
    
    os.system("python py-dMaps/run_delta_maps.py -i playground/config_sla_aviso.json")