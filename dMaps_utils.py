#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:12:20 20212

@author: root
"""

#%% Prepare data

def aviso_download(url, cutdirs, username, password, subdir):
    """
    Uses wget to download all *.nc.gz files from a URL into a new subfolder of
    the current working directory

    Parameters
    ----------
    url : str
        URL that contains the *.nc.gz files that should be downloaded.
    cutdirs : int
        number of parent directories of the URL, e.g. for 
        "ftp://ftp-access.aviso.altimetry.fr/climatology/global/delayed-time/monthly_mean/"
        the data is in the 5th directory level and thus the cutdirs must be 5.
        Otherwise the directory structure of the ftp server will be copied and
        cause problems to find the data later.
    username : str
        Username for the ftp server.
    password : str
        Password for the ftp server.
    subdir : str
        Name of a new subdirectory from the current working directory that will
        be created and used to store the data.

    Returns
    -------
    None.

    Usage
    -------
    aviso_download(url = "ftp://ftp-access.aviso.altimetry.fr/climatology/global/delayed-time/monthly_mean/"
                   cutdirs = 5, 
                   username = "eike.schutt@nioz.nl",
                   password = "password", 
                   subdir = "raw/")
    """
    import os
    # Create new directory to store the raw data in
    if not os.path.exists(subdir):
        os.makedirs(subdir)
        print("New subdirectory " + str(subdir) + " created")
    print("start download")
    
    # -c continue
    # -r recursive download
    # -np no parent
    # -nH no host directories
    # -P all files will be saved into the new folder
    os.system('wget -c -r -np -nH -A nc.gz --user ' + str(username) + 
              ' -P ' + str(subdir) +
              ' --password ' + str(password) + 
              ' --cut-dirs=' + str(cutdirs) +
              ' -erobots=off --no-check-certificate ' + str(url))

def unzip(indir, outdir):
    """
    Unzips all *gz in the indir to the outdir.

    Parameters
    ----------
    indir : str
        Directory with all *.nc.gz files. Needs to have "/" at the end.
    outdir : str
        Directory where all unzipped files will be stored. Needs to have "/" 
        at the end.

    Returns
    -------
    None.

    Usage
    -------
    unzip("raw/", "unzip/")
    """
    import os
    import gzip
    import shutil
    
    # Create new directory to store the unzipped data in
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print("New subdirectory " + str(outdir) + " created")
    
    search_path = indir
    file_type = ".gz"
    # Iterate over each file in the indir and unzip it
    for fname in os.listdir(path=search_path):
        if fname.endswith(file_type):
            with gzip.open(indir + fname,'rb') as f_in:
                with open(outdir + fname[:-3],'wb') as f_out:
                    shutil.copyfileobj(f_in,f_out)
                    
def concat_nc_files(indir, outdir, fname):
    """
    Concatenates all nc-files in the indir-directory to a single nc-file with 
    a time dimension that will be stored in the outdir-directory as fname.

    Parameters
    ----------
    indir : str
        Directory of all nc-files with no time dimension. Needs to have a "\" 
        at the end.
    outdir : str
        Directory where the final nc-file will be stored. Needs to have a "\" 
        at the end. Can be left empty ("") if the file should be stored in the
        current working directory.
    fname : str
        Output filename.

    Returns
    -------
    None.
    
    Usage
    -------
    concat_nc_files(indir = "unzip/", 
                    outdir = "", 
                    fname = "AVISO_MSA_1993-2020.nc")    

    """
    import os
    os.system("cdo mergetime " + 
              str(indir) + "*.nc " +    # input dir and files
              str(outdir) + str(fname)) # output dir and files
    
def get_tmp_fname():
    import tempfile
    _, temp_fname = tempfile.mkstemp()
    temp_fname = temp_fname + ".nc"
    return temp_fname

def nc_prep(infile, outfile, res=None, lonlatbox=None):
    """
    Uses CDO to resample, remove seasonality and linear trend and crop to a 
    specific region a nc-file with a time dimension.

    Parameters
    ----------
    infile : str
        Filename of nc-file that is to be detrended.
    outfile : str
        Filename of output nc-file.
    res : int
        Desired output resolutions [degrees]. Can have values 1 or 2. If other
        values are specified, the nc-file will not be resampled.
    lonlatbox : list of int
        Longitude-Latitude Box to which the nc-file will be cropped. Is in 
        format [lonmin, lonmax, latmin, latmax]. If lonlatbox is unspecified,
        the nc-file will not be cropped.

    Returns
    -------
    None.

    Usage
    -------
    nc_prep(infile = "AVISO_MSA_1993-2020.nc", 
            outfile = "AVISO_MSA_1993-2020_detrend.nc",
            res = 2,
            lonlatbox = [0, 360, -60, 60])

    """
    import os
    # Resample nc file to desired resolution and safe temporarily. 
    temp_fname_res = get_tmp_fname()
    if res==1:
        os.system('cdo -L remapbil,r360x180 ' + 
                  str(infile) + ' ' + str(temp_fname_res))
    elif res==2:
        os.system('cdo -L remapbil,r180x90 ' + 
                  str(infile) + ' ' + str(temp_fname_res))
    else:
        os.system('cp ' + str(infile) + ' ' + str(temp_fname_res))
    
    # Remove seasonality and safe it temporarily
    temp_fname_sea = get_tmp_fname()
    os.system("cdo -L -ymonsub " + str(temp_fname_res) + 
              " -ymonmean " + str(temp_fname_res) + " " +
              temp_fname_sea)
    
    # Remove linear trend
    temp_fname_lin = get_tmp_fname()
    os.system("cdo detrend "+
              temp_fname_sea + " "+
              temp_fname_lin)
    
    # Crop to lonlat-box
    if lonlatbox == None:
        os.system('cp ' + str(temp_fname_lin) + ' ' + str(outfile))
    else:
        os.system("cdo sellonlatbox," + 
                  ','.join([str(i) for i in lonlatbox ]) + #convert lonlatbox 
                                      # to sequence of strings separated by ","
                  " " + temp_fname_lin + " " + str(outfile))
    
def gaus_filter(U, sigma=0.8, truncate=4.0):
    """
    Applies a gaussian filter to a numpy array that is not disturbed by NaNs.
    The code is adapted from an answer to this question:
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    This code uses the gaussian filter from scipy image. It calculates the 
    kernel size internally based on sigma and the truncate parameters as
    int(truncate * sigma + 0.5).

    Parameters
    ----------
    U : numpy array
        Array of the data to which the filter shall be applied.
    sigma : float, optional
        Standard deviation of the gaussian filter. The default is 0.8.
    truncate : float, optional
        Truncate filter at this many sigmas. The default is 4.0.

    Returns
    -------
    Z : TYPE
        DESCRIPTION.

    """
    import numpy as np
    from scipy.ndimage import gaussian_filter
       
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = gaussian_filter(V,sigma=sigma,truncate=truncate)
    
    W = 0*U.copy()+1
    W[np.isnan(U)] = 0
    WW = gaussian_filter(W,sigma=sigma,truncate=truncate)
    
    # replace land with nan again to avoid invalid value in true divide
    WW[np.isnan(U)] = np.nan
    Z = VV/WW

    return Z    

def nc_gaus_filter(infile, outfile, var, sigma=1.0, truncate=3.0):
    """
    Reads a variable from an nc-file, applies a gaussian filter and overwrites
    the original data of that variable in the nc-file with the modified data.


    Parameters
    ----------
    infile : string
        Path and filename of the input nc-file.
    outfile : string
        desired output filename.
    var : string
        Name of the variable in the nc-file that shall be filtered.
    sigma : float, optional
        Standard deviation of the gaussian filter. The default is 1.0.
    truncate : float, optional
        Truncate filter at this many sigmas. The default is 3.0.

    Returns
    -------
    None.

    """

    import netCDF4 as nc
    import numpy as np
    import os
    
    # copy input file to the output file
    os.system('cp ' + str(infile) + ' ' + str(outfile))
    
    ds = nc.Dataset(outfile, 'r+') # open the copied file
    # read the variable data as normal numpy array (not as masked array)
    data = ds[var][:].filled(np.nan)
    data = gaus_filter(data) # Aplly gaussian filter
    ds[var][:] = data # replace old data in dataset by the filtered one
    ds.close() # write data on disk

#%% run dMaps

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

def run_dMaps(config_file_name,
              dmaps_file_name= "/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/py-dMaps/run_delta_maps.py"):
    """
    Runs deltaMaps through the console.

    Parameters
    ----------
    config_file_path : string
        (Relative) File path and file name of the config file created with the
        "create_config_file" function.
    dmaps_fpath : str, optional
        (Relative) File path and file name of the orginal "run_delta_maps.py". 
        The default is "/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/py-dMaps/run_delta_maps.py".

    Returns
    -------
    None.

    """
    import os
    os.system("python "+ dmaps_file_name + " -i " + config_file_name)

#%% Plotting

def importNetcdf(path,variable_name):
    """
    Imports a variable of a netCDF file as a masked array.

    Parameters
    ----------
    path : string
        Path to nc-file.
    variable_name : str
        Name of variable in nc-file.

    Returns
    -------
    field : masked array
        Imported data of the variable of the nc-file.
    """
    from netCDF4 import Dataset
    nc_fid = Dataset(path, 'r')
    field = nc_fid.variables[variable_name][:]     
    return field 

def plot_map(lat, lon, data, seeds, title, outpath=None):
    """
    Plots a contourplot in a map with a title. If an output-path is specified,
    the plot is saved as <title>.png in the output directory. If this directory
    does not exist already, it will be created first.

    Parameters
    ----------
    lat : TYPE
        Latitude coordinates of the data-array.
    lon : TYPE
        Longitude coordinates of the data-array.
    data : array
        Array containing the data that will be plotted.
    seeds : array or None
        Array containing the locations of the seeds (cells without seed=0, 
        cells with seed=1) or None. If None, no seeds will be plotted.
    title : string
        Title of the plot [and output filename if outpath is specified].
    outpath : string, optional
        Path where the plot will be saved. The default is None.

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import os
    import numpy as np
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1,1,1, 
                         projection = ccrs.PlateCarree(central_longitude=180))
    ax.set_global()
    ax.coastlines('110m', alpha=0.1)
    
    # Alternative to contourf: plot the "real" raster using pcolormesh
    # filled_c = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
    #                          cmap='gist_ncar')
    filled_c = ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(), levels=100, 
                               cmap='viridis')
    
    if type(seeds) == np.ndarray:
        # Get index of all seed locations and get their lat/lon coordinates
        y, x = np.where(seeds==1)
        y_lat = lat[y]
        x_lon = lon[x]
        # Plot each seed location
        for i in range(len(x_lon)):
            ax.plot(x_lon[i], y_lat[i], marker='.', c='r', markersize=2, 
                    transform=ccrs.PlateCarree())

    fig.colorbar(filled_c, orientation='horizontal')
    ax.set_title(title)

    if outpath==None:
        plt.show()
    else:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        plt.savefig(outpath + title + '.png')
        plt.close()        

def plot_dMaps_output(geofile, 
                      fpath, 
                      output = 'domain', 
                      outpath=None, 
                      show_seeds=False):
    """
    Function to plot the output of deltaMaps. By default, it plots a map of all
    domains, but it can also visualize the local homogeneity and the location 
    of the seeds as overlay. If no output path (outpath) is specified, the 
    plots will not be saved. If an output path is specified that does not 
    exist, it will be created by plot_map()-function.

    Parameters
    ----------
    geofile : string
        Path to the dataset (nc-file) that has been used for the clustering. 
        (required to get the lat/lon grid.)
    fpath : string
        Path to the directory where deltaMaps saved its results. Must contain
        the subdirectories "domain_identification" and "seed_identification".
    output : string, optional
        Desired extent of output (maps that will be produced). Can take the 
        following values:
            'all' -> plots local homogeneity map and domain map
            'domain' -> plots domain map only
            'homogeneity' -> plots homogeneity map only
        The default is 'domain'.
    outpath : string or None, optional
        Path to the directory where the plots will be stored. If an output path
        is specified that does not exist, it will be created by plot_map()-
        function. If None is given, the plots will not be saved. The default 
        is None.
    show_seeds : string or None, optional
        Specifies whether the seeds locations will be plotted onto the maps. 
        Can take the following values:
            False -> seeds locations will not be plotted
            True -> seeds locations will be plotted on all maps
            'homogeneity' -> seeds locations will be plotted only on the 
                             homogeneity map
        The default is False.

    Returns
    -------
    None.

    Usage
    -------
    plot_dMaps_output(geofile = "data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc",
                      fpath = "playground/output/res_2_k_5/", 
                      output = 'all', 
                      outpath = None,
                      show_seeds = 'homogeneity')

    """
    
    import numpy as np
    # import lat/lon vectors
    lon = importNetcdf(geofile,'lon')
    lat = importNetcdf(geofile,'lat')
    
    if show_seeds == False:
        seeds = None
    else:
        seeds = np.load(fpath + '/seed_identification/seed_positions.npy')  
    
    if output == 'all' or output == 'homogeneity':
        # Import homogeneity field
        homogeneity_field = np.load(fpath + 
                        '/seed_identification/local_homogeneity_field.npy')
 
        plot_map(lat = lat, 
                 lon = lon, 
                 data = homogeneity_field, 
                 seeds = seeds,
                 title = 'local homogeneity field',
                 outpath = outpath) 
        
    if output == 'all' or output == 'domain':
        if show_seeds=='homogeneity':
            seeds = None
        # Import domain maps
        d_maps = np.load(fpath + '/domain_identification/domain_maps.npy')
        # Create array containing the number of each domain
        domain_map = np.zeros((d_maps.shape[1], d_maps.shape[2]))
        i = 1
        for d in range(len(d_maps)):
            domain_map[d_maps[d] == 1] = i
            i += 1
        domain_map[domain_map==0] = np.nan
                
        plot_map(lat = lat, 
                  lon = lon, 
                  data = domain_map,
                  seeds = seeds,
                  title = "Domain map",
                  outpath = outpath)         

#%% Heuristic to determine best k

def get_domain_vec(path_domain_maps):
    """
    Imports the deltaMaps domain map file and produces a numpy vector with the
    assignment of each grid cell to a domain. All grid cells which are in no 
    domain have the value 0. All overlaps between domains are assigned as new 
    domains (i.e. values > len(d_maps)+1).

    Parameters
    ----------
    path_domain_maps : string
        Path to domain maps numpy file (i.e. something like 
                                ".../domain_identification/domain_maps.npy".

    Returns
    -------
    domain_vec : 1D-numpy array of float64
        Numpy vector containing the assignment of each grid cell to a domain.
        All grid cells which are in no domain have the value 0. All overlaps 
        between domains are assigned as new domains.

    """
    import numpy as np
    # Import domain maps
    d_maps = np.load(path_domain_maps)
           
    domain_map = np.zeros((d_maps.shape[1], d_maps.shape[2]))
    i = 1
    k = len(d_maps)+2
    for d in range(len(d_maps)):
        # Account for possible overlaps between two domains: If a domain
        # is assigned to a grid cell which already is in a domain, assign
        # it to a new domain. Overlaps will start at len(d_maps)+1!
        domain_map[np.logical_and(d_maps[d] == 1, 
                                  domain_map != 0)] = k
        k += 1
        # If the grid cell is not assigned to a domain already, copy the
        # original domain number
        domain_map[np.logical_and(d_maps[d] == 1, 
                                          domain_map == 0)] = i
        i += 1
            
    domain_vec = np.concatenate(domain_map)
    return domain_vec


def calc_nmi_matrix(path, res, k, 
                    path_end = '/domain_identification/domain_maps.npy'):

    """
    Calculates a matrix of Normalized Mutual Information between the results
    of deltaMaps for different Neighborhood-sizes (K) based on scikit-learns
    NMI-metric.

    Parameters
    ----------
    path : string
        Path to the directory with the different dMaps outputs for the
        different values of k.
    res : int
        Resolution that shall be assessed.
    k : range
        Range of k for which the output of dMaps is available unter the 
        specified filepath and for which the NMI matrix shall be created. 
    path_end : string, optional
        Path from root directory of dMaps run to the numpy file containing the
        domain maps. The default is '/domain_identification/domain_maps.npy'.

    Returns
    -------
    nmi_matrix : numpy array
        Array containing the NMI for different combinations of K-values.

    """
    import numpy as np   
    from sklearn.metrics.cluster import normalized_mutual_info_score
    nmi_matrix = np.zeros((len(k), len(k)))
    
    for row,row_k in enumerate(k):
        pname = 'res_' + str(res) + '_k_' + str(row_k)
        row_d_vec = get_domain_vec(path+pname+path_end)
        
        for col, col_k in enumerate(k):
            pname = 'res_' + str(res) + '_k_' + str(col_k)
            col_d_vec = get_domain_vec(path+pname+path_end)   
            
            nmi_matrix[row-1, col-1] = normalized_mutual_info_score(row_d_vec,
                                                                    col_d_vec)


    return nmi_matrix

def plot_nmi_matrix(nmi_matrix, k, fname=None):
    """
    Produces a contourf-plot of the NMI-matrix and saves the output in a 
    specified filepath.

    Parameters
    ----------
    nmi_matrix : numpy array
        The NMI matrix from calc_nmi_matrix().
    k : range
        Range of the k-values to be plotted (i.e. x and y values of the NMI 
        matrix). Must be ascending (otherwise the image itself is flipped)!
    fname : string, optional
        Desired output filepath and name. If none is specified, the plot will 
        be shown and not saved. The default is None.

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(dpi=300)
    dat = ax.contourf(k, k, nmi_matrix, 
                      cmap='jet', 
                      levels = np.linspace(0.2, 1.0, 100))
    plt.colorbar(dat, 
                 ax=ax, 
                 extend='both', 
                 ticks=np.arange(0.2, 1.2, 0.2))
    ax.set_ylabel("K")
    ax.set_xlabel("K")
    ax.set_xlim([min(k),max(k)])
    ax.set_ylim([min(k),max(k)])
    ax.set_yticks(np.arange(min(k),max(k)+1,2))
    ax.set_xticks(np.arange(min(k),max(k)+1,2))
    ax.set_aspect('equal', 'box')   
    if fname == None:
        plt.show()
    else:
        plt.savefig(fname)


#%% Download and prepare dataset
if __name__ == "__main__":
    
    # import sys
    # sys.path.append("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
    # from dMaps_SLV import dMaps_utils as dMaps
    import os
    os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/data/AVISO/")
    
    #%% Download the AVISO data
    url = "ftp://ftp-access.aviso.altimetry.fr/climatology/global/delayed-time/monthly_mean/"
    cutdirs = 5
    username = "eike.schutt@nioz.nl"
    password = "cG3mLH"
    subdir = "raw2/"
    aviso_download(url, cutdirs, username, password, subdir)
    
    #%% Unzip the downloaded data
    outdir = "unzip/"
    unzip(indir = subdir, 
          outdir = outdir)
    
    #%% Concatenate all nc files
    # Since the dataset has one nc-file per month but deltaMaps requieres a single
    # nc file with a time dimension, the mergetime command from CDO is used to
    # merge all nc files into a single one.
    
    concat_nc_files(indir = "unzip/", 
                    outdir = "", 
                    fname = "AVISO_MSLA_1993-2020.nc")   
    
    #%% Remove seasonality and linear trend
        
    nc_prep(infile = "AVISO_MSLA_1993-2020.nc", 
            outfile = "AVISO_MSLA_1993-2020_prep_2_deg.nc",
            res = 2,
            lonlatbox = [0, 360, -66, 66])

    #%%% aplly Gaussian filter

    nc_gaus_filter(infile = "AVISO_MSLA_1993-2020_prep_2_deg.nc", 
                   outfile = "AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc",
                   var = "sla", 
                   sigma=1.0, truncate=3.0)


    #%% Run delta maps for a range of k values and plot the results
    os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
    
    # Define dataset that will be used by deltaMaps
    infile = "data/AVISO/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"
    
    # Define range of k-values (neighborhood size)
    k_range = range(1,21,1)
    
    for k in k_range:
        # Define path and filename of output of deltaMaps
        version = "res_2_k_" + str(k) + "_gaus"
        outdir = "playground/output/" + version
        
        path_to_config = "playground/"
        config_name = "config_sla_aviso_gaus"
        # Create the configuration file
        create_config_file(path_to_config = path_to_config,
                           config_name = config_name,
                           path_to_file = infile,
                           output_dir = outdir,
                           var_name = "sla",
                           lat_name = 'lat',lon_name = 'lon',
                           delta_samples = 10000,
                           alpha = 0.01,
                           k = k,
                           tau_max = 12,
                           q = 0.05)
        
        # Run deltaMaps with the configuration file that was just created
        run_dMaps(config_file_name = path_to_config+config_name+".json",
                        dmaps_file_name = "py-dMaps/run_delta_maps.py")
        
        # Create and store plots of the results of deltaMaps
        plot_dMaps_output(geofile = 'data/AVISO/AVISO_MSLA_1993-2020_prep_2_deg.nc',
                          fpath = outdir, 
                          output = 'all',
                          outpath = 'playground/plots/2_deg_gaus/'+version+"/",
                          show_seeds = 'homogeneity') 
    
    #%% NMI calculation to find best value for k
        
    # Path to the domain maps
    path = 'playground/output/'
    nmi_matrix = calc_nmi_matrix(path = path, 
                                 res = 2, 
                                 k = k_range,
                                 path_end = '_gaus/domain_identification/domain_maps.npy')
        
    plot_nmi_matrix(nmi_matrix = nmi_matrix,
                    k = k_range,
                    fname = "playground/plots/2_deg_gaus/nmi_matrix_res_2_gaus.png")