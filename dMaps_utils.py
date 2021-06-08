

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
    if variable_name == 'time':
        from netCDF4 import num2date
        import numpy as np
        time_var = nc_fid.variables[variable_name]
        field = num2date(time_var[:],time_var.units, 
                         only_use_cftime_datetimes=False,
                         only_use_python_datetimes=True).filled(np.nan).reshape(len(time_var),1)
    else:
        field = nc_fid.variables[variable_name][:]     
    return field 

def plot_map(lat, lon, data, seeds, title, cmap = 'viridis', alpha=1.,
             show_colorbar=True, show_grid=False, outpath=None, 
             labels=False, extent=None, pos_dict=None, draw_box=False,
             ax = None):
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
    cmap : string, optional
        Colormap of the plot. The default is 'viridis'. 
    alpha : float, optional
        Alpha (opacity) of the domains.
    show_colorbar :  boolean, optional
        Whether to draw the colorbar or not. Default is True.
    show_grid :  boolean, optional
        Whether to draw gridlines and labels or not. Default is False.        
    outpath : string, optional
        Path where the plot will be saved. The default is None.
    labels : boolean, optional
        If true, labels will be drawn at each domain (mean of the position of 
        all non-nan values in data). The default is False.        
    extent : list, optional
        The extent of the map. The list must have the following structure: 
        [lon_min, lon_max, lat_min, lat_max]. If None is given, the entire 
        earth will be shown. The default is None.             
    pos_dict : dict, optinal
        Points on the map that will be highlighted with a cross (+) and a label
        indicating the locations latitude and longitude, if draw_box=False. 
        Must be in format {"lat": pos_lat, "lon": pos_lon} where pos_lat and 
        pos_lon are lists of coordinates in WGS84.
    draw_box : boolean, optional
        If True, the positions in pos_dict will be interpreted as outer points
        of an area that will be filled with a color. Default is False.
        

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    from  cartopy import crs as ccrs, feature as cfeature
    import os
    import numpy as np
    import cmocean
    
    if extent is None:
        crs = ccrs.PlateCarree(central_longitude=180)
    else:
        crs = ccrs.PlateCarree()
        lon_min, lon_max,  lat_min, lat_max = extent
        # convert longitude coordinates to 0-360 scale
        if lon_min < 0: lon_min = lon_min + 360
        if lon_max < 0: lon_max = lon_max + 360
    
    if ax is None:
        fig, ax =  plt.subplots(1,1,figsize=(12,8), dpi=300,
                                subplot_kw=dict(projection=crs))   
    else:
        ax=ax
    
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(1,1,1, 
    #                      projection = ccrs.PlateCarree(central_longitude=180))
    
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # ax.coastlines('110m', alpha=0.1)
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m"), 
                   facecolor='xkcd:grey', zorder=0)
    
    # Alternative to contourf: plot the "real" raster using pcolormesh
    # filled_c = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
    #                          cmap='gist_ncar')

    filled_c = ax.contourf(lon, lat, data, transform = ccrs.PlateCarree(), 
                           levels = 100, cmap = cmap, alpha=alpha)#, vmin = 0, vmax=100)
    
    if type(seeds) == np.ndarray:
        # Get index of all seed locations and get their lat/lon coordinates
        y, x = np.where(seeds==1)
        y_lat = lat[y]
        x_lon = lon[x]
        # Plot each seed location
        for i in range(len(x_lon)):
            ax.plot(x_lon[i], y_lat[i], marker='.', c='r', markersize=2, 
                    transform=ccrs.PlateCarree())
            
    if labels == True:
        for i in np.unique(data[~np.isnan(data)]):
            y, x = np.where(data==i)
            
            # if domain crosses LON=0, assign the label to one 1° or -1°
            # (otherwise it will be somehwere on the other side of the earth)
            if 0 in x and 179 in x:
                x = int(np.round(np.mean(x)))
                if x < 90:
                    x = 0
                else:
                    x = 179
            else:
                x = int(np.round(np.mean(x)))
            y = int(np.round(np.mean(y)))
            
            if extent is not None:
                # plot label only if it's inside the extent of the plot
                if lon[x] > lon_min and lon[x] < lon_max and \
                   lat[y] > lat_min and lat[y] < lat_max:
                    ax.text(lon[x],lat[y], int(i-1), c='k', transform=ccrs.PlateCarree())

    # plot positions and their labels
    if pos_dict and draw_box==False:
        for i in range(len(pos_dict['lat'])):
            ax.plot(pos_dict['lon'][i], pos_dict['lat'][i], marker='+', 
                    color='k', markersize=12, markeredgewidth = 2,
                    transform=ccrs.Geodetic())
            
            ax.text(pos_dict['lon'][i], pos_dict['lat'][i]+3, 
                    "lat = {lat}\nlon = {lon}".format(lat=pos_dict['lat'][i],
                                                      lon=pos_dict['lon'][i]), 
                    verticalalignment='bottom', horizontalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'),
                    color='k', transform=ccrs.Geodetic())
    
    # Plot box
    if pos_dict and draw_box==True:
        if type(pos_dict)==list:
            # cols = cmocean.cm.haline(len(pos_dict))
            for i in range(len(pos_dict)):
                temp = pos_dict[i]
                # ax.fill(temp["lon"], temp["lat"], 
                #         color=cmocean.cm.haline(i/len(pos_dict)*256), 
                #         transform=ccrs.Geodetic(), alpha=0.8)
                ax.plot(temp["lon"], temp["lat"], marker='o', 
                        transform=ccrs.Geodetic())
                if len(pos_dict)>1:
                    region_label = "Region {}".format(i)
                else:
                    region_label = "Region"
                ax.text((temp['lon'][0]+temp['lon'][2])/2, 
                        (temp['lat'][0]+temp['lat'][1])/2, 
                        region_label,
                        verticalalignment='bottom', horizontalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'),
                    color='k', transform=ccrs.Geodetic())
        else:
            ax.fill(pos_dict["lon"], pos_dict["lat"], 
                    color=cmocean.cm.haline(128), 
                    transform=ccrs.Geodetic(), alpha=0.8)           
            ax.plot(pos_dict["lon"], pos_dict["lat"], marker='o', 
                        transform=ccrs.Geodetic())
            
            
    if show_grid==True:
        g1 = ax.gridlines(draw_labels=True)
        g1.top_labels = False
        g1.right_labels = False

        
    if show_colorbar==True:
        fig.colorbar(filled_c, orientation='horizontal')
    ax.set_title(title)

    if outpath==None and ax is None:
        #return ax
        plt.show()
    elif outpath is not None:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        plt.savefig(outpath + title + '.png', bbox_inches = 'tight')
        plt.close()    
        
        

def plot_dMaps_output(geofile, 
                      fpath, 
                      output = 'domain', 
                      outpath=None, 
                      show_seeds=False,
                      extent = None):
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
    extent : list, optional
        The extent of the map. The list must have the following structure: 
        [lon_min, lon_max, lat_min, lat_max]. If None is given, the entire 
        earth will be shown. The default is None.               

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
                 cmap = 'viridis',
                 outpath = outpath,
                 extent = extent)     
        
    if output == 'all' or output == 'domain':
        if show_seeds=='homogeneity':
            seeds = None
        # Import domain maps
        d_maps = np.load(fpath + '/domain_identification/domain_maps.npy')
        # Create array containing the number of each domain
        domain_map = get_domain_map(d_maps)
                
        plot_map(lat = lat, 
                  lon = lon, 
                  data = domain_map,
                  seeds = seeds,
                  title = "Domain map",
                  cmap = 'prism',
                  outpath = outpath,
                  labels = True,
                  extent = extent)             
        
    if output == 'all' or output == 'domain strength':
        seeds = None
        # Import domain maps
        strength_map = np.load(fpath + '/network_inference/strength_map.npy')
        strength_map[strength_map==0] = np.nan
                
        plot_map(lat = lat, 
                 lon = lon, 
                 data = strength_map,
                 seeds = seeds,
                 title = "Strength map",
                 cmap = 'viridis',
                 outpath = outpath,
                 extent = extent)          
        
def get_domain_map(d_maps):
    """
    Helper function that returns an array with the grid values for the 
    corresponding domain.

    Parameters
    ----------
    d_maps : np.array
        Three dimensional umpy array from 
        .../domain_identification/domain_maps.npy.

    Returns
    -------
    domain_map : np.array
        Two dimensional numpy array with the domain number as grid cell values.
        If no domain is present at a grid cell, a np.nan will be inserted.

    """
    import numpy as np
    # Create array containing the number of each domain
    domain_map = np.zeros((d_maps.shape[1], d_maps.shape[2]))
    i = 1
    for d in range(len(d_maps)):
        domain_map[d_maps[d] == 1] = i
        i += 1
    domain_map[domain_map==0] = np.nan
    return domain_map
      
    
     
#%% plot network

def create_network(net_list, strength_list, graph_type):
    """
    Creates a networkx-network from dMaps output data. 

    Parameters
    ----------
    net_list : np.array
        Array from network_inference/network_list.npy.
    strength_list : np.array
        array from network_inference/strength_list.npy.
    graph_type : nx.Graph() or nx.DiGraph()
        Decides if the network has directed egdes (arrows on the edges in the
        plot). nx.DiGraph() creates a directed network, where the edges are 
        directed from the domain in the first column in net_list to the domain 
        in the second column.

    Returns
    -------
    G : networkx.classes.graph.Graph
        Network of the deltaMaps output.
    edges : tuple
        Tuples of all edges between domains.
    weights : tuple
        Weights of each edge defined in edges.
    nodes_strength : list
        List of the strength of each node (can be used to assign different 
        colors for the nodes).

    """
    import networkx as nx
    import pandas as pd
    import numpy as np
    # Build a dataframe with your connections
    # directly convert from dMaps numbering to numbers from my plot
    df = pd.DataFrame({'from': [np.where(strength_list[:,0]==i)[0][0] for i in net_list[:,0]], 
                       'to': [np.where(strength_list[:,0]==i)[0][0] for i in net_list[:,1]], 
                       'weight': net_list[:,5],
                       'lag': abs(net_list[:,4])})
    
    # Build the graph
    G=nx.from_pandas_edgelist(df, 'from', 'to',edge_attr=['weight', 'lag'], 
                              create_using=graph_type )
    
    # Create tuples for each edge and the corresponding weights
    try:
        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    except ValueError: # if graph is empty
        edges = weights = None
    
    # get the domain strengths for each domain in the correct order
    # nodes colours are associated with nodes in G.nodes (other ordering than in
    # strength_list)
    nodes_strength = [strength_list[i,1] for i in G.nodes()]
    return G, edges, weights, nodes_strength



def plot_network(fpath, geofile, out_fpath, extent = None):
    """
    Plots the network and the domain strengths infered by deltaMaps onto a 
    projected world map. It creates a networkx-network based on the deltaMaps
    data and uses cartopy to plot this data.

    Parameters
    ----------
    fpath : str
        Path to the folder where deltaMaps stored everything (i.e. has subdirs
        network_inference and domain_identification).
    geofile : str
        Filepath and filename of the nc-file that has been analysed by 
        deltaMaps (Required for determining the geo-coordinates of the nodes)
    out_fpath : str
        filepath and filename of the output plot.
    extent : list, optional
        The extent of the map. The list must have the following structure: 
        [lon_min, lon_max, lat_min, lat_max]. If None is given, the entire 
        earth will be shown. The default is None.               

    Returns
    -------
    None.

    """
    

    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from  cartopy import crs as ccrs, feature as cfeature
    import cmocean
    
    # create the network with networkX
    
    net_list = np.load(fpath + "network_inference/network_list.npy")
    strength_list = np.load(fpath + "network_inference/strength_list.npy")
    strength_map = np.load(fpath + "network_inference/strength_map.npy")
    strength_map[strength_map==0] = np.nan
    d_maps = np.load(fpath + '/domain_identification/domain_maps.npy')
    
    # geofile = "data/AVISO/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"
    lon = importNetcdf(geofile,'lon')
    lat = importNetcdf(geofile,'lat')
    
    
    # Domain calculations
    
    # create an array containing all domains with their respective number as 
    # cell value
    domain_map = get_domain_map(d_maps)
            
    # Calculate average coordinates for each domain (will be the coordinates 
    # for the nodes in the plot)
    ids = []
    coords_temp = []
    for i in np.unique(domain_map[~np.isnan(domain_map)]):
        y, x = np.where(domain_map==i)
        
        if 0 in x and 179 in x:
            x = int(np.round(np.mean(x)))
            if x < 90:
                x = 0
            else:
                x = 179
        else:
            x = int(np.round(np.mean(x)))
        y = int(np.round(np.mean(y)))
        
        x = lon[x]-180
        y = lat[y]

        ids.append(i)
        coords_temp.append((x,y))
    
    coords = {ids[i]-1: coords_temp[i] for i in range(len(ids))}
    
    
    # get all edges with a direction
    i_undir = []
    i_dir = []
    for i in range(len(net_list)):
        if 0 in range(int(net_list[i,2]), int(net_list[i,3])+1):
            i_undir.append(i)
        else:
            i_dir.append(i)
    
    # get all undirected edges
    edge_undir = net_list[i_undir,:]
    
    # get all directed edges
    edge_dir = net_list[i_dir, :]
    # if values are negative, edges goes from B to A -> flip values in columns 
    # 1+2
    idx = edge_dir[:,4]<0
    edge_dir[:,0][idx], edge_dir[:,1][idx] = edge_dir[:,1][idx], edge_dir[:,0][idx]
    
    
    
    # Plot the network on a map
    node_vmin = 0 # np.quantile(nodes_strength, 0.05)
    node_vmax = 10 # np.quantile(nodes_strength, 0.95)
    cmap_nodes = cmocean.cm.thermal
    
    edge_vmin = -0.8 # np.quantile(df.weight,0.05)# df.weight.min()
    edge_vmax = 0.8 # np.quantile(df.weight,0.95) # df.weight.max()
    cmap_edges = cmocean.cm.balance
    
    if extent is None:
        crs = ccrs.PlateCarree(central_longitude=180)
    else:
        crs = ccrs.PlateCarree()
    
    fig, ax =  plt.subplots(1,1,figsize=(12,8), dpi=300,
                            subplot_kw=dict(projection=crs))
    
    # ax.add_feature(cfeature.LAND, edgecolor='k')
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m"), 
                   facecolor='xkcd:grey', zorder=0)
    
    ax.contourf(lon, lat, strength_map, transform = ccrs.PlateCarree(), 
                levels=100, cmap = cmap_nodes, 
                vmin = node_vmin, vmax=node_vmax)
    
    
    G, edges, weights, nodes_strength = create_network(net_list = net_list, 
                                                strength_list = strength_list, 
                                                graph_type = nx.Graph())
        
    nx.draw_networkx_nodes(G, pos = coords, node_size=150, cmap = cmap_nodes,
                               #node_color = nodes_strength, 
                               node_color='w',
                               vmin = node_vmin, vmax = node_vmax)
    
    nx.draw_networkx_labels(G, pos=coords, font_size=10, font_color='k')
    
    edgy = [edge_undir, edge_dir]
    graph_types = [nx.Graph(), nx.DiGraph()]
    
    for i in range(len(edgy)):
        edges_array = edgy[i]
        graph_type = graph_types[i]
        G, edges, weights, nodes_strength = create_network(net_list = edges_array, 
                                                strength_list = strength_list, 
                                                graph_type=graph_type)
        
    
        
        if i==1:
            nx.draw_networkx_edge_labels(G,pos=coords, 
                                     edge_labels=nx.get_edge_attributes(G,'lag'),
                                     label_pos=0.5, font_size=6)
    
        nx.draw_networkx_edges(G, pos=coords, edgelist = edges, 
                               edge_color=weights,
                               edge_cmap = cmap_edges, 
                               edge_vmin = edge_vmin, edge_vmax = edge_vmax,
                               width = 2,
                               connectionstyle="arc3,rad=0.3")
    
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Define colorbar for the nodes
    sm = plt.cm.ScalarMappable(cmap=cmap_nodes, 
                               norm=plt.Normalize(vmin = node_vmin,
                                                  vmax = node_vmax))
    
    cbaxes = fig.add_axes([0.125, 0.1, 0.35, 0.04])
    sm._A = []
    cbar_nodes = fig.colorbar(sm, cax=cbaxes, orientation='horizontal', 
                              shrink=0.3, pad=0.05)
    cbar_nodes.set_label("Domain strength")
    # cbar_nodes.ax.set_title("Node & Domain colour")
    
    # Define colorbar for the edges
    sm = plt.cm.ScalarMappable(cmap=cmap_edges, 
                               norm=plt.Normalize(vmin = edge_vmin, 
                                                  vmax = edge_vmax))
    
    cbaxes = fig.add_axes([0.55, 0.1, 0.35, 0.04])
    sm._A = []
    cbar_edges = fig.colorbar(sm, cax=cbaxes, orientation='horizontal', 
                              shrink=0.3, pad=0.05)
    cbar_edges.set_label("Cross-Correlation (Edges)")
    # cbar_edges.ax.set_title("Edge colour")
    
    # plt.savefig("playground/plots/Network/res_2_k_8_gaus/dMaps_network.png")
    plt.savefig(out_fpath, bbox_inches = 'tight')


        

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
        plt.savefig(fname, bbox_inches = 'tight')
        
#%% Functions to return domain signals
def get_domain_signals(domains, sla, lat, signal_type = 'cumulative'):
    """
    Function that returns the cumulative or average domain signals for each 
    time step in sla. The signal of each grid cell is weighted according to
    the latitude of the grid cell.

    Parameters
    ----------
    domains : np.array
        Domain output array of deltaMaps (i.e. with an (x,y,z) dimensionality).
    sla : np.array
        Array with the signals (i.e. sea level anomalies here). This is an
        array with three dimensions (t,x,y) that was also used to identify the
        domains.
    lat : np.array
        Vector with latitudes of each row of grids.
    signal_type : string, optional
        Type of the signal to be returned. Can be 'cumulative' (i.e. the sum of
        the signal of all grid cells in a domain) or 'average' (i.e. the mean 
        signal of all grid cells). The default is 'cumulative'.

    Returns
    -------
    np.array
        Numpy array containing the domain signals for each domain (x-axis) for each timestep in sla (y-axis).

    """
    import numpy as np
    # transform latitudes to radians
    lat_rad = np.radians(lat)
    # Assigna a weight to each latitude phi
    lat_weights = np.cos(lat_rad).reshape(len(lat_rad),1)
    # Define the weighted domain
    weighted_domains = domains*lat_weights
    
    
    signals = []
    for i in range(len(weighted_domains)):
        if signal_type == 'cumulative':
            signals.append(cumulative_anomaly(sla, weighted_domains[i]))
        # in case you want the average anomaly
        elif signal_type == 'average':
            signals.append(average_anomaly(sla, domains[i]))#weighted_domains[i])) 
    return np.array(signals).T

def cumulative_anomaly(data,domain):
    # input:
    #       (a) data: spatiotemporal climate field (i.e., np.shape(data) = (time, lats, lon))
    #       (b) domain: weighted domain (i.e., np.shape(domain) = (lats, lon))
    # output:
    #       (a) domain's signal:
    #           X(t) = Sum(x_i(t)*cos(phi_i))
    #           where: x_i(t): timeseries at grid point i
    #                  phi_i: latitude of grid point i
    import numpy as np
    return np.nansum(data*domain,axis=(1,2))

# Function to compute the average domain's anomaly
def average_anomaly(data,domain):
    # input:
    #       (a) data: spatiotemporal climate field (i.e., np.shape(data) = (time, lats, lon))
    #       (b) domain: weighted domain (i.e., np.shape(domain) = (lats, lon))
    # output:
    #       (a) domain's signal:
    #           X(t) = (1/n)Sum(x_i(t)*cos(phi_i))
    #           where: x_i(t): timeseries at grid point i
    #                  phi_i: latitude of grid point i
    #                  n: number of time series in the domain
    import numpy as np
    # Number of grid cells inside the domain
    domain[domain==0] = np.nan
    mult = data*domain
    n = np.count_nonzero(~np.isnan(mult), axis=(1,2)) #len(domain[domain>0])
    return (1/n) * np.nansum(mult,axis=(1,2))

def plot_domain_signals(signals, domain_ids, time, var_names, 
                        filepath = None, filename = None):
    """
    Creates a line plot of the domain signals for specified regions.

    Parameters
    ----------
    signals : np.array
        Output array of dMaps.get_domain_signals (or a derivative...).
    domain_ids : list
        List of the index of the domains to be plotted. Can have any length.
    time : np.arary with datetime.datetime objects
        Time that will be plotted on the x-axis. Is the output of e.g.
        dMaps.importNetcdf(geofile,'time')
    var_names : list
        List of the names of all domains. Will be used for the legend label.
    filepath : str, optional
        Filepath location. If None is given, the plot will just be shown and 
        not be saved. The default is None.
    filename : str, optional
        The filename of the output image. E.g. "Domain_signals.png". The 
        default is None.
    
    Returns
    -------
    None.

    Example usage:
    --------------
    signals = dMaps.get_domain_signals(d_maps, sla, lat, signal_type='average')
    domain_list = [21,24,25,45,37,13,7,35, 41, 19]
    # keep only some regions
    signals = signals[:,domain_list]
    var_names = ["ENSO", "SE Asia",  "W Indian", "S Atl", "Trop Atl", "E USA", "E Can", "S HS", "N Atl", "NE Atl"]
    domain_ids = [0,1] # Plot only these rergions
    filepath = "playground/plots/2_deg_gaus_a_1_tau_24/res_2_k_12_gaus_a_1_tau_24/network/"
    filename = "domain_signals_ENSO_.png"
    plot_domain_signals(signals = signals, 
                        domain_ids = domain_ids, 
                        time = time, 
                        var_names = var_names, 
                        filepath = None,
                        filename = filename)
    """
    import matplotlib.pyplot as plt
    import os
    
    fig, ax = plt.subplots(figsize=(8,4), dpi=300)
    for i in domain_ids:
        ax.plot(time, signals[:,i],linewidth = 2, label=var_names[i])
    ax.set_ylabel('average domain SLA [cm]',fontsize = 15)
    ax.legend()
    ax.grid()
    if filepath==None:
        plt.show()
    else:
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        plt.savefig(filepath+filename, bbox_inches = 'tight')
        plt.close()  


#%% Download and prepare dataset
if __name__ == "__main__":
    
    import os
    os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/data/AVISO/")
    
    #%% Download the AVISO data
    url = "ftp://ftp-access.aviso.altimetry.fr/climatology/global/delayed-time/monthly_mean/"
    cutdirs = 5
    username = "youremail@adress.com"
    password = "yourpassword"
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


    # #%% Run delta maps for a range of k values and plot the results
    # os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
    
    # # Define dataset that will be used by deltaMaps
    # infile = "data/AVISO/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"
    
    # # Define range of k-values (neighborhood size)
    # k_range = range(1,21,1)
    
    # for k in k_range:
    #     # Define path and filename of output of deltaMaps
    #     version = "res_2_k_" + str(k) + "_gaus"
    #     outdir = "playground/output/" + version
        
    #     path_to_config = "playground/"
    #     config_name = "config_sla_aviso_gaus"
    #     # Create the configuration file
    #     create_config_file(path_to_config = path_to_config,
    #                        config_name = config_name,
    #                        path_to_file = infile,
    #                        output_dir = outdir,
    #                        var_name = "sla",
    #                        lat_name = 'lat',lon_name = 'lon',
    #                        delta_samples = 10000,
    #                        alpha = 0.01,
    #                        k = k,
    #                        tau_max = 12,
    #                        q = 0.05)
        
    #     # Run deltaMaps with the configuration file that was just created
    #     run_dMaps(config_file_name = path_to_config+config_name+".json",
    #                     dmaps_file_name = "py-dMaps/run_delta_maps.py")
        
    #     # Create and store plots of the results of deltaMaps
    #     plot_dMaps_output(geofile = 'data/AVISO/AVISO_MSLA_1993-2020_prep_2_deg.nc',
    #                       fpath = outdir, 
    #                       output = 'all',
    #                       outpath = 'playground/plots/2_deg_gaus/'+version+"/",
    #                       show_seeds = 'homogeneity') 
    
    # #%% NMI calculation to find best value for k
        
    # # Path to the domain maps
    # path = 'playground/output/'
    # nmi_matrix = calc_nmi_matrix(path = path, 
    #                              res = 2, 
    #                              k = k_range,
    #                              path_end = '_gaus/domain_identification/domain_maps.npy')
        
    # plot_nmi_matrix(nmi_matrix = nmi_matrix,
    #                 k = k_range,
    #                 fname = "playground/plots/2_deg_gaus/nmi_matrix_res_2_gaus.png")