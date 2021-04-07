#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 08:44:20 2021

@author: root
"""




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
    


#%%
if __name__ == "__main__":
    
    import os
    os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/data/AVISO/")
    
    #%% Download the AVISO data
    url = "ftp://ftp-access.aviso.altimetry.fr/climatology/global/delayed-time/monthly_mean/"
    cutdirs = 5
    username = "eike.schutt@nioz.nl"
    password = "cG3mLH"
    subdir = "raw/"
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
            outfile = "AVISO_MSLA_1993-2020_detrend.nc",
            res = 2,
            lonlatbox = [0, 360, -80, 80])













