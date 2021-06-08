import numpy as np
import os
try:
    os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
except FileNotFoundError:
    os.chdir("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")    
from dMaps_SLV import dMaps_utils as dMaps
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import scipy.stats  as stats

def wind_to_stress(vnorth, veast):
    """
    Calculate the north-and wastward component of wind stress (Large and Pond-
    Method) using the windstress-function from the air-sea package 
    (https://github.com/pyoceans/python-airsea) under the hood. Uses the 
    default values for air density (rho_air) of 1.22 and ameasurement height 
    of 10 m.

    Parameters
    ----------
    vnorth : np.array or similar
        Array containing the northward component (v component) of wind in m/s.
    veast : np.array or similar
        Array containing the northward component (u component) of wind in m/s.

    Returns
    -------
    stress_north : np.array
        Northward component (v component) of wind stress in N/m-2.
    stress_east : TYPE
        Eastward component (u component) of wind stress in N/m-2.

    """
    import numpy as np
    from airsea.airsea import windstress as ws
    # convert E(u) and N(v) component of wind to wind speed and direction
    w_vel = np.sqrt(vnorth**2 + veast**2)
    w_dir = np.arctan2(veast, vnorth) * 180/np.pi
    w_dir = (360 + w_dir) % 360
    w_dir = w_dir+180
    
    # Calculate wind stress based on 
    w_stress = ws.stress(w_vel)
    
    # convert wind stress and direction of wind stress back into E (u) and N 
    # (v) component
    
    stress_north = w_stress*np.sin(np.deg2rad(270-w_dir))
    stress_east  = w_stress*np.cos(np.deg2rad(270-w_dir))
    
    return stress_north, stress_east

def prep_clim_index(fpath, skiprows=2, skipfooter=0, na_values=-99.99):
    """
    Prepares climate index in the same format as domain signals (timeseries
    between Jan 1993 and Feb 2020) as a Pandas series.

    Parameters
    ----------
    fpath : str
        Path and filename of txt file that will be imported. Must be structured 
        with rows=years and columns=months.
    skiprows : int, optional
        Numbers of rows to be skipped during import of the txt file. The 
        default is 2.
    skipfooter : int, optional
        Numbers of rows to be skipped during import at the bottom of the txt 
        file. The default is 0.
    na_values : float, optional
        No-Data value. The default is -99.99.

    Returns
    -------
    pd.Series
        Monthly timeseries of the climate index between Jan 1993 and Feb 2020.

    """
    
    months = ['year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 
              'Sep', 'Oct', 'Nov', 'Dec']
    
    df = pd.read_csv(fpath, skiprows=skiprows, skipfooter=skipfooter, 
                       delim_whitespace=True, header=None, na_values=na_values)
        
    df.columns = months
    
    df = df.melt(id_vars=["year"], value_vars = months[1:], var_name="month",
                   value_name = "value")
    
    df = df.dropna(axis=0)
    
    df["date"] = pd.to_datetime(df['year'].astype(str)  + df['month'], format='%Y%b')
    
    df = df[(df.date>=pd.to_datetime("1993-01-01")) & 
             (df.date<=pd.to_datetime("2020-02-01"))]
    
    df = df.sort_values(by="date").reset_index(drop=True)
    dt = df.date
    
    df = df.drop(["year", "month", "date"], axis=1)
    
    return df.squeeze(), dt.squeeze()



def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    from https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return stats.pearsonr(datax,shiftedy)
        # return datax.corr(shiftedy)
    else: 
        df = pd.concat([datax, datay.shift(lag)], axis=1)
        df = df.dropna()
        df.columns = ["datax", "datay"]
        return stats.pearsonr(df.datax,df.datay)
        # return datax.corr(datay.shift(lag))

def calc_plot_cross_corr(data1, data2, data1_label, data2_label, time,
                         data1_ylabel="clim. index [-]",
                         data2_ylabel="dim. sig. [cm]",
                         lag_range=range(12,-13,-1),
                         out_fname = None,
                         return_crosscorr = False,
                         data1_lead_label=None,
                         data2_lead_label=None,
                         leg_loc=3):
    """
    Calculates and plots cross correlation between two time series. A negative
    peak synchrony in the cross-corr plot means that data 2 preceeds data1.

    Parameters
    ----------
    data1 : pd.Series
        Timeseries of data (e.g. climate index).
    data2 : pd.Series
        Timeseries of data (e.g. domain signal).
    data1_label : sting
        Label for data1 in the plot.
    data2_label : string
        Label for data2 in the plot.
    time : np.array or similar
        Array containing the timevector of data1 and data2.
    data1_ylabel : string, optional
        y axis label for data1. The default is "clim. index [-]".
    data2_ylabel : TYPE, optional
        y axis label for data2. The default is "dim. sig. [cm]".
    lag_range : range object, optional
        Range for which the cross correlations will be calculated. The default 
        is range(12,-13,-1), i.e. +/- one year for data with monthly timesteps.
    out_fname : str, optional
        Filepath and name to store the plot. If None is given, the plot will 
        be shown and not saved. The default is None.
    return_crosscorr : boolean, optional
        If True, the list with cross-correlations will be returned.
    leg_loc : int, topional
        location of the legend in the cross-corr plot. Default is 3 (i.e.lower 
        left corner)

    Returns
    -------
    rs : list
        Cross-correlations between data1 and data2 for the different time lags.

    """


    
    r_p = [crosscorr(datax = data1, 
                    datay = data2,
                    lag = lag) for lag in lag_range]
    
    rs = [i[0] for i in r_p]
    
    # get absolute values in rs
    rs_abs = [abs(i) for i in rs]
    
    offset = np.floor(len(rs)/2)-np.argmax(rs_abs) # Lag with highest correlation is 
    # negative: datax preceeds datay! (correlation is maximised if datay is pulled
    # foreward by <offset> months)
    # Note: Negative and positive is flipped in figure!
    
    
    # Get a list of all significant time lags
    ps = np.array([i[1] for i in r_p])
    ps[ps>=0.05] = np.nan
    ps[~np.isnan(ps)] = 1
    sig_lags = np.array(rs*ps)
    
    
    fig, axs = plt.subplots(3,1, figsize=(8,6), dpi=150)
    axs[0].plot(time, data1, label=data1_label)
    axs[0].legend(loc=3)
    axs[0].set_ylabel(data1_ylabel)
    axs[0].set_xticklabels([])
    axs[0].grid()
    axs[0].set_xlim([time[0], time[len(time)-1]])
    
    axs[1].plot(time, data2, label=data2_label)
    axs[1].legend(loc=3)
    axs[1].set_ylabel(data2_ylabel)
    axs[1].set_xlabel("year")
    axs[1].grid()
    axs[1].set_xlim([time[0], time[len(time)-1]])
    
    axs[2].plot(rs)
    axs[2].scatter(np.arange(0,len(sig_lags)),sig_lags, label="p < 0.05")
    axs[2].axvline(np.floor(len(rs)/2),color='k',linestyle='--')
    axs[2].axvline(np.argmax(rs_abs),color='r',linestyle='--',
                   label='Peak synchrony (' + str(round(-offset)) + ' months)')
    # Peak synchrony is positive: Climate index preceeds domain signal by x 
    # months!
    axs[2].legend(loc=leg_loc)
    axs[2].set_ylabel("R")
    axs[2].set_xlabel("lag [months]")
    axs[2].set_xticks(np.arange(0,25,3))
    axs[2].set_xticklabels(np.arange(-12,13,3))
    axs[2].set_ylim([-1,1])
    axs[2].grid()
    
    if data1_lead_label:
        axs[2].text(17, -1.58, "{label} leading".format(label=data1_lead_label))
        axs[2].text(1, -1.58, "{label} leading".format(label=data2_lead_label))

    if out_fname is None:
        plt.show()
    else:
        plt.savefig(out_fname, bbox_inches = 'tight')
        
    if return_crosscorr is True:
        return rs
    
def calc_ax_cross_corr(data1, data2, time, ax,
                       lag_range=range(12,-13,-1),
                       out_fname = None,
                       return_crosscorr = False,
                       data1_lead_label=None,
                       data2_lead_label=None,
                       ):
    """
    Calculates and plots cross correlation between two time series. A negative
    peak synchrony in the cross-corr plot means that data 2 preceeds data1.

    Parameters
    ----------
    data1 : pd.Series
        Timeseries of data (e.g. climate index).
    data2 : pd.Series
        Timeseries of data (e.g. domain signal).
    time : np.array or similar
        Array containing the timevector of data1 and data2.
    lag_range : range object, optional
        Range for which the cross correlations will be calculated. The default 
        is range(12,-13,-1), i.e. +/- one year for data with monthly timesteps.
    out_fname : str, optional
        Filepath and name to store the plot. If None is given, the plot will 
        be shown and not saved. The default is None.
    return_crosscorr : boolean, optional
        If True, the list with cross-correlations will be returned.

    Returns
    -------
    rs : list
        Cross-correlations between data1 and data2 for the different time lags.

    """


    
    r_p = [crosscorr(datax = data1, 
                    datay = data2,
                    lag = lag) for lag in lag_range]
    
    rs = [i[0] for i in r_p]
    
    # get absolute values in rs
    rs_abs = [abs(i) for i in rs]
    
    offset = np.floor(len(rs)/2)-np.argmax(rs_abs) # Lag with highest correlation is 
    # negative: datax preceeds datay! (correlation is maximised if datay is pulled
    # foreward by <offset> months)
    # Note: Negative and positive is flipped in figure!
    
    
    # Get a list of all significant time lags
    ps = np.array([i[1] for i in r_p])
    ps[ps>=0.05] = np.nan
    ps[~np.isnan(ps)] = 1
    sig_lags = np.array(rs*ps)
    
    ax = ax

    
    ax.plot(rs)
    ax.scatter(np.arange(0,len(sig_lags)),sig_lags, label="p < 0.05")
    ax.axvline(np.floor(len(rs)/2),color='k',linestyle='--')
    ax.axvline(np.argmax(rs_abs),color='r',linestyle='--',
                   label='Peak synchrony (' + str(round(-offset)) + ' months)')
    # Peak synchrony is positive: Climate index preceeds domain signal by x 
    # months!
    ax.legend(loc=3)
    ax.set_ylabel("R")
    ax.set_xlabel("lag [months]")
    ax.set_xticks(np.arange(0,25,3))
    ax.set_xticklabels(np.arange(-12,13,3))
    ax.set_ylim([-1,1])
    ax.grid()
    
    if data1_lead_label:
        ax.text(19, -1.58, "{label} leading".format(label=data1_lead_label))
        ax.text(2, -1.58, "{label} leading".format(label=data2_lead_label))

    return ax  
    

def integrate_flow(ds, pos_lat, pos_lon):
    """
    Calculates the depth integrated flow through a 1 km transect and the
    weighted average flow direction from CMEMS GLOBAL_REANALYSIS_PHY_001_030
    current velocities for each time step at a given location. 

    Parameters
    ----------
    ds : xarray dataset
        Modified GLOBAL_REANALYSIS_PHY_001_030 product from CMEMS (contains a 
        time dimension.
    pos_lat : float
        Latitude of the position. Can range from -90° to 90°.
    pos_lon : float
        Longitude of the position. Can range from -180° to 180°. (CMEMS uses a
        range from 0-360°, this code automatically converts the longitude 
        coordinate if necessary.)

    Returns
    -------
    flow_depth_integ : np.array
        Numpy array of dimension (timesteps,) containing the depth integrated
        flow of the whole water column through a 1 km transect (unit Sv) for
        each time step in the CMEMS dataset.
    mean_fdir : np.array
        Numpy array of dimeions (timesteps,) containing the weighted average
        flow direction for each time step in the CMEMS dataset. (Unit: °. 
        Accounts for different current velocities and directionsacross the 
        water column.)
    flow_time : ndarray of datetime64
        Time vector for flow_depth_integ and mean_fdir.
    Usage
    -------
    import xarray as xr
    nc_fname = "data/CMEMS/CMEMS_phy_030_uo_vo_mlotst_1993_1994.nc"
    ds = xr.open_dataset(nc_fname) # Open dataset
    pos_lat = -36.5 # lat/lon coordinates of the position
    pos_lon = -88
    flow, flow_dir, flow_time = integ_flow(ds, pos_lat, pos_lon)
    """
    import numpy as np
    # import matplotlib.pyplot as plt
    # import cmocean
    
    
    # # nc_fname = "F:/CMEMS/test/subset/CMEMS_phy_030_uo_vo_mlotst_1993_1994.nc"
    # nc_fname = "data/CMEMS/CMEMS_phy_030_uo_vo_mlotst_1993_1994.nc"
    # ds = xr.open_dataset(nc_fname)
    
    # pos_lat = -36.5
    # pos_lon = -88
    
    # lon values are from 0 to 360, so convert values form -180 - 180 range
    if pos_lon<0:
        pos_lon = pos_lon+360
    
    # extract data array with data for the position
    vnorth = ds["vo"].sel(lon=pos_lon, lat=pos_lat, method="nearest")
    veast = ds["uo"].sel(lon=pos_lon, lat=pos_lat, method="nearest")
    
    # calculate current direction
    # https://support.nortekgroup.com/hc/en-us/articles/360012774640-How-to-calculate-current-speed-and-direction-from-ADCP-velocity-components
    
    curr_dir = 180+180/np.pi*(np.arctan2(-veast, -vnorth))
    
    # calculate speed
    curr_vel = np.sqrt(veast**2 + vnorth**2)
    
    # # plot
    # # get the correct lat/lon coordinates (not the ones we want but the ones of 
    # # grid cell we are using)
    # lat = np.array([ds.lat.sel(lat=pos_lat, method="nearest").values])[0]
    # lon = np.array([ds.lon.sel(lon=pos_lon, method="nearest").values])[0]
    # # convert longitudes from the western half of the hemisphere back on -180 - 0 
    # # range
    # if lon > 180:
    #     lon=lon-360
    
    # fig, axs = plt.subplots(2,1, figsize=(10,6), dpi=150)
    # curr_vel[:,0:46].plot(x = curr_vel.dims[0], y = curr_vel.dims[1], ax=axs[0], 
    #                       cmap=cmocean.cm.tempo, yincrease=False,
    #                       cbar_kwargs={'label': ''})
    # axs[0].set_xlabel('')
    # axs[0].set_xticklabels('')
    # axs[0].set_title('lat = {lat}, lon = {lon}'.format(lat= lat, lon= lon) +
    #                  '\ncurrent velocity [m s$^{-1}$]')
    
    
    # curr_dir[:,0:46].plot(x =curr_dir.dims[0], y=curr_dir.dims[1], ax=axs[1], 
    #                       cmap=cmocean.cm.phase, vmin=0, vmax=360, 
    #                       yincrease=False, 
    #                       cbar_kwargs={'label': ''})
    
    # axs[1].set_xlabel('time')
    # axs[1].set_title('current direction [°]')
    # plt.show()
    
    # calculate flow of whole water column
    # sverdrup flow: width (km) * depth (km) * current (m/s)
    
    # calculate the depth of each depth step
    # add a zero, becuase first depth is from surface to depth[0]
    depth = np.concatenate((np.array([0]), ds.depth.values))
    # difference between the consecutive elements of the array and conversion to km
    depth_diff = np.ediff1d(depth)/1000
    
    # calculate flow for each depth interval and each time step
    flow = depth_diff*curr_vel
    # integrate over depth of each timestep
    flow_depth_integ = np.nansum(flow, axis=1)
    
    
    # weightet mean flow direction
    # from here: https://math.stackexchange.com/questions/44621/calculate-average-wind-direction
    v_east = np.mean(flow * np.sin(curr_dir * np.pi/180), axis=1)
    v_north =  np.mean(flow * np.cos(curr_dir * np.pi/180), axis=1)
    mean_fdir = np.arctan2(v_east, v_north) * 180/np.pi
    mean_fdir = (360 + mean_fdir) % 360
    
    
    # # plot of depth integrated flow and weighted average flow direction
    # fig, axs = plt.subplots(2,1, figsize=(8,6), dpi=150)
    # axs[0].plot(ds.time, flow_depth_integ)
    # axs[0].set_title('lat = {lat}, lon = {lon}'.format(lat= lat, lon= lon) +
    #                  '\nDepth integrated flow through 1 km transect [Sv]')
    # # axs[0].set_title("Flow")
    # axs[0].set_ylabel("")
    # axs[0].set_xticklabels([])
    
    # mean_fdir.plot(ax=axs[1])
    # axs[1].set_title("Weighted average flow direction [°]")
    # axs[1].set_ylabel("")
    # axs[1].set_xlabel("Year-Month")
    # plt.show()
    return flow_depth_integ, mean_fdir.values, ds.time.values
    
# def overlap_timeseries(data1, ts1, data2, ts2):
#     """
#         Identifies the overlap betweent two time series and keeps only the 
#     overlapping timespan.

#     Parameters
#     ----------
#     data1, data2 : np.array
#         np.array with data.
#     ts1, ts2 : ndarray of datetime64
#         Array with the datetimes corresponding to data1/data2.

#     Returns
#     -------
#     data1 : np.array
#         np.array containing the overlapping timespan.
#     ts1 : ndarray of datetime64
#         Array with the datetimes corresponding to data1.
#     data2 : TYPE
#         np.array containing the overlapping timespan..
#     ts2 : TYPE
#         Array with the datetimes corresponding to data2.
#     """
    
#     start_time = max(ts1[0], ts2[0])
#     end_time = min(ts1[-1], ts2[-1])
#     data1 = data1[(ts1>=start_time) & (ts1<=end_time)]
#     data2 = data2[(ts2>=start_time) & (ts2<=end_time)]  
#     ts1 = ts1[(ts1>=start_time) & (ts1<=end_time)]
#     ts2 = ts2[(ts2>=start_time) & (ts2<=end_time)]
#     return data1, ts1, data2, ts2


def low_pass_weights(window, cutoff):
    """Calculate weights for a c.

    Args:

    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.

    From here:
    https://scitools.org.uk/iris/docs/v1.2/examples/graphics/SOI_filtering.html
    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]



def low_pass_filter(data, window, cutoff):
    """
    Applies a low pass Lanczos filter to a timeseries (pandas Series)

    Parameters
    ----------
    data : pd.Series
        Pandas series with values of the timeseries.
    window : int
        The length of the filter window. Must be an odd number.
    cutoff : float
        The cutoff frequency in inverse time steps.

    Returns
    -------
    filtered_data : pd.Series
        Low pass Lanczos filtered series.
        
    Usage
    -------
    from dMaps_SLV.network import network_analysis_utils as nau
    import matplotlib.pyplot as plt
    
    # Import data
    fpath = "data/climate_indices/PNA.txt"
    pna, pna_time = nau.prep_clim_index(fpath, skipfooter=0)
    
    # Apply filter
    window = 7
    cutoff = 1./6.
    pna_filtered = low_pass_filter(pna, window, cutoff)
    
    # Plot results
    plt.plot(pna_time, pna, label="PNA")
    plt.plot(pna_time, pna_filtered, label="low pass filtered")
    plt.legend()
    plt.show()
    """
    
    weights = low_pass_weights(window, cutoff)
    sum_weights = np.sum(weights)
    
    filtered_data = (data
        .rolling(window=window, center=True)
        .apply(lambda x: np.sum(weights*x) / sum_weights, raw=False)
    )
    return filtered_data


def avg_flow_depth_integ(ds, extent, timestep, ds_mean=None):
        
    # extent = [-170, -10, -60, 10]
    
    if extent[0]<0:
        extent[0] = extent[0]+360
    if extent[1]<0:
        extent[1] = extent[1]+360
    
    lat = ds.coords["lat"]
    lon = ds.coords["lon"]
    # time = ds.coords["time"]
    depth = ds.coords["depth"]
    
    lat_mask = lat[(lat>extent[2]) & (lat < extent[3])]
    lon_mask = lon[(lon>extent[0]) & (lon < extent[1])]
    
    vnorth = ds["vo"].loc[dict(lon=lon_mask, lat=lat_mask, 
                               time=ds.time[timestep])]
    veast = ds["uo"].loc[dict(lon=lon_mask, lat=lat_mask, 
                              time=ds.time[timestep])]
    
    # Calculate flow anomaly for each depth if requested
    if ds_mean:
        vnorth_mean = ds_mean["vo"].loc[dict(lon=lon_mask, lat=lat_mask)]
        veast_mean = ds_mean["uo"].loc[dict(lon=lon_mask, lat=lat_mask)]
        
        vnorth = vnorth - vnorth_mean
        veast = veast - veast_mean
    
    
    curr_dir = 180+180/np.pi*(np.arctan2(-veast, -vnorth))
    curr_vel = np.sqrt(veast**2 + vnorth**2)
    
    # calculate flow of whole water column
    # sverdrup flow: width (km) * depth (km) * current (m/s)
        
    # calculate the depth of each depth step
    # add a zero, becuase first depth is from surface to depth[0]
    depth = np.concatenate((np.array([0]), ds.depth.values))
    # difference between the consecutive elements of the array and conversion to km
    depth_diff = np.ediff1d(depth)/1000
    # bring it into the same dimesion as our data grid
    depth_diff = np.expand_dims(depth_diff, axis=[1,2])
    depth_diff = np.repeat(depth_diff, len(vnorth["lat"]), axis=1)
    depth_diff = np.repeat(depth_diff, len(vnorth["lon"]), axis=2)
    
    # calculate flow for each depth interval and each time step
    flow = depth_diff*curr_vel
    # integrate over depth of each timestep
    # flow_depth_integ = np.nansum(flow, axis=0)
    
    
    # weightet mean flow direction
    # from here: https://math.stackexchange.com/questions/44621/calculate-average-wind-direction
    v_east = np.mean(flow * np.sin(curr_dir * np.pi/180), axis=0)
    v_north =  np.mean(flow * np.cos(curr_dir * np.pi/180), axis=0)
    # mean_fdir = np.arctan2(v_east, v_north) * 180/np.pi
    # mean_fdir = (360 + mean_fdir) % 360
    # return v_east, v_north, mean_fdir, lon_mask, lat_mask
    return v_east, v_north# , flow_depth_integ, lon_mask, lat_mask

def avg_wind(ds, extent, timestep):
        
    # extent = [-170, -10, -60, 10]
    
    if extent[0]<0:
        extent[0] = extent[0]+360
    if extent[1]<0:
        extent[1] = extent[1]+360
    
    lat = ds.coords["lat"]
    lon = ds.coords["lon"]
    
    lat_mask = lat[(lat>extent[2]) & (lat < extent[3])]
    lon_mask = lon[(lon>extent[0]) & (lon < extent[1])]
    
    vnorth = ds["v10"].loc[dict(lon=lon_mask, lat=lat_mask, 
                               time=ds.time[timestep], expver=1)]
    veast = ds["u10"].loc[dict(lon=lon_mask, lat=lat_mask, 
                              time=ds.time[timestep], expver=1)]
    
    return veast, vnorth

def mean_areal_flow(ds, extents, param, ds_mean=None,
                    window=7, cutoff = 1./6.):
    """
    Calculates average north- and eastward flow [Sv, across 1 km transect] in 
    a specified area.

    Parameters
    ----------
    ds : xarray.core.dataset.Dataset
        CMEMS dataset containing north- and eastward current velocities.
    extents : list of list of lists
        Lists containint the boundaries of the area for which the average flow
        will be calculated. Format is [lonmin, lonmax, latmin, latmax] (WGS84).
        If the analysis shall be done for more than one area, use a list of 
        several lists of this format.
    param : string
        For which parameter the cross correlation will be. Can be eighter
        "flow" or "wind".
    ds_mean : xarray.core.dataset.Dataset, optional.
        Dataset containing the mean flow data. If a dataset is specified,
        the flow anomalies from the mean will be calculated. Must be in 
        the same resolution as ds. If ds_mean=None, the values of ds will be
        returned directly. Default is None.
    window : int, optional.
        The length of the low pass filter window. Must be an odd number. 
        Default is 7. If window=0, no low pass filter will be applied.
    cutoff : float, optional.
        The cutoff frequency in inverse time steps for the low pass filter.
        Default is 1./6.

    Returns
    -------
    v_east, v_north: pd.Series of north and eastward flow.

    """

    for k, extent in enumerate(extents):
        # bring negative longitudes into CMEMS format
        for i in range(2):
            if extent[i]<0:
                extent[i]=extent[i]+360
        
        # Import nc file with CMEMS data (flow anomalies)
        v_east_mean = []
        v_north_mean = []
        
        for i in range(len(ds.time)):
            if param == "flow":
                v_east, v_north = avg_flow_depth_integ(ds, extent, i, ds_mean)
            elif param == "wind":
                v_east, v_north = avg_wind(ds, extent, i)
            else:
                raise ValueError('Wrong parameter selected. Only supports "flow" and "wind".')
            v_east_mean.append(np.nanmean(v_east.values))
            v_north_mean.append(np.nanmean(v_north.values))
            
        v_mean = [v_east_mean, v_north_mean]            
        
        # Apply low pass filter only if window > 0
        if window !=0:
            v_east = low_pass_filter(data = pd.Series(v_mean[0]), 
                                     window = window, 
                                     cutoff = cutoff)
            v_north = low_pass_filter(data = pd.Series(v_mean[1]), 
                                      window = window, 
                                      cutoff = cutoff)

        return v_east, v_north
    
def mean_areal_cross_corr(ds, domain_id, domain_signals, extents, param, 
                          out_dir):
    """
    Calculates cross-correaltion between average north- and eastward flow, wind
    speed or wind stress (anomaly) direction in an area and domain SLA signal 
    of a specific domain and plots them. Also, a map showing the locations of 
    the domains and all areas is produced.

    Parameters
    ----------
    ds : xarray.core.dataset.Dataset
        CMEMS dataset containing north- and eastward current velocities.
    domain_id : int
        ID of domain to compute the cross-correlation for.
    domain_signals : np.ndarray
        Array containig the average domain SLA signals for each time step.
        (Output of dMaps.get_domain_signals()-function)
    extents : list of list of lists
        Lists containint the boundaries of the area for which the average flow
        will be calculated. Format is [lonmin, lonmax, latmin, latmax] (WGS84).
        If the analysis shall be done for more than one area, use a list of 
        several lists of this format.
    param : string
        For which parameter the cross correlation will be. Can be eighter
        "flow", "wind" or "wind stress". If "wind stress" is chosen, wind will
        be converted to wind stress using Large and Pond (1981) equation im-
        plemented in the airsea-package .
        (https://github.com/pyoceans/python-airsea/)
    out_dir : str
        Directory where all plots will be stored.

    Returns
    -------
    None.

    """
    # Parameters for low pass filtering
    window = 7
    cutoff = 1./6.
            
    for k, extent in enumerate(extents):
        # bring negative longitudes into CMEMS format
        for i in range(2):
            if extent[i]<0:
                extent[i]=extent[i]+360
        
        # Import nc file with CMEMS data (flow anomalies)

        
        v_east_mean = []
        v_north_mean = []
        
        for i in range(len(ds.time)):
            if param == "flow":
                v_east, v_north = avg_flow_depth_integ(ds, extent, i)
            elif param == "wind":
                v_east, v_north = avg_wind(ds, extent, i)
            elif param == "wind stress":
                # Get the wind
                v_east, v_north = avg_wind(ds, extent, i)
                # Calculate the wind stress components using Large and Pond-
                # Method using the default value for air density (rho_air) of 
                # 1.22 and a measurement height of 10 m.
                v_north, v_east = wind_to_stress(v_north, v_east)
                # ws.stress returns a numpy array and no xarray dataarray.
                # therefore no ".values" ! (compare to a few lines below)
                v_east_mean.append(np.nanmean(v_east))
                v_north_mean.append(np.nanmean(v_north))
            else:
                raise ValueError("Parameter not supported. \n\
                                 Choose flow, wind or wind stress.")
            if param == "flow" or param == "wind":                                 
                v_east_mean.append(np.nanmean(v_east.values))
                v_north_mean.append(np.nanmean(v_north.values))
        
            
        v_mean = [v_east_mean, v_north_mean]    
        v_label = ["east", "north"]    
        
        #% calc cross correlations and plot timeseries
        data1 = pd.Series(domain_signals[:,domain_id])[:312]
        data1_label = "Domain {domain} SLA".format(domain=domain_id)
        
        for i in range(len(v_label)):
            # data2 = pd.Series(v_mean[i])
            data2 = low_pass_filter(data = pd.Series(v_mean[i]), 
                                        window = window, 
                                        cutoff = cutoff)
            data2_label = "Region {nr} {direction}ward {param} anomalies ({window} months low pass filtered".format(nr=k, 
                                                                              direction=v_label[i],
                                                                              param=param,
                                                                              window=window)
            
            out_fname = "region_{region}_{direction}.png".format(region=k,
                                                                 direction=v_label[i])
            
            if param == "wind":
                label = "[m s$^{-1}$]"
            elif param == "wind stress":
                label = "[N m$^{-2}$]"
            elif param == "flow":
                label = "[Sv]"
            
            calc_plot_cross_corr(data1, data2[:312], data1_label, data2_label,
                                 data1_ylabel="[m]",
                                 data2_ylabel=label,
                                 time = ds.time.values[:312],
                                 lag_range=range(12,-13,-1),
                                 out_fname = out_dir+out_fname)
    
    # Now plot the map
    # Min/Max Extent to Lat/Lon-Dict
    pos_dict = []
    for extent in extents:
        pos_lat = [extent[3], extent[2], extent[2], extent[3], extent[3]]
        pos_lon = [extent[0], extent[0], extent[1], extent[1], extent[0]]
    
        pos_dict_temp = {"lat": pos_lat,
                         "lon": pos_lon}
        pos_dict.append(pos_dict_temp)
    
        
        
    # Plot positions into the domain map
    geofile = 'dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc'
    lon = dMaps.importNetcdf(geofile,'lon')
    lat = dMaps.importNetcdf(geofile,'lat')
        
    dmaps_path = 'dMaps_SLV/results/dMaps/res_2_k_11_gaus/'
    # Import domain maps
    d_maps = np.load(dmaps_path + 'domain_identification/domain_maps.npy')
    # Create array containing the number of each domain
    domain_map = dMaps.get_domain_map(d_maps)
    
    
                
    dMaps.plot_map(lat = lat, 
                   lon = lon, 
                   data = domain_map,
                   seeds = None,
                   title = "Domain map",
                   cmap = 'prism',
                   alpha = 0.3,
                   show_colorbar=False,
                   show_grid=True,
                   outpath = out_dir, # 'playground/plots/Network/flow/dom_25_pos',
                   labels = True,
                   extent = [-140, -60, -60, 10],#[-130, -60, -60, 10], # [-120, -60, -60, 10],
                   pos_dict = pos_dict,
                   draw_box = True)  
    

def create_gif(indir, outfname, fps, extension = ".png"):
    """
    Creates a gif with all files in "indir" with a specified fps.

    Parameters
    ----------
    infolder : str
        Path to directory with all pngs.
    outfname : str
        Path and filename under which the gif will be stored.
    fps : int/float
        Frames per second of the gif.
    extentsion :  string
        All files of this extentsion will be put into the gif.

    Returns
    -------
    None.

    """
    from PIL import Image
    import os
    
    
    # Create the frames
    frames = []
    #fpath = "playground/plots/Network/flow/Animation/sla_wind_pressure/"
    imgs = [f for f in os.listdir(indir) if f.endswith(extension)]
    # imgs = os.listdir(fpath)
    for i in imgs:
        new_frame = Image.open(indir+i)
        frames.append(new_frame)
    
    
    # Convert fps to duration of each image in ms
    duration = 1/fps*1000#len(frames)/fps
    
    # Save into a GIF file that loops forever
    frames[0].save(outfname, format='GIF',
                   append_images=frames[1:],
                   optimize=False, # show each frame and don't replace similar ones
                   save_all=True,
                   duration=duration, # time in ms a frame will be shown
                   loop=0) # change to 1 to play it one time and than stop


#%%
if __name__ == "__main__":
    # get ENSO index
    
    fpath = "data/climate_indices/nino_34_anomaly.txt"
    enso34, enso34_time = prep_clim_index(fpath)
    
    fpath = "data/climate_indices/meiv2.data"
    mei2, mei2_time = prep_clim_index(fpath)    
    
    fpath = "data/climate_indices/PNA.txt"
    pna, pna_time = prep_clim_index(fpath, skipfooter=0)
    
    
    
    
    
    #%% get domain signal
    dmaps_outpath = "dMaps_SLV/results/dMaps/res_2_k_11_gaus/"
    # Import domain map
    d_maps = np.load(dmaps_outpath + 'domain_identification/domain_maps.npy')
        
    ncfile = "dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"
    # Import SLA data and lat/lon data
    sla = dMaps.importNetcdf(ncfile, "sla")
    lat = dMaps.importNetcdf(ncfile, "lat")
        
        
    # Produce domain signals for each domain in d_maps and time step in sla
    signals = dMaps.get_domain_signals(domains = d_maps,
                                       sla = sla, 
                                       lat = lat, 
                                       signal_type = "average")
    
    #%% calc cross correlations and plot timeseries
    
    

    


    domain_id = 25
    data1 = pd.Series(enso34) #pd.Series(signals[:,49])
    data1_label = "Niño3.4 index"
    data2 = pd.Series(signals[:,domain_id])
    data2_label = "Domain {domain} mean SLA".format(domain=domain_id)
    
    out_fname = "playground/plots/Network/climate_indicies/cross_corr_Nino3.4_{domain}.png".format(domain=domain_id)
    calc_plot_cross_corr(data1, data2, data1_label, data2_label,
                         data1_ylabel="SSTA [°C]",
                         data2_ylabel="SLA [cm]",
                         data1_lead_label="SSTA",
                         data2_lead_label="SLA",
                         time = mei2_time,
                         lag_range=range(12,-13,-1),
                         out_fname = out_fname)#out_fname)


    #%% cross corr of current velocity
    
    # Define positions of analysis
    pos_lat = [-43, -27, -33, -37, -42] # 49: -43; 25: -27
    pos_lon = [-76, -106, -95, -90, -82] # 49: 104/-76; 25: 74/-106
    
    # Plot positions into the domain map
    geofile = 'dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc'
    lon = dMaps.importNetcdf(geofile,'lon')
    lat = dMaps.importNetcdf(geofile,'lat')
    
    dmaps_path = 'dMaps_SLV/results/dMaps/res_2_k_11_gaus/'
    # Import domain maps
    d_maps = np.load(dmaps_path + 'domain_identification/domain_maps.npy')
    # Create array containing the number of each domain
    domain_map = dMaps.get_domain_map(d_maps)
    
    pos_dict = {"lat": pos_lat,
                "lon": pos_lon}
                
    dMaps.plot_map(lat = lat, 
             lon = lon, 
             data = domain_map,
             seeds = None,
             title = "Domain map",
             cmap = 'prism',
             outpath = 'playground/plots/Network/flow/dom_25_pos',
             labels = True,
             extent = [-170, -10, -60, 10],
             pos_dict = pos_dict)    
    
    
    
    # Import nc file with CMEMS data (current velocities)
    nc_fname = "data/CMEMS/CMEMS_phy_030_uo_vo_mlotst_1993_2018_deseasoned.nc"
    ds = xr.open_dataset(nc_fname)
    
    
    for i in [0]:#range(len(pos_dict['lat'])):
        lat = pos_dict['lat'][i]
        lon = pos_dict['lon'][i]
            
        # Plot vertically integrated flow and flow direction
        flow, flow_dir, flow_time = integrate_flow(ds, lat, lon)
    
        # take overlapping subset of both the climate index and the CMEMS data
        # data1, ts1, data2, ts2 = overlap_timeseries(flow, flow_time, 
        #                                             np.array(mei2), np.array(mei2_time))
        
        # keep only signal of domain 25 of the first 312 tiemsteps (CMEMS data is 
        # available only until end of 2018)
        ts1 = flow_time
        data1 = flow_dir
        ts2 = mei2_time[0:312]
        data2 = signals[0:312,25] # mei2[0:24]
        
        # data1 = mei2
        data1_label = "flow direction at lat = {lat}, lon = {lon}".format(lat = lat, 
                                                                lon = lon)
        # data2 = pd.Series(flow)
        data2_label = "Domain signal 25"
        
        out_fname = 'dom_25_pos_flowdir_lat_{lat}_lon_{lon}.png'.format(
            lat=lat, lon=lon)
    
        calc_plot_cross_corr(pd.Series(data1), pd.Series(data2), 
                             data1_label, data2_label,
                             time = ts1,
                             data1_ylabel="[Sv]",
                             data2_ylabel="[cm]",
                             lag_range=range(12,-13,-1),
                             out_fname = None)#'playground/plots/Network/flow/' +
                                         #out_fname)