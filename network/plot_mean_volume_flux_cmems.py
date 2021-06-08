




import os
try: 
    os.chdir("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")
    # sys.path.append("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")
except FileNotFoundError:
    os.chdir("G:/Eigene Dateien/Studium/10. Semester/NIOZ/")
    # sys.path.append("G:/Eigene Dateien/Studium/10. Semester/NIOZ/")
from dMaps_SLV import dMaps_utils as dMaps
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from  cartopy import crs as ccrs, feature as cfeature
import pandas as pd
import matplotlib as mpl
import cmocean.cm as  cm





def arrow_data(ds, extent, timestep):
        
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
    flow_depth_integ = np.nansum(flow, axis=0)
    
    
    # weightet mean flow direction
    # from here: https://math.stackexchange.com/questions/44621/calculate-average-wind-direction
    v_east = np.mean(flow * np.sin(curr_dir * np.pi/180), axis=0)
    v_north =  np.mean(flow * np.cos(curr_dir * np.pi/180), axis=0)
    mean_fdir = np.arctan2(v_east, v_north) * 180/np.pi
    mean_fdir = (360 + mean_fdir) % 360
    # return v_east, v_north, mean_fdir, lon_mask, lat_mask
    return v_east, v_north, flow_depth_integ, lon_mask, lat_mask

def sla_data(ds, extent, timestep):
        
    # extent = [-170, -10, -60, 10]
    
    if extent[0]<0:
        extent[0] = extent[0]+360
    if extent[1]<0:
        extent[1] = extent[1]+360
    
    lat = ds.coords["lat"]
    lon = ds.coords["lon"]
    
    lat_mask = lat[(lat>extent[2]) & (lat < extent[3])]
    lon_mask = lon[(lon>extent[0]) & (lon < extent[1])]
    
    sla = ds["sla"].loc[dict(lon=lon_mask, lat=lat_mask, 
                               time=ds.time[timestep])]

    return sla, lon_mask, lat_mask

def arrow_plot_currents(extent, lon, lat, v_east, v_north, flow_depth_integ,
                        title, vmin=0, vmax=None, 
                        domain_data=None, domain_lat=None, domain_lon=None,
                        sla_data=None, sla_lat=None, sla_lon=None, 
                        sla_vmin=-0.1, sla_vmax=0.1,
                        timestep=0):
    
    
    
    
    
    fig = plt.figure(figsize=(12, 15))
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    #ax1.coastlines('50m')
    ax1.set_extent(extent, ccrs.PlateCarree())
    if vmax is None:
        vmax = np.nanmax(flow_depth_integ)
        

                          
    # plot sea level anomalies
    if sla_data is not None:
        ax1.pcolormesh(sla_lon, sla_lat, sla_data, 
                              transform = ccrs.PlateCarree(), 
                              cmap = cm.balance, vmin = domain_vmin, 
                              vmax = domain_vmax)
    
    
    # plot contours of domains
    if domain_data is not None:
        for i in range(domain_data.shape[0]):
            ax1.contour(domain_lon, domain_lat, domain_data[i,:,:], 
                          levels = 0, transform = ccrs.PlateCarree(), 
                          colors=['black'])#, vmin = domain_vmin, 
                          #vmax = domain_vmax)    
                     
        
    # cbar_dat = ax1.streamplot(lon.values, lat.values, v_east.values, v_north.values, color=flow_depth_integ, 
    #                       transform=ccrs.PlateCarree(), cmap='viridis', 
    #                       norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), density=1)
    cbar_dat = ax1.quiver(lon.values, lat.values, v_east.values, v_north.values, flow_depth_integ, 
                          transform=ccrs.PlateCarree(), cmap=cm.thermal, 
                          norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))    
    
    ax1.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m"), 
                       facecolor='xkcd:grey', zorder=0)
    
    
    # plt.colorbar(cbar_dat, ax=ax1, orientation='horizontal')
    
    # sm = plt.cm.ScalarMappable(cmap=cm.balance, 
    #                            norm=plt.Normalize(vmin = domain_vmin,
    #                                               vmax = domain_vmax))
    
    # cbaxes = fig.add_axes([0.125, 0.15, 0.35, 0.04])
    # sm._A = []
    # fig.colorbar(sm, cax=cbaxes, orientation='horizontal', 
    #              label='Background color: \nSea level anomaly [cm]', 
    #              shrink=0.3, pad=0.05)    
    
    
    cbaxes2 = fig.add_axes([0.125, 0.15, 0.775, 0.03])   
    fig.colorbar(cbar_dat, cax=cbaxes2, orientation='horizontal',
                  label='Arrow color: \ndepth integrated volume flux [Sv]', 
                  shrink=0.3, pad=0.05)
    
    
    #add_aligned_cmap(fig = fig, data = dom_cf)#cbar_dat, label="[Sv]")
    #add_aligned_cmap(fig = fig, data = cbar_dat, label="[Sv]")
    
    ax1.set_title(title)
    #plt.show()
    # out_fname = "playground/plots/Network/flow/Animation/fdir_arrows/flow_domains_49_25_{year}-{month}.png".format(
    #             year=title[-4:], month=title[-7:-5])
    out_fname = "playground/plots/Network/flow/mean_flow_domains_49_25.png"
    plt.savefig(out_fname, bbox_inches = 'tight')
    plt.show()
    
def add_aligned_cmap(fig, data, label = None, tick_pos = None, 
                     ticks_wanted = None):
    """
    Helper function that adds a vertical colorbar to the current plot with the
    colorbar of "data".

    Parameters
    ----------
    data : <matplotlib.collections.PathCollection at 0x1e4201d60d0>
        The handle of a plotting command.
    label : str, optional
        Label for the colorbar. The default is None.
    tick_pos : list of int or float, optional
        List where to put the labels. If None is given, default values will be
        used. The default is None.
    ticks_wanted : list of int or float, optional
        List with what to put on the tick positions. If None is given, default 
        values will be used. The default is None.

    Returns
    -------
    None.

    """
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="3%", pad=0.1, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    cbar = plt.colorbar(data, cax=ax_cb)  
    cbar.set_label(label)
    if tick_pos is not None:
        cbar.set_ticks(tick_pos)
        cbar.set_ticklabels(np.round(ticks_wanted, decimals=3))
        end_tick = len(cbar.ax.yaxis.get_ticklabels())-1

        for index, label in enumerate(cbar.ax.yaxis.get_ticklabels()):
            if index == 0 or index == 9 or index == 18 or index == 27 or \
                index == 36 or index == end_tick:
                label.set_visible(True)
            else:
                label.set_visible(False)       


#%%
# Import nc file with CMEMS data (current velocities)
nc_fname = "data/CMEMS/CMEMS_phy_030_uo_vo_mlotst_1993_2018_1_deg_mean.nc"
ds = xr.open_dataset(nc_fname)

ncfile = "dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"
ds_sla = xr.open_dataset(ncfile)


extent = [-120, -65, -55, -5]
time = ds.time.values

# Prepare domain signal data
domain_ids = [7, 18, 25, 43, 49, 59]
fpath = "dMaps_SLV/results/dMaps/res_2_k_11_gaus/domain_identification/domain_maps.npy"
domains = np.load(fpath)
domain_mask = domains[domain_ids, :, :]
    

sla = dMaps.importNetcdf(ncfile, 'sla')
lat_dom = dMaps.importNetcdf(ncfile, 'lat')
lon_dom = dMaps.importNetcdf(ncfile, 'lon')

domain_signals =  dMaps.get_domain_signals(domains = domains, 
                                    sla = sla, lat = lat_dom, 
                                    signal_type = 'average')
domain_signals = domain_signals[:,domain_ids]

domain_vmax = np.max([abs(np.min(domain_signals)), np.max(domain_signals)])
domain_vmin = -domain_vmax

#timestep = 1
#%%
for timestep in range(len(time)):


    year = pd.to_datetime(time[timestep]).year
    month = pd.to_datetime(time[timestep]).month
    title = "Mean Volume Flux 1993-2018"
    
    v_east, v_north, flow_depth_integ, lon_mask, lat_mask = arrow_data(ds, 
                                                                       extent, 
                                                                       timestep)
    
    domains_ti = domain_mask.copy()
    # for i in range(domain_signals.shape[1]):
    #     #domains_ti[i,:,:][domains_ti[i,:,:]==0] = np.nan
    #     domains_ti[i,:,:][domains_ti[i,:,:]==1] = domain_signals[timestep, i]
        
        
    sla, sla_lon, sla_lat = sla_data(ds_sla, extent, timestep)
    
    #%%
    
    # arrow_plot_currents(extent, lon=lon_mask, lat=lat_mask, 
    #                     v_east=v_east, v_north=v_north, 
    #                     flow_depth_integ=flow_depth_integ,
    #                     title=title, vmax=360, 
    #                     domain_data=sla, domain_lat=lat_dom, 
    #                     domain_lon=lon_dom, domain_vmin=domain_vmin,
    #                     domain_vmax=domain_vmax,
    #                     timestep=timestep)
    arrow_plot_currents(extent, lon=lon_mask, lat=lat_mask, 
                        v_east=v_east, v_north=v_north, 
                        flow_depth_integ=flow_depth_integ,
                        title=title, vmax=np.nanquantile(flow_depth_integ,0.95), 
                        domain_data=domains_ti, domain_lat=lat_dom, 
                        domain_lon=lon_dom, 
                        sla_data=None, sla_lon=sla_lon, sla_lat=sla_lat,
                        sla_vmin=-0.1, sla_vmax=0.1,
                        timestep=timestep)    



#%%




