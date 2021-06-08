




import os
try: 
    os.chdir("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")
    # sys.path.append("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")
except FileNotFoundError:
    os.chdir("G:/Eigene Dateien/Studium/10. Semester/NIOZ/")
    # sys.path.append("G:/Eigene Dateien/Studium/10. Semester/NIOZ/")
from dMaps_SLV import dMaps_utils as dMaps
from dMaps_SLV.network import network_analysis_utils as nau
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from  cartopy import crs as ccrs, feature as cfeature
import pandas as pd
import matplotlib as mpl
import cmocean.cm as  cm





def arrow_data(ds, extent, timestep, param):
        
    # extent = [-170, -10, -60, 10]
    
    if extent[0]<0:
        extent[0] = extent[0]+360
    if extent[1]<0:
        extent[1] = extent[1]+360
    
    lat = ds.coords["lat"]
    lon = ds.coords["lon"]
    # time = ds.coords["time"]
    # depth = ds.coords["depth"]
    
    lat_mask = lat[(lat>extent[2]) & (lat < extent[3])]
    lon_mask = lon[(lon>extent[0]) & (lon < extent[1])]
    
    vnorth = ds["v10"].loc[dict(lon=lon_mask, lat=lat_mask, 
                               time=ds.time[timestep], expver=1)]
    veast = ds["u10"].loc[dict(lon=lon_mask, lat=lat_mask, 
                              time=ds.time[timestep], expver=1)]
    
    if param == "wind stress":
        vnorth, veast = nau.wind_to_stress(vnorth, veast)
    elif param == "wind":
        vnorth = vnorth.values
        veast = veast.values
    else:
        raise ValueError("Wrong parameter. Only supports wind or wind stress!")
    velocity = np.sqrt(veast**2 + vnorth**2)
    
    # only surface winds are needed - no depth integration required
    return veast, vnorth, velocity, lon_mask, lat_mask

def ap_data(ds, extent, timestep):
        
    # extent = [-170, -10, -60, 10]
    
    if extent[0]<0:
        extent[0] = extent[0]+360
    if extent[1]<0:
        extent[1] = extent[1]+360
    
    lat = ds.coords["lat"]
    lon = ds.coords["lon"]
    
    lat_mask = lat[(lat>extent[2]) & (lat < extent[3])]
    lon_mask = lon[(lon>extent[0]) & (lon < extent[1])]
    
    ap = ds["msl"].loc[dict(lon=lon_mask, lat=lat_mask, 
                               time=ds.time[timestep], expver=1)]
    # convert Pa to hPa
    ap = ap/100

    return ap, lon_mask, lat_mask

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

def arrow_plot_currents(extent, lon, lat, v_east, v_north, velocity,
                        title, vmin=0, vmax=None, 
                        domain_data=None, domain_lat=None, domain_lon=None,
                        sla_data=None, sla_lat=None, sla_lon=None, 
                        sla_vmin=-0.1, sla_vmax=0.1,
                        ap_data=None, ap_lat=None, ap_lon=None,
                        arrow_label='Arrow color: \nWind seed anomaly [m s$^{-1}$]',
                        anomaly = False,
                        timestep=0, out_fname=None,
                        ax=None, fig=None, add_colorbars=True,
                        draw_grid_labels=True):
    
    
    
    
    if ax is None:
        fig = plt.figure(figsize=(12, 15))
        ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    else:
        ax1=ax
    #ax1.coastlines('50m')
    ax1.set_extent(extent, ccrs.PlateCarree())
    if vmax is None:
        vmax = np.nanmax(velocity)
        

                          
    # plot sea level anomalies
    if sla_data is not None:
        ax1.pcolormesh(sla_lon, sla_lat, sla_data, 
                              transform = ccrs.PlateCarree(), 
                              cmap = cm.balance, vmin = sla_vmin, 
                              vmax = sla_vmax)
    
    if ax is not None:
        if domain_data is not None:
            for i in range(domain_data.shape[0]):
                ax1.contour(domain_lon, domain_lat, domain_data[i,:,:], 
                              levels = 0, transform = ccrs.PlateCarree(), 
                              linewidths = 1,#2,
                              colors=['gray'])#, vmin = domain_vmin, 
                              #vmax = domain_vmax)    
                         
            
        # cbar_dat = ax1.streamplot(lon.values, lat.values, v_east.values, v_north.values, color=velocity, 
        #                       transform=ccrs.PlateCarree(), cmap='viridis', 
        #                       norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), density=1)
        cbar_dat = ax1.quiver(lon.values, lat.values, v_east.values, v_north.values, velocity, 
                              transform=ccrs.PlateCarree(), cmap=cm.thermal, 
                              norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                              width=0.005,
                              )    
        
        ax1.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m"), 
                           facecolor='xkcd:grey', zorder=0)
        
    
        # Add isobares if requested
        if ap_data is not None:
            if anomaly==False:
                levels= [988,992,996,1000,1004,1008,1012,1016,1020, 1024, 1028]
                lw = 1.5
                fsz=7
            else:
                levels=[-10,-5,0,5,10] #[-8,-6,-4,-2,0,2,4,6,8]
                lw = 1.5
                fsz = 10
            filled_c = ax1.contourf(ap_lon, ap_lat, ap_data, transform=ccrs.PlateCarree(), alpha=0.,
                                   levels=levels)
            
            line_cc = ax1.contour(lon, lat, ap_data, levels=filled_c.levels,
                                    #colors=['black'], 
                                    colors='k', vmin=-10, vmax=10, linewidths=lw+0.5, #cm.balance, vmin=-8, vmax=8,linewidths=lw,
                                    #linestyles='dashdot',
                                    transform=ccrs.PlateCarree(), zorder=50)        
            
            line_c = ax1.contour(lon, lat, ap_data, levels=filled_c.levels,
                                    #colors=['black'], 
                                    cmap = cm.balance, vmin=-10, vmax=10, linewidths=lw, #cm.balance, vmin=-8, vmax=8,linewidths=lw,
                                    #linestyles='dashdot',
                                    transform=ccrs.PlateCarree(), zorder=50)
            
            if anomaly==False:
                levels=[990,994,998,1002, 1006, 1010, 1014, 1018,1022,1026,1030]
            else:
                levels=[-7.5,-2.5,2.5,7.5] #[-7,-5,-3,-1,1,3,5,7]
            # filled_c5 = ax1.contourf(lon, lat, ap_data, transform=ccrs.PlateCarree(), alpha=0.,
            #                        levels=levels)
            
            
            # ax1.contour(lon, lat, ap_data, levels=filled_c5.levels,
            #             # colors=['black'], 
            #             colors='k', vmin=-10, vmax=10, linewidths=lw-1, #cm.balance, vmin=-8, vmax=8, linewidths=lw,
            #             # linestyles='dotted',
            #             transform=ccrs.PlateCarree())        
            
            
            # ax1.contour(lon, lat, ap_data, levels=filled_c5.levels,
            #             # colors=['black'], 
            #             cmap= cm.curl, vmin=-10, vmax=10, linewidths=lw-2, #cm.balance, vmin=-8, vmax=8, linewidths=lw,
            #             #linestyles='dotted',
            #             transform=ccrs.PlateCarree())
            
            
            
            # Use the line contours to place contour labels.
            ax1.clabel(
                    line_c,  # Typically best results when labelling line contours.
                    colors=['black'],
                    manual=False,  # Automatic placement vs manual placement.
                    inline=True,  # Cut the line where the label will be placed.
                    fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
                    fontsize=fsz,
                )
            
            ax1.clabel(
                    line_cc,  # Typically best results when labelling line contours.
                    colors=['black'],
                    manual=False,  # Automatic placement vs manual placement.
                    inline=True,  # Cut the line where the label will be placed.
                    fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
                    fontsize=fsz,
                )    
    
    else:
    # plot contours of domains
        if domain_data is not None:
            for i in range(domain_data.shape[0]):
                ax1.contour(domain_lon, domain_lat, domain_data[i,:,:], 
                              levels = 0, transform = ccrs.PlateCarree(), 
                              linewidths = 2,
                              colors=['gray'])#, vmin = domain_vmin, 
                              #vmax = domain_vmax)    
                         
            
        # cbar_dat = ax1.streamplot(lon.values, lat.values, v_east.values, v_north.values, color=velocity, 
        #                       transform=ccrs.PlateCarree(), cmap='viridis', 
        #                       norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), density=1)
        cbar_dat = ax1.quiver(lon.values, lat.values, v_east.values, v_north.values, velocity, 
                              transform=ccrs.PlateCarree(), cmap=cm.thermal, 
                              norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                              # width=0.005,
                              )    
        
        ax1.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m"), 
                           facecolor='xkcd:grey', zorder=0)
        
    
        # Add isobares if requested
        if ap_data is not None:
            if anomaly==False:
                levels= [988,992,996,1000,1004,1008,1012,1016,1020, 1024, 1028]
                lw = 1.5
                fsz=15#7
            else:
                levels=[-8,-6,-4,-2,0,2,4,6,8]
                lw = 3
                fsz = 15
            filled_c = ax1.contourf(ap_lon, ap_lat, ap_data, transform=ccrs.PlateCarree(), alpha=0.,
                                   levels=levels)
            
  
            line_cc = ax1.contour(lon, lat, ap_data, levels=filled_c.levels,
                                    #colors=['black'], 
                                    colors=['black'], vmin=-8, vmax=8,linewidths=lw+0.5,
                                    #linestyles='dashdot',
                                    transform=ccrs.PlateCarree(), zorder=50)
            
            line_c = ax1.contour(lon, lat, ap_data, levels=filled_c.levels,
                                    #colors=['black'], 
                                    cmap = cm.balance, vmin=-8, vmax=8,linewidths=lw,
                                    #linestyles='dashdot',
                                    transform=ccrs.PlateCarree(), zorder=50)
            
            if anomaly==False:
                levels=[990,994,998,1002, 1006, 1010, 1014, 1018,1022,1026,1030]
            else:
                levels=[-7,-5,-3,-1,1,3,5,7]
            filled_c5 = ax1.contourf(lon, lat, ap_data, transform=ccrs.PlateCarree(), alpha=0.,
                                    levels=levels)
            
            
            ax1.contour(lon, lat, ap_data, levels=filled_c5.levels,
                        # colors=['black'], 
                        cmap= cm.balance, vmin=-8, vmax=8, linewidths=lw,
                        linestyles='dotted',
                        transform=ccrs.PlateCarree())
            
            
            
            # Use the line contours to place contour labels.
            ax1.clabel(
                    line_c,  # Typically best results when labelling line contours.
                    colors=['black'],
                    manual=False,  # Automatic placement vs manual placement.
                    inline=True,  # Cut the line where the label will be placed.
                    fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
                    fontsize=fsz,
                )
            
            ax1.clabel(
                    line_cc,  # Typically best results when labelling line contours.
                    colors=['black'],
                    manual=False,  # Automatic placement vs manual placement.
                    inline=True,  # Cut the line where the label will be placed.
                    fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
                    fontsize=fsz,
                )            
    
    
    if add_colorbars==True:
        
        #plt.colorbar(dom_cf, ax=ax1, orientation='horizontal')
        sm = plt.cm.ScalarMappable(cmap=cm.balance, 
                                   norm=plt.Normalize(vmin = sla_vmin,
                                                      vmax = sla_vmax))
        
        cbaxes = fig.add_axes([0.125, 0.15, 0.35, 0.04])
        sm._A = []
        fig.colorbar(sm, cax=cbaxes, orientation='horizontal', 
                     label='Background color: \nSea level anomaly [cm]', 
                     shrink=0.3, pad=0.05)    
        
        
        cbaxes2 = fig.add_axes([0.55, 0.15, 0.35, 0.04])   
        fig.colorbar(cbar_dat, cax=cbaxes2, orientation='horizontal',
                     label=arrow_label, 
                     shrink=0.3, pad=0.05)
        
    
    #add_aligned_cmap(fig = fig, data = dom_cf)#cbar_dat, label="[Sv]")
    #add_aligned_cmap(fig = fig, data = cbar_dat, label="[Sv]")
    
    ax1.set_title(title)
    
    if draw_grid_labels==True:
        g1 = ax1.gridlines(draw_labels=True)
        g1.top_labels = False
        g1.right_labels = False
    else:
        g1 = ax1.gridlines(draw_labels=False)
    #plt.show()
    # out_fname = "playground/plots/Network/flow/Animation/fdir_arrows/flow_domains_49_25_{year}-{month}.png".format(
    #             year=title[-4:], month=title[-7:-5])
    if out_fname:
        plt.savefig(out_fname, bbox_inches = 'tight')
    #plt.show()
    
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
if __name__ == "__main__":
    #extent = [-130, -65, -55, 0]
    extent = [-150, -50, -66, 10]
    
    # Import nc file with ERA5 data (wind anomalies)
    nc_fname = "data/ERA5/ERA5_wind_pressure_05_anomalies_2_deg.nc"
    # nc_fname = "data/ERA5/ERA5_wind_pressure_2_deg.nc"
    ds = xr.open_dataset(nc_fname)
    
    ncfile = "data/AVISO/AVISO_MSLA_1993-2020.nc"#"dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"
    ds_sla = xr.open_dataset(ncfile)
    
    time = ds_sla.time.values
    
    # Prepare domain signal data
    domain_ids = [5, 7, 18, 25, 43, 49, 59]
    fpath = "dMaps_SLV/results/dMaps/res_2_k_11_gaus/domain_identification/domain_maps.npy"
    domains = np.load(fpath)
    domain_mask = domains[domain_ids, :, :]
        
    
    # sla = dMaps.importNetcdf(ncfile, 'sla')
    fpath_2_deg = "dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"
    lat_dom = dMaps.importNetcdf(fpath_2_deg, 'lat')
    lon_dom = dMaps.importNetcdf(fpath_2_deg, 'lon')
    
    # domain_signals =  dMaps.get_domain_signals(domains = domains, 
    #                                     sla = sla, lat = lat_dom, 
    #                                     signal_type = 'average')
    # domain_signals = domain_signals[:,domain_ids]
    
    # domain_vmax = np.max([abs(np.min(domain_signals)), np.max(domain_signals)])
    # domain_vmin = -domain_vmax
    
    #timestep = 1
    #%%
    for timestep in range(len(time)):
    
    
        year = pd.to_datetime(time[timestep]).year
        month = pd.to_datetime(time[timestep]).month
        title = "SLA, Wind Stress and Surface Pressure Anomalies\n{month:02}-{year}".format(
            year=year, month=month)
        
        v_east, v_north, velocity, lon_mask, lat_mask = arrow_data(ds, 
                                                            extent, 
                                                            timestep,
                                                            param = "wind stress")
        
        domains_ti = domain_mask.copy()
        # for i in range(domain_signals.shape[1]):
        #     #domains_ti[i,:,:][domains_ti[i,:,:]==0] = np.nan
        #     domains_ti[i,:,:][domains_ti[i,:,:]==1] = domain_signals[timestep, i]
            
        ap, ap_lon, ap_lat = ap_data(ds, extent, timestep)
            
        sla, sla_lon, sla_lat = sla_data(ds_sla, extent, timestep)
        
        #%%
        outdir = "playground/plots/Network/flow/Animation/sla_wind_stress_pressure_SO_anomalies_eddies/"
        out_fname = outdir + "timestep_{timestep:03}.png".format(timestep=timestep)
        
    
        arrow_plot_currents(extent, lon=lon_mask, lat=lat_mask, 
                            v_east=v_east, v_north=v_north, 
                            velocity=velocity,
                            title=title, vmax=0.025, 
                            domain_data=domains_ti, domain_lat=lat_dom, 
                            domain_lon=lon_dom, 
                            sla_data=sla, sla_lon=sla_lon, sla_lat=sla_lat,
                            sla_vmin=-0.13, sla_vmax=0.13, #sla_vmin=-0.09, sla_vmax=0.09,
                            ap_data=ap, ap_lat=ap_lat, ap_lon=ap_lon,
                            arrow_label='Arrow color: \nWind stress anomaly [N m$^{-2}$]',
                            anomaly = True,
                            timestep=timestep,
                            out_fname=out_fname)    
    
    
    
    #%% create gif
    
    fps_list = [0.5,1,2,5,10]
    for fps in fps_list:
        out_fname = outdir + "sla_wind_stress_pressure_anomalies_{fps}_fps.gif".format(fps=fps)
        
        nau.create_gif(indir = outdir, 
                       outfname = out_fname, 
                       fps = fps,
                       extension = ".png")
    

