
from tigramite.independence_tests import ParCorr


def run_pcmci_on_dMaps(dmaps_outpath, ncfile, 
                       tau_min=0, tau_max=12, out_fname = None,
                       nc_var='sla', nc_lat='lat', nc_lon='lon',
                       signal_type='cumulative', 
                       cond_ind_test=ParCorr(significance='analytic'),
                       extent=None):
    """
    Applies PCMCI on the output of deltaMaps. The domain signal (cumulative or
    mean signal in each domain for each time step) is calculated first. Then 
    PCMCI is applied and the result visualized with a projected plot of the 
    network. Additionally, plots of the Average Causal Effect, Average Causal 
    Suscpeptebility and the Average Mediated Causal Effect are produced. If no
    out_fname is specified, the plots will only be shown.

    Parameters
    ----------
    dmaps_outpath : str
        Path to the main directrory of the deltaMaps output (i.e. the directory
        with the subdirectories "domain_identification", "network_inference" 
        and "seed_identification".
    ncfile : str
        Path and filename of the nc-file that has been used for deltaMaps.
    tau_min : int, optional
        Minimum time delay that will be tested by PCMCI. The default is 0.
    tau_max : int, optional
        Maximum time delay that will be tested by PCMCI. The default is 12.
    out_fname : str, optional
        Filepath and name for the plots that will be generated. The specified
        filename will be extended with appropriate suffixes (e.g. 
        "_network.png" for the network-plot). If no out_fname is specified, the
        plots will be shown and not saved. The default is None.
    nc_var : str, optional
        Name of the variable in the nc-file contining the data. The default is
        'sla'.
    nc_lat : str, optional
        Name of the variable containint the latitude-values. The default is 
        'lat'.
    nc_lon : str, optional
        Name of the variable containint the longitude-values. The default is 
        'lon'.
    signal_type : str, optional
        Type of the domain signal to be calculated. Can be 'cumulative' (i.e. 
        the sum of the signal of all grid cells in a domain) or 'average' (i.e. 
        the mean signal of all grid cells). The default is 'cumulative'.
    cond_ind_test : TYPE, optional
        Type of conditional independence test to be used by PCMCI. The default 
        is ParCorr(significance='analytic') -> linear partial correlation.
    extent : list, optional
        Extent of the plots in the format [lon min, lon max, lat min, lat max].
        If None is given, the plot will show the whole earth and center at 180°
        longitude. The default is None.

    Returns
    -------
    None.
    
    Usage
    -------
    dmaps_outpath = "dMaps_SLV/data/dMaps/res_2_k_11_gaus/"
    ncfile = "dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"
    tau_min = 1
    tau_max = 15
    out_fname = "dMaps_SLV/plots/tigramite/tig_dMaps_res_2_k_11_gaus"
    
    run_pcmci_on_dMaps(dmaps_outpath=dmaps_outpath, 
                       ncfile=ncfile, 
                       tau_min=tau_min, tau_max=tau_max, out_fname = out_fname,
                       signal_type='cumulative', 
                       cond_ind_test=ParCorr(significance='analytic'))

    """

    from dMaps_SLV import dMaps_utils as dMaps
    import numpy as np
    import matplotlib.pyplot as plt
    from tigramite import data_processing as pp
    from tigramite import plotting as tp
    from tigramite.pcmci import PCMCI
    from  cartopy import crs as ccrs, feature as cfeature
    from tigramite.models import LinearMediation
    
    
    # Import domain map
    d_maps = np.load(dmaps_outpath + 'domain_identification/domain_maps.npy')
    
    # Import SLA data and lat/lon data
    sla = dMaps.importNetcdf(ncfile, nc_var)
    lat = dMaps.importNetcdf(ncfile, nc_lat)
    lon = dMaps.importNetcdf(ncfile, nc_lon)
    
    
    # Produce domain signals for each domain in d_maps and time step in sla
    signals = dMaps.get_domain_signals(domains = d_maps,
                                       sla = sla, 
                                       lat = lat, 
                                       signal_type = signal_type)
    
    # Create a name for each region
    var_names = [str(i) for i in range(signals.shape[1])]
    
    # 
    dataframe = pp.DataFrame(signals, 
                             datatime = np.arange(len(signals)), 
                             var_names=var_names)
    
    
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=cond_ind_test,
        verbosity=1)
    
    # correlations = pcmci.get_lagged_dependencies(tau_max=20, val_only=True)['val_matrix']
    # lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names':var_names, 
    #                                     'x_base':5, 'y_base':.5}); plt.show()
    
    
    pcmci.verbosity = 1
    results = pcmci.run_pcmci(tau_min=tau_min, 
                              tau_max=tau_max, 
                              pc_alpha=None)
    
    #%%
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')

    
    #%%
    link_matrix = pcmci.return_significant_links(pq_matrix=q_matrix,
                            val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']
    
    
    
    #%%
    
    # Domain calculations
        
    # create an array containing all domains with their respective number as 
    # cell value
    domain_map = dMaps.get_domain_map(d_maps)
                
    # Calculate average coordinates for each domain (will be the coordinates 
    # for the nodes in the plot)
    dom_lat = []
    dom_lon = []
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
            
        
        dom_lon.append(lon[x]-180)
        dom_lat.append(lat[y])
    
    node_pos =  {'x': np.array(dom_lon),
                 'y': np.array(dom_lat)}
    
    
    
    
    #%%
    
    vmin_edges = -1
    vmax_edges = 1
    cmap_edges = "RdBu_r"
    vmin_nodes = 0
    vmax_nodes = 1
    cmap_nodes = "OrRd"
    
    
    
    crs = ccrs.PlateCarree(central_longitude=180)
        
    fig, ax =  plt.subplots(1,1,figsize=(12,8), dpi=300,
                            subplot_kw=dict(projection=crs))
    
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m"), 
                    facecolor='xkcd:grey', zorder=-100)
    
    # ax.contourf(lon, lat, strength_map, transform = ccrs.PlateCarree(), 
    #                 levels=100, cmap = "RdBu_r", 
    #                 vmin = 0, vmax = 10, zorder=-80)
    
    tp.plot_graph(
        fig_ax = (fig, ax),
        val_matrix=results['val_matrix'],
        link_matrix=link_matrix,
        var_names=var_names,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI',
        node_pos = node_pos,
        node_size = 10,
        show_colorbar = False,
        vmin_edges = vmin_edges,
        vmax_edges = vmax_edges,
        cmap_edges = cmap_edges,
        vmin_nodes = vmin_nodes,
        vmax_nodes = vmax_nodes,
        cmap_nodes = cmap_nodes
        )
    
    
    
    # Define colorbar for the nodes
    sm = plt.cm.ScalarMappable(cmap=cmap_nodes, 
                               norm=plt.Normalize(vmin = vmin_nodes,
                                                  vmax = vmax_nodes))
    
    cbaxes = fig.add_axes([0.125, 0.2, 0.35, 0.04])
    sm._A = []
    cbar_nodes = fig.colorbar(sm, cax=cbaxes, orientation='horizontal', 
                              shrink=0.3, pad=0.05)
    cbar_nodes.set_label("auto-MCI")
    # cbar_nodes.ax.set_title("Node & Domain colour")
    
    # Define colorbar for the edges
    sm = plt.cm.ScalarMappable(cmap=cmap_edges, 
                               norm=plt.Normalize(vmin = vmin_edges, 
                                                  vmax = vmax_edges))
    
    cbaxes = fig.add_axes([0.55, 0.2, 0.35, 0.04])
    sm._A = []
    cbar_edges = fig.colorbar(sm, cax=cbaxes, orientation='horizontal', 
                              shrink=0.3, pad=0.05)
    cbar_edges.set_label("cross-MCI")
    
    if out_fname is None:
        plt.show()
    else:
        plt.savefig(out_fname + "_network.png", bbox_inches = 'tight')
    
    
    
    
    
    
    #%% plot mediation graph
    
    
    
    med = LinearMediation(dataframe=dataframe)
    med.fit_model(all_parents=pcmci.all_parents)
    
    
    
    # i=5; tau=7; j=17
    # graph_data = med.get_mediation_graph_data(i=i, tau=tau, j=j)
    # tp.plot_mediation_graph(
    #                     var_names=var_names,
    #                     path_val_matrix=graph_data['path_val_matrix'], 
    #                     path_node_array=graph_data['path_node_array'],
    #                     ); plt.show()
    
    
    
    
    
    #%%
    
    
            
    
    def plot_causal_effect(data, lat, lon, vmin, vmax, cmap, title,
                           data_2=None, # vmin_2=None, vmax_2=None, cmap_2=None,
                           extent=None, out_fname=None):
        
             
                
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        
        crs = ccrs.PlateCarree(central_longitude=180)
        fig, ax =  plt.subplots(1,1,figsize=(12,8), dpi=300,
                                subplot_kw=dict(projection=crs))
        
        ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m"), 
                        facecolor='xkcd:grey', zorder=-100)
        
        da = ax.scatter(lon_plot, dom_lat, c=data, transform=ccrs.PlateCarree(),
                        vmin=vmin, vmax=vmax, cmap = cmap, s=400)
        
        if data_2 is not None:
            ax.scatter(lon_plot, dom_lat, c=data_2, 
                              transform=ccrs.PlateCarree(),
                              vmin=vmin, vmax=vmax, cmap = cmap, s=100)
        
        if extent is None:
            ax.set_global()
        else:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
        ax.set_title(title)
        
        # Add alligned cmap
        divider = make_axes_locatable(plt.gca())
        ax_cb = divider.new_horizontal(size="3%", pad=0.1, axes_class=plt.Axes)
        fig.add_axes(ax_cb)
        plt.colorbar(da, cax=ax_cb)  
        
        if out_fname is None:
            plt.show()
        else:
            plt.savefig(out_fname, bbox_inches = 'tight')
        
    #%% plot ace
    
    data = med.get_all_ace()
    vmin = 0
    vmax = np.max(data)
    cmap = 'viridis'
    
    lon_plot = [i-180 for i in dom_lon]
    title = "Average Causal Effect"
    
    if out_fname is None:
        out_fname_tmp = None
    else:
        out_fname_tmp = out_fname + "_ace.png"
    
    plot_causal_effect(data=data, 
                       lat = dom_lat,
                       lon = lon_plot,
                       vmin = vmin,
                       vmax = vmax,
                       cmap = cmap,
                       title = title,
                       out_fname = out_fname_tmp)
    
    
    #%% plot Average Causal Susceptibility
    
    data = med.get_all_acs()
    vmin = 0
    vmax = np.max(data)
    cmap = 'viridis'
    
    lon_plot = [i-180 for i in dom_lon]
    title = "Average Causal Susceptibility"
    
    if out_fname is None:
        out_fname_tmp = None
    else:
        out_fname_tmp = out_fname + "_acs.png"
    
    plot_causal_effect(data=data, 
                       lat = dom_lat,
                       lon = lon_plot,
                       vmin = vmin,
                       vmax = vmax,
                       cmap = cmap,
                       title = title,
                       out_fname = out_fname_tmp,
                       extent = extent,
                       )
    
    
    
    #%% plot Average Causal Susceptibility
    
    data = med.get_all_acs()
    vmin = 0
    
    cmap = 'viridis'
    
    data_2 = med.get_all_ace()    
    vmax = np.max([data, data_2])
    # vmin_2 = 0
    # vmax_2 = np.max(data_2)
    # cmap_2 = 'Spectral_r'
    
    lon_plot = [i-180 for i in dom_lon]
    title = "Average Causal Effect (inner dot) and Average Causal Susceptibility (outer dot)"
    
    if out_fname is None:
        out_fname_tmp = None
    else:
        out_fname_tmp = out_fname + "_amce_acs.png"
    
    plot_causal_effect(data=data, # outer dot
                       lat = dom_lat,
                       lon = lon_plot,
                       vmin = vmin,
                       vmax = vmax,
                       cmap = cmap,
                       title = title,
                       out_fname = out_fname_tmp,
                       data_2 = data_2, # inner dot
                       extent = extent,
                       #vmin_2 = vmin_2,
                       #vmax_2 = vmax_2,
                       #cmap_2 = cmap_2,
                       )    
    
    #%% plot Average Mediated Causal Effect
    
    data = med.get_all_amce()
    vmin = 0
    vmax = np.max(data)
    cmap = 'viridis'
    
    lon_plot = [i-180 for i in dom_lon]
    title = "Average Mediated Causal Effect"
    
    if out_fname is None:
        out_fname_tmp = None
    else:
        out_fname_tmp = out_fname + "_acme.png"
    
    plot_causal_effect(data=data, 
                       lat = dom_lat,
                       lon = lon_plot,
                       vmin = vmin,
                       vmax = vmax,
                       cmap = cmap,
                       title = title,
                       out_fname = out_fname_tmp,
                       extent = extent,
                       )


#%%

if __name__ == "__main__":
        
    
    import os
    try:
        os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
    except FileNotFoundError:
        os.chdir("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")    
        
    dmaps_outpath = "dMaps_SLV/data/dMaps/res_2_k_11_gaus/" #"playground/dMaps_regional/indian/data/res_2_k_11_gaus_ind/"
    ncfile = "dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc" #"data/AVISO/basins/AVISO_MSLA_1993-2020_prep_2_deg_gaus_indian.nc"
    tau_min = 1
    tau_max = 15
    out_fname = "dMaps_SLV/plots/tigramite/tig_dMaps_res_2_k_11_gaus"# "playground/dMaps_regional/indian/plots/tigramite"
    
    run_pcmci_on_dMaps(dmaps_outpath=dmaps_outpath, 
                       ncfile=ncfile, 
                       tau_min=tau_min, tau_max=tau_max, out_fname = out_fname,
                       signal_type='cumulative', 
                       cond_ind_test=ParCorr(significance='analytic'))
    
    
    
    
    