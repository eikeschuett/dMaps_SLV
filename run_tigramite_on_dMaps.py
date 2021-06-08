
from tigramite.independence_tests import ParCorr

def run_pcmci_on_dMaps(dmaps_outpath, ncfile, 
                       tau_min=0, tau_max=12, out_fname = None,
                       nc_var='sla', nc_lat='lat', nc_lon='lon',
                       signal_type='cumulative', 
                       cond_ind_test=ParCorr(significance='analytic'),
                       alpha_level=0.01,
                       results_fpath=None,
                       ci_dict=None,
                       add_domains=None):
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
    cond_ind_test : tigramite.independence_tests, optional
        Type of conditional independence test to be used by PCMCI. The default 
        is ParCorr(significance='analytic') -> linear partial correlation.
    alpha_level : float, optional
        Significance level for the edges to be drawn. The default is 0.01.
    ci_dict : dict, optional
        Optionally, additional timeseries can be added to the network, e.g. 
        climate indices. This requires a dictionary with keys "name", "fpath", 
        "lat", "lon":
        
        ci_dict = {
                    "name":["ENSO", 
                            "PNO"],
                    "fpath": ["H:/Eigene Dateien/Studium/10. Semester/NIOZ/data/climate_indices/nino_34_anomaly.txt",
                              "H:/Eigene Dateien/Studium/10. Semester/NIOZ/data/climate_indices/PNA.txt"],
                    "lat":[0,
                           50],
                    "lon":[-145,
                           -145]}
        
        If None is given, no climate indices will be used in the network 
        analysis. The default is None.

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
    from dMaps_SLV.network import network_analysis_utils as nau
    import numpy as np
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    
    
    # Import domain map
    d_maps = np.load(dmaps_outpath + 'domain_identification/domain_maps.npy')
    if type(add_domains)==np.ndarray:
        d_maps = np.vstack((d_maps, add_domains))
    
    
    # Import SLA data and lat/lon data
    sla = dMaps.importNetcdf(ncfile, nc_var)
    lat = dMaps.importNetcdf(ncfile, nc_lat)
    
    
    # Produce domain signals for each domain in d_maps and time step in sla
    signals = dMaps.get_domain_signals(domains = d_maps,
                                       sla = sla, 
                                       lat = lat, 
                                       signal_type = signal_type)
    
    # Create a name for each region
    var_names = [str(i) for i in range(signals.shape[1])]
    
    
    # Add climate indices
    if ci_dict is not None:
        for i in range(len(ci_dict["name"])):
            fpath = ci_dict["fpath"][i]
            a, _ = nau.prep_clim_index(fpath)
            a = np.array(a)
            a = a[:,np.newaxis]
            signals = np.hstack((signals, a))
            var_names.append(ci_dict["name"][i])
    
    
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
    
    # save the dictionary with the results as a pickle
    if results_fpath is not None:
        import pickle
        with open(results_fpath, "wb") as tf:
            pickle.dump(results,tf)

    return pcmci, dataframe, results, var_names


    #%%
    
def _domain_coordinates(dmaps_outpath, ncfile, nc_lat, nc_lon):
    from dMaps_SLV import dMaps_utils as dMaps
    import numpy as np
    
    
    # Import domain map
    d_maps = np.load(dmaps_outpath + 'domain_identification/domain_maps.npy')
    
    # Import SLA data and lat/lon data
    # sla = dMaps.importNetcdf(ncfile, nc_var)
    lat = dMaps.importNetcdf(ncfile, nc_lat)
    lon = dMaps.importNetcdf(ncfile, nc_lon)
    
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
    return node_pos, dom_lat, dom_lon 

#%% 
def _get_domain_map(d_maps):
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

#%%
    
def plot_pcmci_network(pcmci, results, alpha_level, dmaps_outpath, 
                       lat, lon,
                       var_names,
                       ncfile, nc_var='sla', nc_lat='lat', nc_lon='lon',
                       single_node=False,
                       out_fname=None,
                       ci_dict = None,
                       extent = None,
                       figsize=(12,8),
                       node_size=10):
    
    
    # from dMaps_SLV import dMaps_utils as dMaps
    import numpy as np
    import matplotlib.pyplot as plt
    from tigramite import plotting as tp
    from  cartopy import crs as ccrs, feature as cfeature

    
    

    
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')

    
    
    link_matrix = pcmci.return_significant_links(pq_matrix=q_matrix,
                            val_matrix=results['val_matrix'], 
                            alpha_level=alpha_level)['link_matrix']
    
    # if only edges at a single node shall be shown, replace all other edges 
    # in the link matrix with NaNs
    if single_node is not False:
        link_matrix2 = np.full(link_matrix.shape, np.nan)
        link_matrix2[:,single_node,:] = link_matrix[:,single_node,:]
        link_matrix2[single_node,:,:] = link_matrix[single_node,:,:]
        # Create a boolean array
        link_matrix = (link_matrix2==1)
    

    
    
    node_pos, dom_lat, dom_lon = _domain_coordinates(dmaps_outpath = dmaps_outpath, 
                                                     ncfile = ncfile, 
                                                     nc_lat = nc_lat, 
                                                     nc_lon = nc_lon)
    
    if ci_dict is not None:
        node_pos["x"] = np.append(node_pos["x"], [i+180 for i in ci_dict["lon"]])
        node_pos["y"] = np.append(node_pos["y"], ci_dict["lat"])
    
    domain_map = _get_domain_map(np.load(dmaps_outpath + 
                                         '/domain_identification/domain_maps.npy'))
    
    vmin_edges = -1
    vmax_edges = 1
    cmap_edges = "RdBu_r"
    vmin_nodes = 0
    vmax_nodes = 1
    cmap_nodes = "OrRd"
    
    
    
    crs = ccrs.PlateCarree(central_longitude=180)
        
    fig, ax =  plt.subplots(1,1,figsize=figsize, dpi=300,
                            subplot_kw=dict(projection=crs))
    
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m"), 
                    facecolor='xkcd:grey', zorder=-100)
    
    ax.contourf(lon, lat, domain_map, transform = ccrs.PlateCarree(), 
                    levels=100, cmap = "prism", 
                    vmin = 0, vmax = 101, zorder=-80,
                    alpha = 0.5)
    
    tp.plot_graph(
        fig_ax = (fig, ax),
        val_matrix=results['val_matrix'],
        link_matrix=link_matrix,
        var_names=var_names,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI',
        node_pos = node_pos,
        node_size = node_size,
        show_colorbar = False,
        vmin_edges = vmin_edges,
        vmax_edges = vmax_edges,
        cmap_edges = cmap_edges,
        vmin_nodes = vmin_nodes,
        vmax_nodes = vmax_nodes,
        cmap_nodes = cmap_nodes
        )
    
    
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent)
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
    
    
def plot_mediation_graph(pcmci, dataframe, i, j, tau):

    from tigramite.models import LinearMediation  
    from tigramite import plotting as tp    
    import matplotlib.pyplot as plt
    
    med = LinearMediation(dataframe=dataframe)
    med.fit_model(all_parents=pcmci.all_parents)
    
    
    
    #i=5; tau=7; j=17
    graph_data = med.get_mediation_graph_data(i=i, tau=tau, j=j)
    tp.plot_mediation_graph(
                        var_names=var_names,
                        path_val_matrix=graph_data['path_val_matrix'], 
                        path_node_array=graph_data['path_node_array'],
                        ); plt.show()
    
    
    
    
    
    #%%
    
    
            
    
def _plot_causal_effect(data, lat, lon, vmin, vmax, cmap, title,
                       data_2=None, # vmin_2=None, vmax_2=None, cmap_2=None,
                       extent=None, out_fname=None):
                
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        
    if extent is None:
        crs = ccrs.PlateCarree(central_longitude=180)
    else:
        crs = ccrs.PlateCarree()
    fig, ax =  plt.subplots(1,1,figsize=(12,8), dpi=300,
                            subplot_kw=dict(projection=crs))
    
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m"), 
                   facecolor='xkcd:grey', zorder=-100)
    
    da = ax.scatter(lon, lat, c=data, transform=ccrs.PlateCarree(),
                    vmin=vmin, vmax=vmax, cmap = cmap, s=400)
    
    if data_2 is not None:
        ax.scatter(lon, lat, c=data_2, 
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
        

    
    #%% plot Average Causal Effect & Average Causal Susceptibility
def plot_ACE_ACS(pcmci, dataframe, dmaps_outpath, 
                 ncfile, nc_lat='lat', nc_lon='lon',
                 extent = None, out_fname = None,
                 ci_dict = None):
    
    from tigramite.models import LinearMediation  
    import numpy as np
    
    node_pos, dom_lat, dom_lon = _domain_coordinates(dmaps_outpath = dmaps_outpath, 
                                                     ncfile = ncfile, 
                                                     nc_lat = nc_lat, 
                                                     nc_lon = nc_lon)
    
    med = LinearMediation(dataframe=dataframe)
    med.fit_model(all_parents=pcmci.all_parents)
    
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
    
    if ci_dict is not None:
        lon_plot = np.append(lon_plot, [i for i in ci_dict["lon"]])
        dom_lat = np.append(dom_lat, ci_dict["lat"])
    
    _plot_causal_effect(data=data, # outer dot
                        lat = dom_lat,
                        lon = lon_plot,
                        vmin = vmin,
                        vmax = vmax,
                        cmap = cmap,
                        title = title,
                        out_fname = out_fname_tmp,
                        data_2 = data_2, # inner dot
                        extent = extent,
                        )    
    
    #%% plot Average Mediated Causal Effect
def plot_AMCE(pcmci, dataframe, dmaps_outpath, 
                 ncfile, nc_lat='lat', nc_lon='lon',
                 extent = None, out_fname = None):
    
    from tigramite.models import LinearMediation  
    import numpy as np
    
    node_pos, dom_lat, dom_lon = _domain_coordinates(dmaps_outpath = dmaps_outpath, 
                                                     ncfile = ncfile, 
                                                     nc_lat = nc_lat, 
                                                     nc_lon = nc_lon)
    
    med = LinearMediation(dataframe=dataframe)
    med.fit_model(all_parents=pcmci.all_parents)
    
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
    
    _plot_causal_effect(data=data, 
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
def _load_data(fpath,
               dmaps_outpath, ncfile, 
               nc_var='sla', nc_lat='lat', nc_lon='lon',
               signal_type='cumulative', 
               cond_ind_test=ParCorr(significance='analytic'),
               ci_dict=None):
    
    import pickle
    from dMaps_SLV import dMaps_utils as dMaps
    import numpy as np
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    
    # Import results
    with open(fpath, 'rb') as f:
         results = pickle.load(f)
       
    # Create PCMCI Object
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
    
        # Add climate indices
    if ci_dict is not None:
        for i in range(len(ci_dict["name"])):
            var_names.append(ci_dict["name"][i])
    
    # 
    dataframe = pp.DataFrame(signals, 
                             datatime = np.arange(len(signals)), 
                             var_names=var_names)
    
    
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=cond_ind_test,
        verbosity=1)   
    return pcmci, dataframe, results, var_names, lat, lon



def lag_func_matrix(pcmci, tau_max, ncfile, 
                    nc_var='sla', nc_lat='lat', 
                    signal_type='cumulative', 
                    cond_ind_test=ParCorr(significance='analytic')):
    
    from tigramite import plotting as tp  
    from dMaps_SLV import dMaps_utils as dMaps
    import numpy as np
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    import math

    
    
    # Import domain map
    d_maps = np.load(dmaps_outpath + 'domain_identification/domain_maps.npy')
    
    # Import SLA data and lat/lon data
    sla = dMaps.importNetcdf(ncfile, nc_var)
    lat = dMaps.importNetcdf(ncfile, nc_lat)
    
    
    # Produce domain signals for each domain in d_maps and time step in sla
    signals_all = dMaps.get_domain_signals(domains = d_maps,
                                       sla = sla, 
                                       lat = lat, 
                                       signal_type = signal_type)
    
    it = math.ceil(signals_all.shape[1]/10)-1
    
    for i in range(it-1):
        idx = np.r_[0:10,10*i+10:10*i+20]
        signals = signals_all[:,idx]
        
        # Create a name for each region
        var_names = [str(i) for i in idx]
        
        # 
        dataframe = pp.DataFrame(signals, 
                                 datatime = np.arange(len(signals)), 
                                 var_names=var_names)
        
        
        pcmci = PCMCI(
            dataframe=dataframe, 
            cond_ind_test=cond_ind_test,
            verbosity=1)
        
        
        correlations = pcmci.get_lagged_dependencies(tau_max=tau_max, val_only=True)['val_matrix']
        
        # fig, ax = plt.subplots(figsize=(10,10))
        test = tp.setup_matrix(N=20, tau_max=12, var_names=var_names, figsize=(20,20))
        
        test.add_lagfuncs(val_matrix=correlations, sig_thres=None, conf_matrix=None, color='black', label=None, two_sided_thres=True, marker='.', markersize=5, alpha=1.0)
        
        test.savefig("dMaps_SLV/results/plots/tigramite/Tau_matrix_" + str(i) + ".png")
    
        
    
    
#%%

if __name__ == "__main__":
        
    
    import os
    try:
        os.chdir("/mnt/h/Eigene Dateien/Studium/10. Semester/NIOZ/")
    except FileNotFoundError:
        os.chdir("H:/Eigene Dateien/Studium/10. Semester/NIOZ/")    
        
    # deltaMaps result-folder for the k-value that performs best
    dmaps_outpath = "dMaps_SLV/results/dMaps/res_2_k_11_gaus/"
    # original SLA nc-file (coordinates are required for plotting)
    ncfile = "dMaps_SLV/data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc"
    
    # Minimim tau value (if 0, contemporaneous links are included as well. To
    # exclude them, use tau_min=1.)
    tau_min = 0
    # Maximum time lag considered
    tau_max = 15
    
    # 
    out_fname = None #"dMaps_SLV/results/plots/tigramite/tig_dMaps_res_2_k_11_gaus_mean"
    
    # Save results of PCMCI run as pickle
    results_fpath = "dMaps_SLV/results/tigramite/results_taumin_{tmin}_taumax_{tmax}.pkl".format(
        tmin = tau_min,
        tmax = tau_max)
    
    # ci_dict = {
    # "name":["ENSO", 
    #         "PNO"],
    # "fpath": ["H:/Eigene Dateien/Studium/10. Semester/NIOZ/data/climate_indices/nino_34_anomaly.txt",
    #           "H:/Eigene Dateien/Studium/10. Semester/NIOZ/data/climate_indices/PNA.txt"],
    # "lat":[0,
    #        70],
    # "lon":[-145,
    #        -145]}
        
    pcmci, dataframe, results, var_names = run_pcmci_on_dMaps(
                        dmaps_outpath=dmaps_outpath, 
                        ncfile=ncfile, 
                        tau_min=tau_min, tau_max=tau_max, out_fname = out_fname,
                        signal_type='average', #'cumulative', 
                        alpha_level = 0.01,
                        cond_ind_test=ParCorr(significance='analytic'),
                        results_fpath = results_fpath,
                        ci_dict = None,
                        )
    
    # Get lat and lon vector for plotting of results
    _, _, _, _, lat, lon = _load_data(fpath = results_fpath,
                                      dmaps_outpath = dmaps_outpath, 
                                      ncfile = ncfile)    
    
    # Optionally, all results can be loaded fromthe pickle file (e.g. if you 
    # don't want to run PCMCI again after restarting the kernel)
    # pcmci, dataframe, results, var_names, lat, lon = _load_data(fpath = results_fpath,
    #                                                   dmaps_outpath = dmaps_outpath, 
    #                                                   ncfile = ncfile,
    #                                                   ci_dict = ci_dict)
    
    # lag_func_matrix(pcmci=pcmci, tau_max=12)
    

    # Specify if you want to plot the full network or only conections to/from
    # a single domain
    single_node = None
    
    out_fname = "dMaps_SLV/results/plots/tigramite/tig_dMaps_res_2_k_11_gaus_taumin_{tmin}_network.png".format(
                                                                tmin = tau_min)

    
    plot_pcmci_network(pcmci, 
                       results, 
                       alpha_level = 0.01, 
                       dmaps_outpath = dmaps_outpath, 
                       var_names = var_names,
                       ncfile = ncfile, 
                       lat = lat,
                       lon = lon,
                       single_node = single_node,
                       out_fname = out_fname,
                       ci_dict = None,
                       figsize = (24,16),#(12,8),
                       )
    