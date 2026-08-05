from io import StringIO
import os
from owslib.ogcapi.features import Features
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import interp1d
import multiprocessing as mp
import logging
import time

# Converts datetime columns to tz-aware in UTC
def convert_to_utc(df, col_name, prov=None):
    tz_map = {
        'AB': 'America/Edmonton',
        'AK': 'America/Anchorage',
        'BC': 'America/Edmonton',
        'ID': 'America/Edmonton',
        'MB': 'America/Edmonton',
        'ME': 'America/New_York',
        'MI': 'America/New_York',
        'MN': 'America/Chicago',
        'MT': 'America/Edmonton',
        'NB': 'America/Halifax',
        'NL': 'America/St_Johns',
        'NS': 'America/Halifax',
        'NT': 'America/Edmonton',
        'NU': 'America/Edmonton',
        'ON': 'America/Toronto',
        'PE': 'America/Halifax',
        'QC': 'America/Montreal',
        'SK': 'America/Edmonton',
        'WA': 'America/Los_Angeles',
        'YT': 'America/Edmonton',
    }

    tz = tz_map.get(prov, 'UTC')

    df[col_name] = pd.to_datetime(df[col_name], errors='coerce')

    if df[col_name].dt.tz is None:
        df[col_name] = df[col_name].dt.tz_localize(
            tz,
            nonexistent='shift_forward',
            ambiguous='NaT'
        )

    df[col_name] = df[col_name].dt.tz_convert('UTC')

    return df

# Function to add WSE data to the plot
def add_wse_to_plot(df, ax, name, par_column='wse', date_column='time_str', color_series=None, quality_column=None, markertype='o', markersize=5):
    if not df.empty:
        if color_series=='blue':
            color_map = {"0": "blue", "1": "lightblue"}  # Blue for 0, Light Blue for 1
        if color_series=='red':
            color_map = {"0": "red", "1": "lightcoral"}  # Red for 0, Light Coral for 1
        if quality_column is not None:
            colors = df[quality_column].astype(str).map(color_map)
            ax.scatter(
                df[date_column],
                df[par_column],
                c=colors,
                label=name,
                marker=markertype,
                s=markersize
            )
            #ax.plot(df[date_column], df[par_column], c=colors, label=f'{name}', marker=markertype, linestyle='None', markersize=markersize)
        else:
            ax.plot(
                df[date_column],
                df[par_column],
                label=name,
                marker=markertype,
                linestyle='None',
                c='black',
                markersize=markersize
            )
            #ax.plot(df[date_column], df[par_column], label=f'{name}', marker=markertype, linestyle='None', markersize=markersize)

# Merges multiple DataFrames on a datetime column
def merge_dfs_keep_last(dfs, datetime_col='Date'):
    # Concatenate all DataFrames (ignore_index) and drop all-empty columns
    dfs = [df for df in dfs if df is not None and not df.empty]
    merged = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Drop duplicates and sort if there are any rows
    if not merged.empty:
        merged = merged.drop_duplicates(subset=[datetime_col], keep='last') \
                       .sort_values(datetime_col) \
                       .reset_index(drop=True)
    else:
        merged = pd.DataFrame(columns=[datetime_col,'Level'])

    return merged

# Interpolates reference dataframe values onto target datetime array
def interpolate_to_target(reference_df, ref_time_col, ref_value_col, target_times):
    """
    Interpolate reference dataframe values onto target datetime array.
    
    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference dataframe with datetime and value columns.
    ref_time_col : str
        Name of the datetime column in reference_df (tz-aware in UTC).
    ref_value_col : str
        Name of the column to interpolate.
    target_times : pd.Series or pd.DatetimeIndex
        Target timestamps (tz-aware in UTC).
    
    Returns
    -------
    np.ndarray
        Interpolated values at target timestamps.
    """
    # Convert to numeric seconds since epoch
    reference_df = reference_df.sort_values(ref_time_col)
    ref_numeric = reference_df[ref_time_col].astype(np.int64) / 1e9
    target_numeric = target_times.astype(np.int64) / 1e9
    
    f = interp1d(ref_numeric, reference_df[ref_value_col], kind='linear',
                 bounds_error=False, fill_value='extrapolate')
    
    return f(target_numeric)

# Worker initializer to set up logging for each process
def init_worker(log_dir="logs"):

    os.makedirs(log_dir, exist_ok=True)

    pid = os.getpid()

    logger = logging.getLogger("diagnostics")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(processName)s | %(levelname)s | %(message)s"
    )

    log_file = os.path.join(
        log_dir,
        f"diagnostics_{pid}.log"
    )

    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

# Merges SWOT node dataframes based on quality and time
def merge_swot_nodes(df1, df2, df3):

    # Remove the following fields from all dataframes
    fields_to_remove = ['sword_version', 'wse_units', 'width_units', 'geoid_hght_units', 'xtrk_dist_units', 'node_dist_units', 'area_wse_units', 'n_good_pix_units', 'wse_sm_units']

    df1 = df1.drop(columns=fields_to_remove, errors='ignore')
    df2 = df2.drop(columns=fields_to_remove, errors='ignore')
    df3 = df3.drop(columns=fields_to_remove, errors='ignore')

    # Add suffixes
    dfs = [
        df1.add_suffix("_1").rename(columns={"time_str_1": "time_str"}),
        df2.add_suffix("_2").rename(columns={"time_str_2": "time_str"}),
        df3.add_suffix("_3").rename(columns={"time_str_3": "time_str"}),
    ]

    merged = dfs[0].merge(dfs[1], on="time_str", how="outer")
    merged = merged.merge(dfs[2], on="time_str", how="outer")

    # Columns that exist in the original dataframe
    cols = df1.columns.drop("time_str")

    out = pd.DataFrame({"time_str": merged["time_str"]})

    # Quality columns
    qcols = ["node_q_1", "node_q_2", "node_q_3"]

    # Missing quality -> very poor quality
    quality = merged[qcols].fillna(np.inf)

    # Which dataframe has the best quality?
    # idxmin() naturally breaks ties by taking the first column,
    # so df1 > df2 > df3.
    best = quality.idxmin(axis=1)

    for col in cols:
        out[col] = np.nan

        for i in (1, 2, 3):
            mask = best == f"node_q_{i}"
            src = f"{col}_{i}"
            if src in merged.columns:
                out.loc[mask, col] = merged.loc[mask, src]

    return out.sort_values("time_str").reset_index(drop=True)

# Function to process each station and compute statistics
def process_station(args):
    INVALID_WSE = -999999999999
    ICE_THRESHOLD = 0   # 0 for likely not ice covered, 1 includes may or may not be covered, 2 for all data including likely fully covered with ice
    NODE_QUALITY_THRESHOLD = 2  # 0 for good, 1 includes suspect, 2 includes degraded, 3 includes bad
    USE_3NODES = True

    logger = logging.getLogger("diagnostics")

    el, row = args
    station = row["STATION_NU"]
    sword_node_id = row["node_id"]
    sword_node_id_2 = row["node_id_2"]
    sword_node_id_3 = row["node_id_3"]
    sword_reach_id = row["reach_id"]
    elevation_wsc = row["STN_DATU_2"]
    datum_to = row["STN_DATU_1"]
    cgg2013 = row["CGVD20131"]
    cgg28 = row["CGVD281"]
    province = row["PROV_TERR_"]

    # Gets the conversion values to NAD83
    conversion_from_cgvd2013 = cgg2013
    conversion_from_cgvd28 = cgg28

    # List of datums corresponding to CGVD28 and CGVD2013
    datum_28 = [10,11,12,13,14,35,40,110,405,609,610] + list(range(113,194))
    datum_2013 = [605,606,607,608]

    if datum_to in datum_28:
        conversion = conversion_from_cgvd28
    elif datum_to in datum_2013:
        conversion = conversion_from_cgvd2013
    else:
        conversion = 0.0

    logger.info(f"Index in gdf: {el}")
    logger.info(f"Station: {station}")
    logger.info(f"SWOT Node ID: {sword_node_id}")
    logger.info(f"SWOT Reach ID: {sword_reach_id}")

    # Read data from csv files
    try:
        data_swot_node = pd.read_csv('./csv_data/data_SWOT_Node_' + str(sword_node_id) + '.csv')
        data_swot_node['time_str'] = pd.to_datetime(data_swot_node['time_str'], format='%Y-%m-%d %H:%M:%S')
        data_swot_node.sort_values(by='time_str', inplace = True)
        
        data_swot_reach = pd.read_csv('./csv_data/data_SWOT_Reach_' + str(sword_reach_id) + '.csv')
        data_swot_reach['time_str'] = pd.to_datetime(data_swot_reach['time_str'], format='%Y-%m-%d %H:%M:%S')
        data_swot_reach.sort_values(by='time_str', inplace = True)
        
        data_wsc_real_time = pd.read_csv('./csv_data/data_realtime_WSC_' + station + '.csv')
        data_wsc_real_time['Date'] = pd.to_datetime(data_wsc_real_time['Date'], format='%Y-%m-%d %H:%M:%S%z')
        data_wsc_real_time.sort_values(by='Date', inplace = True)
        
        data_wsc_historical = pd.read_csv('./csv_data/data_hist_WSC_' + station + '.csv')
        data_wsc_historical['Date'] = pd.to_datetime(data_wsc_historical['Date'])
        data_wsc_historical.sort_values(by='Date', inplace = True)
        
        data_wsc_api = pd.read_csv('./csv_data/data_hist_API_WSC_' + station + '.csv')
        data_wsc_api['DATE'] = pd.to_datetime(data_wsc_api['DATE'])
        data_wsc_api.sort_values(by='DATE', inplace = True)
    
    except Exception as e:
        logger.error(f"Error reading data for station {station}: {e}")
        return

    try:
        data_swot_node2 = pd.read_csv('./csv_data/data_SWOT_Node_' + str(sword_node_id_2) + '.csv')
        data_swot_node2['time_str'] = pd.to_datetime(data_swot_node2['time_str'], format='%Y-%m-%d %H:%M:%S')
        data_swot_node2.sort_values(by='time_str', inplace = True)
    except Exception as e:
        logger.error(f"Error reading data for node 2 linked to station {station}: {e}")
        return

    try:
        data_swot_node3 = pd.read_csv('./csv_data/data_SWOT_Node_' + str(sword_node_id_3) + '.csv')
        data_swot_node3['time_str'] = pd.to_datetime(data_swot_node3['time_str'], format='%Y-%m-%d %H:%M:%S')
        data_swot_node3.sort_values(by='time_str', inplace = True)
    except Exception as e:
        logger.error(f"Error reading data for node 3 linked to station {station}: {e}")
        return

    if USE_3NODES:
        try:
            # set index to time_str for concatenation
            data_swot_node.set_index('time_str', inplace=True)
            data_swot_node2.set_index('time_str', inplace=True)
            data_swot_node3.set_index('time_str', inplace=True)
            data_swot_node = merge_swot_nodes(data_swot_node.reset_index(), data_swot_node2.reset_index(), data_swot_node3.reset_index())
        except Exception as e:
            logger.error(f"Error merging SWOT nodes for station {station}: {e}")
            return

    # Filter data based on quality
    data_swot_node = data_swot_node[data_swot_node.node_q <= NODE_QUALITY_THRESHOLD]
    data_swot_reach = data_swot_reach[data_swot_reach.reach_q <= NODE_QUALITY_THRESHOLD]

    # Filter invalid wse data
    data_swot_node = data_swot_node[data_swot_node.wse != INVALID_WSE]
    data_swot_reach = data_swot_reach[data_swot_reach.wse != INVALID_WSE]
    
    # Filter for ice_clim_f and ice_dyn_f 
    data_swot_node = data_swot_node[(data_swot_node.ice_clim_f <= ICE_THRESHOLD) & (data_swot_node.ice_dyn_f <= ICE_THRESHOLD)]
    data_swot_reach = data_swot_reach[(data_swot_reach.ice_clim_f <= ICE_THRESHOLD) & (data_swot_reach.ice_dyn_f <= ICE_THRESHOLD)]

    #WSC real-time data
    subset = data_wsc_real_time[data_wsc_real_time['Parameter/Paramètre'] == 46]
    df1 = subset[['Date', 'Value/Valeur']].rename(columns={'Value/Valeur': 'Level'}).copy()
    df1 = convert_to_utc(df1, 'Date', prov=province)

    #WSC historical data
    subset = data_wsc_historical[data_wsc_historical['Parameter/Paramètre'] == 'water level/niveau']
    df2 = subset[['Date', 'Value/Valeur']].rename(columns={'Value/Valeur': 'Level'}).copy()
    df2 = convert_to_utc(df2, 'Date', prov=province)

    #WSC API data
    if data_wsc_api is None or data_wsc_api.empty:
        df3 = pd.DataFrame(columns=['Date', 'Level'])
    else:
        df3 = data_wsc_api[['DATE', 'LEVEL']].rename(columns={'DATE': 'Date', 'LEVEL': 'Level'}).copy()
        df3 = convert_to_utc(df3, 'Date', prov=province)

    # Merge WSC data and plot if available
    merged_wsc = merge_dfs_keep_last([df1, df2, df3], datetime_col='Date')
    merged_wsc = merged_wsc.dropna(subset=['Level']).reset_index(drop=True)

    # Get valid WSC data
    valid_wsc = (
        merged_wsc is not None and
        not merged_wsc.empty and
        'Level' in merged_wsc.columns and
        merged_wsc['Level'].notna().any()
    )

    # Get valid SWOT node data
    valid_node = (
        data_swot_node is not None and
        not data_swot_node.empty and
        data_swot_node['wse'].notna().any()
    )

    # Get valid SWOT reach data
    valid_reach = (
        data_swot_reach is not None and
        not data_swot_reach.empty and
        data_swot_reach['wse'].notna().any()
    )

    # Initialize statistics with NaN values
    corr_swot_node, corr_swot_reach = np.nan, np.nan
    p_node, p_reach = np.nan, np.nan
    bias_swot_node, bias_swot_reach = np.nan, np.nan
    mae_swot_node, mae_swot_reach = np.nan, np.nan
    rmse_swot_node, rmse_swot_reach = np.nan, np.nan
    spearman_swot_node, spearman_swot_reach = np.nan, np.nan
    ps_node, ps_reach = np.nan, np.nan
    good_points_N, suspect_points_N, degraded_points_N, points_reach = np.nan, np.nan, np.nan, np.nan

    # Compute statistics if valid data is available for WSC and SWOT datasets
    if valid_wsc and valid_node:
        # Apply geoid height correction to SWOT data
        data_swot_node['wse_egm'] = data_swot_node['wse'] + data_swot_node['geoid_hght']    
        data_swot_reach['wse_egm'] = data_swot_reach['wse'] + data_swot_reach['geoid_hght']

        if not np.isnan(elevation_wsc):
            # Apply elevation and datum conversion to WSC data
            merged_wsc['Level'] = merged_wsc['Level'] + elevation_wsc + conversion  

        # Interpolate WSC data to SWOT timestamps
        wsc_at_swot_node_time = interpolate_to_target(merged_wsc, 'Date', 'Level', data_swot_node['time_str'])
        wsc_at_swot_reach_time = interpolate_to_target(merged_wsc, 'Date', 'Level', data_swot_reach['time_str'])

        # Calculate number of points in SWOT node and reach datasets
        points_node = len(wsc_at_swot_node_time)
        points_reach = len(wsc_at_swot_reach_time)

        good_points_N = len(data_swot_node[data_swot_node.node_q == 0])
        suspect_points_N = len(data_swot_node[data_swot_node.node_q == 1])
        degraded_points_N = len(data_swot_node[data_swot_node.node_q == 2])

        # Checks if there is enough data points after filtering to compute statistics
        if points_node < 2 or points_reach < 2:
            logger.info(f"Not enough valid data points for station {station}. Skipping statistics.")
            return

        # If WSC elevation is available, plot the time series and scatter plots, and compute bias, RMSE and MAE
        if not np.isnan(elevation_wsc):
            '''
            # Plot WSC and SWOT data
            fig, ax = plt.subplots(figsize=(10, 5))
            add_wse_to_plot(merged_wsc, ax, 'WSC', par_column='Level', date_column='Date', markertype='.')
            add_wse_to_plot(data_swot_node, ax, 'SWOT Node', par_column='wse_egm', color_series='blue', quality_column='node_q')
            add_wse_to_plot(data_swot_reach, ax, 'SWOT Reach', par_column='wse_egm', color_series='red', quality_column='reach_q')
            ax.grid(True)
            ax.legend()
            ax.set_xlabel('Date')
            ax.set_ylabel('Water Surface Elevation (m)')
            fig.tight_layout()
            plt.savefig('./plots/plot_station_' + str(station) + '.png')
            plt.close()

            # Scatter plot of WSC vs SWOT with 1:1 line
            fig, ax = plt.subplots(figsize=(6, 6))
            color_map = {"0": "blue", "1": "lightblue"}  # Blue for 0, Light Blue for 1
            colors = data_swot_node['node_q'].astype(str).map(color_map)
            ax.scatter(wsc_at_swot_node_time, data_swot_node['wse_egm'], facecolors='none', edgecolors=colors, s=10, marker='o', label='SWOT Node')
            color_map = {"0": "red", "1": "lightcoral"}  # Red for 0, Light Coral for 1
            colors = data_swot_reach['reach_q'].astype(str).map(color_map)
            ax.scatter(wsc_at_swot_reach_time, data_swot_reach['wse_egm'], facecolors='none', edgecolors=colors, s=10, marker='o', label='SWOT Reach')
            ax.set_title(f"Scatter Plot of Station {station}\n(Node r = {corr_swot_node:.2f}, Reach r = {corr_swot_reach:.2f})")
            ax.grid(True)
            ax.legend()
            ax.set_xlabel("WSC data")
            ax.set_ylabel("SWOT data")
            fig.tight_layout()
            x1, x2 = wsc_at_swot_node_time, wsc_at_swot_reach_time
            y1, y2 = data_swot_node['wse_egm'], data_swot_reach['wse_egm']
            all_arrays = np.concatenate([x1, x2, y1, y2])
            lims = [
                all_arrays.min(),  # min of both axes
                all_arrays.max()   # max of both axes
            ]
            ax.plot(lims, lims, '--')
            plt.savefig('./plots/scatter_station_' + str(station) + '.png')
            plt.close()
            '''

            # Compute bias (mean difference) between WSC and SWOT
            delta_node = wsc_at_swot_node_time - data_swot_node['wse_egm']
            delta_reach = wsc_at_swot_reach_time - data_swot_reach['wse_egm']
            bias_swot_node = delta_node.mean()
            bias_swot_reach = delta_reach.mean()

            # Compute RMSE between WSC and SWOT
            rmse_swot_node = np.sqrt(np.nanmean((wsc_at_swot_node_time - data_swot_node['wse_egm'])**2))
            rmse_swot_reach = np.sqrt(np.nanmean((wsc_at_swot_reach_time - data_swot_reach['wse_egm'])**2))

            # Compute mean absolute error (MAE) between WSC and SWOT
            mae_swot_node = np.nanmean(np.abs(wsc_at_swot_node_time - data_swot_node['wse_egm']))
            mae_swot_reach = np.nanmean(np.abs(wsc_at_swot_reach_time - data_swot_reach['wse_egm']))

        try:
            # Compute Pearson correlation
            corr_swot_node, p_node = pearsonr(wsc_at_swot_node_time, data_swot_node['wse_egm'])
            corr_swot_reach, p_reach = pearsonr(wsc_at_swot_reach_time, data_swot_reach['wse_egm'])

            # Compute Spearman correlation
            spearman_swot_node, ps_node = spearmanr(wsc_at_swot_node_time, data_swot_node['wse_egm'])
            spearman_swot_reach, ps_reach = spearmanr(wsc_at_swot_reach_time, data_swot_reach['wse_egm'])

        except Exception as e:
            logger.error(f"Error computing statistics for station {station}: {e}")
            

    logger.info("\n")

    return {
        "index": el,
        "r_node": corr_swot_node,
        "r_reach": corr_swot_reach,
        "p_node": p_node,
        "p_reach": p_reach,
        "spearman_node": spearman_swot_node,
        "spearman_reach": spearman_swot_reach,
        "ps_node": ps_node,
        "ps_reach": ps_reach,
        "b_node": bias_swot_node,
        "b_reach": bias_swot_reach,
        "rmse_node": rmse_swot_node,
        "rmse_reach": rmse_swot_reach,
        "mae_node": mae_swot_node,
        "mae_reach": mae_swot_reach,
        "good_points_N": good_points_N,
        "suspect_points_N": suspect_points_N,
        "degraded_points_N": degraded_points_N,
        "points_reach": points_reach
        }

if __name__ == "__main__":

    # Record the starting point
    start_time = time.perf_counter()

    # Path to your stations shapefile (.shp)
    shapefile_path = "./shapefile/selected_3Nodes.shp"

    # Load shapefile
    gdf = gpd.read_file(shapefile_path)
    
    tasks = [(idx, row) for idx, row in gdf.iterrows()]
    with mp.Pool(processes=mp.cpu_count(), initializer=init_worker) as pool:
        results = pool.map(process_station, tasks) 

    ### debugging with a subset of the data
    #results = [process_station(task) for task in tasks if task[1]['STATION_NU'] == '08EB005']
    ###

    for r in results:
        if r is None:
            continue

        idx = r["index"]

        gdf.at[idx, "r_N"] = r["r_node"]
        gdf.at[idx, "r_R"] = r["r_reach"]
        gdf.at[idx, "p_N"] = r["p_node"]
        gdf.at[idx, "p_R"] = r["p_reach"]
        gdf.at[idx, "sp_N"] = r["spearman_node"]
        gdf.at[idx, "sp_R"] = r["spearman_reach"]
        gdf.at[idx, "ps_N"] = r["ps_node"]
        gdf.at[idx, "ps_R"] = r["ps_reach"]
        gdf.at[idx, "b_N"] = r["b_node"]
        gdf.at[idx, "b_R"] = r["b_reach"]
        gdf.at[idx, "rmse_N"] = r["rmse_node"]
        gdf.at[idx, "rmse_R"] = r["rmse_reach"]
        gdf.at[idx, "mae_N"] = r["mae_node"]
        gdf.at[idx, "mae_R"] = r["mae_reach"]
        gdf.at[idx, "good_N"] = r["good_points_N"]
        gdf.at[idx, "susp_N"] = r["suspect_points_N"]
        gdf.at[idx, "degr_N"] = r["degraded_points_N"]
        gdf.at[idx, "data_R"] = r["points_reach"]

    # Save updated gdf to new shapefile
    gdf.to_file("./journal/shapefiles/selected_Journal_Sim12.shp")

    # Record the ending point
    end_time = time.perf_counter()

    # Calculate total execution runtime
    execution_time = end_time - start_time
    print(f"Script finished in {execution_time/60:.2f} minutes")

