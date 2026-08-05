from io import StringIO
import os
from venv import logger
from owslib.ogcapi.features import Features
import pandas as pd
import geopandas as gpd
import requests
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import interp1d
import multiprocessing as mp
import logging
import time
from datetime import datetime

# Convert date string to proper format for retrieval routines
def convert_date(date_str, in_fmt, out_fmt):
    return datetime.strptime(date_str, in_fmt).strftime(out_fmt)

# Read SWOT nodes
def retrieve_swot_node(sword_id, start_date, end_date, write=False):
    start_date = convert_date(start_date, '%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ')
    end_date = convert_date(end_date, '%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ')

    parameters = (
        f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?"
        f"feature=Node&feature_id={sword_id}"
        "&output=csv"
        "&collection_name=SWOT_L2_HR_RiverSP_D"
        f"&start_time={start_date}" #2022-07-01T00:00:00Z
        f"&end_time={end_date}"
        "&fields=node_id,time_str,wse,width,node_q,node_q_b,geoid_hght,xtrk_dist,node_dist,ice_clim_f,ice_dyn_f,area_wse,n_good_pix,wse_sm,sword_version"
        )

    try:
        hydrocron_response = requests.get(parameters, timeout=60).json()
    except:
        return None

    if hydrocron_response.get('error'):
        print(hydrocron_response['error'])
        return None

    if hydrocron_response.get('results') is None or hydrocron_response['results'].get('csv') is None:
        print(f"No CSV data in response for SWOT Node: {sword_id}")
        print(f"Response content: {hydrocron_response}")
        return None
    
    print(f"Successfully retrieved SWOT Node data for SWORD ID: {sword_id}")
    csv_str = hydrocron_response['results']['csv']

    # Read data from url and process it according to date format
    data_swot = pd.read_csv(StringIO(csv_str))
    data_swot = data_swot[data_swot['time_str'] != 'no_data']
    data_swot.time_str = pd.to_datetime(data_swot.time_str, format='%Y-%m-%dT%H:%M:%SZ')
    data_swot.sort_values(by='time_str', inplace = True)

    if write:
        os.makedirs("./csv_data/", exist_ok=True)
        data_swot.to_csv('./csv_data/data_SWOT_Node_' + str(sword_id) + '.csv')

    if data_swot.empty:
        print(f"No data retrieved from SWOT Node: {sword_id}")

    return data_swot

# Read SWOT reaches
def retrieve_swot_reach(sword_id, start_date, end_date, logger, write=False):
    start_date = convert_date(start_date, '%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ')
    end_date = convert_date(end_date, '%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ')

    parameters = (
        f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?"
        f"feature=Reach&feature_id={sword_id}"
        "&output=csv"
        "&collection_name=SWOT_L2_HR_RiverSP_D"
        f"&start_time={start_date}"
        f"&end_time={end_date}"
        "&fields=reach_id,time_str,wse,wse_c,width,width_c,slope,slope2,area_wse,xtrk_dist,loc_offset,node_dist,ice_clim_f,ice_dyn_f,reach_q,reach_q_b,obs_frac_n,geoid_hght,sword_version"
    )
    try:
        hydrocron_response = requests.get(parameters, timeout=60).json()
    except:
        logger.error(f"Error retrieving SWOT Reach data for SWORD ID: {sword_id}")
        return None

    if hydrocron_response.get('error'):
        logger.info(hydrocron_response['error'])
        return None

    if hydrocron_response.get('results') is None or hydrocron_response['results'].get('csv') is None:
        logger.info(f"No CSV data in response for SWOT Reach: {sword_id}")
        logger.info(f"Response content: {hydrocron_response}")
        return None
    logger.info(f"Successfully retrieved SWOT Reach data for SWORD ID: {sword_id}")
    csv_str = hydrocron_response['results']['csv']

    # Read data from url and process it according to date format
    data_swot = pd.read_csv(StringIO(csv_str))
    data_swot = data_swot[data_swot['time_str'] != 'no_data']
    data_swot.time_str = pd.to_datetime(data_swot.time_str, format='%Y-%m-%dT%H:%M:%SZ')
    data_swot.sort_values(by='time_str', inplace = True)

    if write:
        os.makedirs("./csv_data/", exist_ok=True)
        data_swot.to_csv('./csv_data/data_SWOT_Reach_' + str(sword_id) + '.csv')

    if data_swot.empty:
        logger.info(f"No data retrieved from SWOT Reach: {sword_id}")

    return data_swot

# Read WSC real-time data
def retrieve_wsc_real_time(station, start_date, end_date, logger, write=False):
    datecolumn = 'Date' # Date column to read date from wsc url service

    start_date = convert_date(start_date, '%Y-%m-%d', '%Y-%m-%d %H:%M:%S')
    end_date = convert_date(end_date, '%Y-%m-%d', '%Y-%m-%d %H:%M:%S')

    params = {
        'stations[]': station,
        'start_date': start_date,
        'end_date': end_date,
        'parameters[]': [46, 47]
    }

    try:
        response = requests.get(
            'https://wateroffice.ec.gc.ca/services/real_time_data/csv/inline',
            params=params,
            timeout=60
        )
    except:
        logger.error(f"Error retrieving WSC real time data for station: {station}")
        return None

	# Read data from url and process it according to date format
    try:
        data_wsc = pd.read_csv(StringIO(response.text))
    except pd.errors.EmptyDataError:
        logger.debug(f"No data returned for real time WSC station: {station}")
        return None
        
    data_wsc[datecolumn] = pd.to_datetime(data_wsc[datecolumn])
    data_wsc.sort_values(by=datecolumn, inplace = True)

    if write:
        os.makedirs("./csv_data/", exist_ok=True)
        data_wsc.to_csv('./csv_data/data_realtime_WSC_' + station + '.csv')

    if data_wsc.empty:
        logger.info(f"No data for WSC Real-Time station: {station}")

    return data_wsc

# Read WSC historical daily data
def retrieve_wsc_historical(station, start_date, end_date, logger, write=False):
    datecolumn = 'Date' # Date column to read date from wsc url service

    params = {
        'stations[]': station,
        'start_date': start_date,
        'end_date': end_date,
        'parameters[]': ['level', 'flow']
    }

    try:
        response = requests.get(
            'https://wateroffice.ec.gc.ca/services/daily_data/csv/inline',
            params=params,
            timeout=60
        )
    except:
        logger.error(f"Error retrieving WSC real time data for station: {station}")
        return None

	# Read data from url and process it according to date format
    try:
        data_wsc = pd.read_csv(StringIO(response.text))
    except pd.errors.EmptyDataError:
        logger.debug(f"No data returned for real time WSC station: {station}")
        return None
    data_wsc[datecolumn] = pd.to_datetime(data_wsc[datecolumn])
    data_wsc.sort_values(by=datecolumn, inplace = True)

    if write:
        os.makedirs("./csv_data/", exist_ok=True)
        data_wsc.to_csv('./csv_data/data_hist_WSC_' + station + '.csv')

    if data_wsc.empty:
        logger.info(f"No data for WSC Historical station: {station}")

    return data_wsc

# Read WSC from API
def retrieve_wsc_from_api(station, start_date, end_date, logger, write=False):
    limit=10000
    api_url = 'https://api.weather.gc.ca/openapi?f=json'
    collection = 'hydrometric-daily-mean'

    # Set the time limits for the data retrieval
    time_ = f"{start_date}/{end_date}"

    # Set columns to be saved to the dataframe
    query_variables = [
        "DATE",
        "STATION_NUMBER",
        "DISCHARGE",
		"LEVEL"
        ]

    # Instansiate features
    oafeat = Features(api_url)
    
    # Data retrieval and creation of the data frames
    hydro_data = oafeat.collection_items(
        collection,
        limit=limit,
        STATION_NUMBER=station,
        datetime=time_,
    )

    # Creation of a data frame if there is data for the chosen time period
    if hydro_data["features"]:
        ## Creation of a dictionary in a format compatible with Pandas
        historical_data_format = [
            {
                "LATITUDE": el["geometry"]["coordinates"][1],
                "LONGITUDE": el["geometry"]["coordinates"][0],
                **el["properties"],
            }
            for el in hydro_data["features"]
        ]

        ## Creation of the data frame
        historical_data_df = pd.DataFrame(
            historical_data_format,
            columns=query_variables,
        )
        
        ## Detect and convert data types of columns
        historical_data_df = historical_data_df.infer_objects(copy=False)

        # Creating an index with the date in a datetime format
        historical_data_df['DATE'] = pd.to_datetime(
            historical_data_df['DATE']
        )
        #historical_data_df.set_index([datetime_column], inplace=True, drop=True)
        
        historical_data_df = historical_data_df.sort_values(by='DATE')

        if write:
            os.makedirs("./csv_data/", exist_ok=True)
            historical_data_df.to_csv('./csv_data/data_hist_API_WSC_' + station + '.csv')

        return historical_data_df
    else:
        logger.info(f"No data for WSC API station: {station}")
        return None

def retrieve_wsc_realtime_from_api(station, start_date, end_date, logger, write=False):
    limit=10000
    api_url = 'https://api.weather.gc.ca/openapi?f=json'
    collection = 'hydrometric-realtime'

    # Set the time limits for the data retrieval
    time_ = f"{start_date}/{end_date}"

    # Set columns to be saved to the dataframe
    query_variables = [
        "DATETIME",
        "STATION_NUMBER",
        "DISCHARGE",
		"LEVEL"
        ]

    # Instansiate features
    oafeat = Features(api_url)
    
    # Data retrieval and creation of the data frames
    hydro_data = oafeat.collection_items(
        collection,
        limit=limit,
        STATION_NUMBER=station,
        datetime=time_,
    )

    # Creation of a data frame if there is data for the chosen time period
    if hydro_data["features"]:
        ## Creation of a dictionary in a format compatible with Pandas
        historical_data_format = [
            {
                "LATITUDE": el["geometry"]["coordinates"][1],
                "LONGITUDE": el["geometry"]["coordinates"][0],
                **el["properties"],
            }
            for el in hydro_data["features"]
        ]

        ## Creation of the data frame
        historical_data_df = pd.DataFrame(
            historical_data_format,
            columns=query_variables,
        )
        
        ## Detect and convert data types of columns
        historical_data_df = historical_data_df.infer_objects(copy=False)

        # Creating an index with the date in a datetime format
        historical_data_df['DATETIME'] = pd.to_datetime(
            historical_data_df['DATETIME']
        )
        #historical_data_df.set_index([datetime_column], inplace=True, drop=True)
        
        historical_data_df = historical_data_df.sort_values(by='DATETIME')

        if write:
            os.makedirs("./csv_data/", exist_ok=True)
            historical_data_df.to_csv('./csv_data/data_realtime_API_WSC_' + station + '.csv')

        return historical_data_df
    else:
        logger.info(f"No data for WSC API station: {station}")
        return None

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

def process_station(args):
    # Set up the date range
    start_date = '2023-07-01'
    end_date = '2027-01-01' 

    logger = logging.getLogger("diagnostics")

    el, row = args
    station = row["STATION_NU"]
    sword_node_id = row["node_id"]
    sword_node_id = row["node_id"]
    sword_node_id_2 = row["node_id_2"]
    sword_node_id_3 = row["node_id_3"]
    sword_reach_id = row["reach_id"]

    logger.info(f"Index in gdf: {el}")
    logger.info(f"Station: {station}")
    logger.info(f"SWOT Node ID: {sword_node_id}")
    logger.info(f"SWOT Reach ID: {sword_reach_id}")

    # Retrieve data
    #retrieve_swot_node(sword_node_id, start_date, end_date, logger, write=True)
    retrieve_swot_node(sword_node_id_2, start_date, end_date, logger, write=True)
    retrieve_swot_node(sword_node_id_3, start_date, end_date, logger, write=True)
    #retrieve_swot_reach(sword_reach_id, start_date, end_date, logger, write=True)
    #retrieve_wsc_real_time(station, start_date, end_date, logger, write=True)
    #retrieve_wsc_historical(station, start_date, end_date, logger, write=True)
    #retrieve_wsc_from_api(station, start_date, end_date, logger, write=True)
    #retrieve_wsc_realtime_from_api(station, start_date, end_date, logger, write=True)


start_date = '2026-07-01'
end_date = '2026-08-01'
retrieve_swot_node(71290000310521, start_date, end_date, write=True)
