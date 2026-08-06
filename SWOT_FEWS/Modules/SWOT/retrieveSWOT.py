import logging
import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from io import StringIO

quality_flag = 2    # Only use data with quality flag less than 2

def write_diagnostics_xml(diagnostics, file_path):
    # Root element with namespaces
    ns = {
        "": "http://www.wldelft.nl/fews/PI",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }
    ET.register_namespace("", ns[""])
    ET.register_namespace("xsi", ns["xsi"])

    root = ET.Element("Diag", {
        "version": "1.2",
        "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation":
            "http://www.wldelft.nl/fews/PI http://fews.wldelft.nl/schemas/version1.0/pi-schemas/pi_diag.xsd"
    })

    for diag in diagnostics:
        line = ET.SubElement(root, "line")
        line.set("level", str(diag["level"]))
        line.set("description", diag["description"])
        if "eventCode" in diag:
            line.set("eventCode", diag["eventCode"])

    # Write to file
    tree = ET.ElementTree(root)
    tree.write(file_path, encoding="utf-8", xml_declaration=True)    


def read_swot_node(sword_id, write=False, csv_dir="./csv_data"):
    logger = logging.getLogger("worker")
    logger.info(f"Reading SWOT node {sword_id}")
    parameters = (
        f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?"
        f"feature=Node&feature_id={sword_id}"
        "&output=csv"
        "&collection_name=SWOT_L2_HR_RiverSP_D"
        "&start_time=2022-01-01T00:00:00Z"
        "&end_time=2027-01-01T00:00:00Z"
        "&fields=node_id,time_str,wse,node_q,width,geoid_hght,area_wse,n_good_pix,sword_version"
        )

    try:
        hydrocron_response = requests.get(parameters).json()
        if hydrocron_response.get('error'):
            logger.error(hydrocron_response['error'])
            return None
    except:
        logger.error(f"Error retrieving SWOT Node data for SWORD ID: {sword_id}")
        return None
    csv_str = hydrocron_response['results']['csv']

    # Read data from url and process it according to date format
    data_swot = pd.read_csv(StringIO(csv_str))
    data_swot = data_swot[data_swot['time_str'] != 'no_data']
    data_swot.time_str = pd.to_datetime(data_swot.time_str, format='%Y-%m-%dT%H:%M:%SZ')
    data_swot.sort_values(by='time_str', inplace = True)
    ind = data_swot.node_q < quality_flag
    data_swot_filter = data_swot[ind]

    # Filter out invalid WSE values
    data_swot_valid = data_swot_filter[data_swot_filter['wse'] != -999999999999]

    if write == True:
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        data_swot_valid.to_csv(os.path.join(csv_dir, f'data_SWOT_Node_{sword_id}.csv'), index=False)

    if data_swot_valid.empty:
        logger.debug(f"No data retrieved from SWOT Node: {sword_id}")

    return data_swot_valid

def read_swot_reach(sword_id, write=False, csv_dir="./csv_data"):
    logger = logging.getLogger("worker")
    logger.info(f"Reading SWOT reach {sword_id}")
    parameters = (
        f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?"
        f"feature=Reach&feature_id={sword_id}"
        "&output=csv"
        "&collection_name=SWOT_L2_HR_RiverSP_D"
        "&start_time=2022-01-01T00:00:00Z"
        "&end_time=2027-01-01T00:00:00Z"
        "&fields=reach_id,time_str,wse,reach_q,width,geoid_hght,area_wse,n_good_nod,sword_version"
        )

    try:
        hydrocron_response = requests.get(parameters).json()
        if hydrocron_response.get('error'):
            logger.error(hydrocron_response['error'])
            return None
    except:
        logger.error(f"Error retrieving SWOT Reach data for SWORD ID: {sword_id}")
        return None
    csv_str = hydrocron_response['results']['csv']

    # Read data from url and process it according to date format
    data_swot = pd.read_csv(StringIO(csv_str))
    data_swot = data_swot[data_swot['time_str'] != 'no_data']
    data_swot.time_str = pd.to_datetime(data_swot.time_str, format='%Y-%m-%dT%H:%M:%SZ')
    data_swot.sort_values(by='time_str', inplace = True)
    ind = data_swot.reach_q < quality_flag
    data_swot_filter = data_swot[ind]

    # Filter out invalid WSE values
    data_swot_valid = data_swot_filter[data_swot_filter['wse'] != -999999999999]

    if write == True:
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        data_swot_valid.to_csv(os.path.join(csv_dir, f'data_SWOT_Reach_{sword_id}.csv'), index=False)

    if data_swot_valid.empty:
        logger.debug(f"No data retrieved from SWOT Reach: {sword_id}")

    return data_swot_valid

def read_swot_lake(lake_id, write=False, csv_dir="./csv_data"):
    logger = logging.getLogger("worker")
    logger.info(f"Reading SWOT Lake {lake_id}")
    parameters = (
        f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?"
        f"feature=PriorLake&feature_id={lake_id}"
        "&output=csv"
        "&start_time=2022-01-01T00:00:00Z"
        "&end_time=2027-01-01T00:00:00Z"
        "&fields=lake_id,time_str,wse,geometry,quality_f,collection_shortname,PLD_version,range_start_time"
        )

    try:
        hydrocron_response = requests.get(parameters).json()
        if hydrocron_response.get('error'):
            logger.error(hydrocron_response['error'])
            return None
    except:
        logger.error(f"Error retrieving SWOT Lake data for Lake ID: {lake_id}")
        return None
    csv_str = hydrocron_response['results']['csv']

    # Read data from url and process it according to date format
    data_swot = pd.read_csv(StringIO(csv_str))
    data_swot = data_swot[data_swot['time_str'] != 'no_data']
    data_swot.time_str = pd.to_datetime(data_swot.time_str, format='%Y-%m-%dT%H:%M:%SZ')
    data_swot.sort_values(by='time_str', inplace = True)
    ind = data_swot.reach_q < quality_flag
    data_swot_filter = data_swot[ind]

    # Filter out invalid WSE values
    data_swot_valid = data_swot_filter[data_swot_filter['wse'] != -999999999999]

    if write == True:
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        data_swot_valid.to_csv(os.path.join(csv_dir, f'data_SWOT_Lake_{lake_id}.csv'), index=False)

    if data_swot_valid.empty:
        logger.debug(f"No data retrieved from SWOT Lake: {lake_id}")

    return data_swot_valid

def read_args():
    import argparse
    parser = argparse.ArgumentParser(description="Read SWOT data for a given SWORD ID and save it to a CSV file.")
    parser.add_argument("--feature", type=str, required=True, help="The feature type (node or reach) to read.")
    parser.add_argument("--sword_id", type=int, required=True, help="The SWORD ID of the SWOT node or reach to read.")
    parser.add_argument("--write", action="store_true", help="Whether to write the data to a CSV file.")
    parser.add_argument("--csv_dir", type=str, default="./csv_data", help="The directory to save the CSV file.")
    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    diagnostics = []
    args = read_args()
    if args["feature"] == "node":
        read_swot_node(args["sword_id"], write=args["write"], csv_dir=args["csv_dir"])
    elif args["feature"] == "reach":
        read_swot_reach(args["sword_id"], write=args["write"], csv_dir=args["csv_dir"])
    elif args["feature"] == "lake":
        read_swot_lake(args["sword_id"], write=args["write"], csv_dir=args["csv_dir"])
    else:
        diagnostics.append({
            "level": 2,
            "description": f"Invalid feature type: {args['feature']}. Please choose 'node', 'reach', or 'lake'."
        })
    write_diagnostics_xml(diagnostics, "diagnostics.xml")
