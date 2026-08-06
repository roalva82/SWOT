import logging
import pandas as pd
from owslib.ogcapi.features import Features
import xml.etree.ElementTree as ET


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

def read_wsc_from_api(station, time_limits=True, start_date='2022-01-01', end_date='..', limit=10000, write=True):
    logger = logging.getLogger("worker")
    logger.info(f"Reading API data from station {station}")
    api_url = 'https://api.weather.gc.ca/openapi?f=json'
    collection = 'hydrometric-daily-mean'

    # Set the time limits for the data retrieval
    if time_limits:
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
    if time_limits:
        hydro_data = oafeat.collection_items(
            collection,
            limit=limit,
            STATION_NUMBER=station,
            datetime=time_,
        )
    else:
        hydro_data = oafeat.collection_items(
            collection,
            limit=limit,
            STATION_NUMBER=station,
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
        
        historical_data_df = historical_data_df.sort_values(by='DATE')

        if historical_data_df.empty:
            logger.debug(f"Empty data for WSC API station: {station}")
            return None
        else:
            if write == True:
                historical_data_df.to_csv('./csv_data/data_hist_API_WSC_' + station + '.csv',index=False)
            return historical_data_df
    else:
        logger.debug(f"No data for WSC API station: {station}")
        return None

def read_wsc_historical(station, write=True):
    logger = logging.getLogger("worker")
    logger.info(f"Reading historical data from station {station}")

    # Parameter level corresponds to water level / flow for discharge (unit values)
    request_url = 'https://wateroffice.ec.gc.ca/services/daily_data/csv/inline?stations[]=' \
        + station + \
        '&parameters[]=level&parameters[]=flow&start_date=2022-01-01&end_date=2027-01-01'

    # Read data from url and process it according to date format
    dateformat = 'Date' # Format to read date from wsc url service
    data_wsc = pd.read_csv(request_url)
    data_wsc[dateformat] = pd.to_datetime(data_wsc[dateformat])
    data_wsc.sort_values(by=dateformat, inplace = True)

    data_wsc.rename(columns={"Parameter/Paramètre": "Parameter"}, inplace=True)
    data_wsc["Parameter"] = data_wsc["Parameter"].replace("discharge/débit","Q.obs")
    data_wsc["Parameter"] = data_wsc["Parameter"].replace("water level/niveau","H.obs")

    if write == True:
        data_wsc.to_csv('./csv_data/data_hist_WSC_' + station + '.csv',index=False)

    if data_wsc.empty:
        logger.debug(f"No data for WSC Historical station: {station}")

    return data_wsc


# Read WSC real-time data
def read_wsc_real_time(station, write=True):
    logger = logging.getLogger("worker")
    logger.info(f"Reading real-time data from station {station}")
    # Parameter 46 corresponds to water level / 47 for discharge (unit values) - 3 and 6 for daily values respectively
    request_url = 'https://wateroffice.ec.gc.ca/services/real_time_data/csv/inline?stations[]=' \
        + station + \
        '&parameters[]=46&parameters[]=47&start_date=2022-01-01%2000:00:00&end_date=2027-01-01%2000:00:00'

	# Read data from url and process it according to date format
    dateformat = 'Date' # Format to read date from wsc url service
    data_wsc = pd.read_csv(request_url)
    data_wsc[dateformat] = pd.to_datetime(data_wsc[dateformat])
    data_wsc.sort_values(by=dateformat, inplace = True)

    data_wsc.rename(columns={"Parameter/Paramètre": "Parameter"}, inplace=True)
    data_wsc["Parameter"] = data_wsc["Parameter"].replace(47,"Q.obs")
    data_wsc["Parameter"] = data_wsc["Parameter"].replace(46,"H.obs")

    if write == True:
        data_wsc.to_csv('./csv_data/data_realtime_WSC_' + station + '.csv', index=False)    

    if data_wsc.empty:
        logger.debug(f"No data for WSC Real-Time station: {station}")

    return data_wsc


def read_args():
    import argparse
    parser = argparse.ArgumentParser(description="Retrieve WSC API data")
    parser.add_argument("--station", type=str, required=True, help="WSC station number")
    parser.add_argument("--start_date", type=str, default='2022-01-01', help="Start date for data retrieval (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default='..', help="End date for data retrieval (YYYY-MM-DD or '..' for no limit)")
    parser.add_argument("--limit", type=int, default=10000, help="Maximum number of records to retrieve")
    parser.add_argument("--write", action='store_true', help="Whether to write the retrieved data to a CSV file")
    return parser.parse_args()

if __name__ == "__main__":
    diagnostics = []
    args = read_args()
    read_wsc_from_api(
        station=args.station,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        write=args.write
    )

    read_wsc_historical(
        station=args.station,
        write=args.write
    )

    read_wsc_real_time(
        station=args.station,
        write=args.write
    )

    write_diagnostics_xml(diagnostics, "diagnostics.xml")

