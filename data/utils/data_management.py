import pandas as pd
import numpy as np

def vectorized_haversine_formula(latitudes1, longitudes1, latitudes2, longitudes2):
    lat1, lon1 = np.radians(latitudes1), np.radians(longitudes1)
    lat2, lon2 = np.radians(latitudes2), np.radians(longitudes2)
    d_haversine = 2 * 6378 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2 +
                                               np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))
    return d_haversine



def vectorized_cell_area(latitudes, longitudes, incr_lat, incr_lon):
    # Calculate the vertex of the cell
    origine_lat = latitudes - incr_lat / 2
    origine_lon = longitudes - incr_lon / 2
    point1_lat = latitudes + incr_lat / 2
    point2_lon = longitudes + incr_lon / 2

    # Calculate distance between the vertex and multiply the distance to get the area
    largeur = vectorized_haversine_formula(origine_lat, origine_lon, point1_lat, origine_lon)
    longueur = vectorized_haversine_formula(origine_lat, origine_lon, origine_lat, point2_lon)
    area = largeur * longueur #km2
    return area*100 #HA
def add_coo_area(dataframe_weather, dataframe_wildfire):
    # Min of latitude and longitude
    lat_min = dataframe_weather["LATITUDE"].min()
    lon_min = dataframe_weather["LONGITUDE"].min()

    # get increment
    first_row = dataframe_weather.iloc[0]
    first_lat = first_row["LATITUDE"]
    first_lon = first_row["LONGITUDE"]
    second_lat = dataframe_weather[dataframe_weather["LATITUDE"] != first_lat].iloc[0]["LATITUDE"]
    second_lon = dataframe_weather[dataframe_weather["LONGITUDE"] != first_lon].iloc[0]["LONGITUDE"]

    incr_lat = abs(second_lat - first_lat)
    incr_lon = abs(second_lon - first_lon)

    # calculate coordinate system
    dataframe_weather[["COORDINATES_LAT", "COORDINATES_LON"]] = ((dataframe_weather[["LATITUDE", "LONGITUDE"]] -
                                                                  [lat_min,lon_min]) / [incr_lat, incr_lon]).astype(int)
    # Calculate the area of the cell
    dataframe_weather["AREA_HA"] = vectorized_cell_area(
        dataframe_weather["LATITUDE"], dataframe_weather["LONGITUDE"], incr_lat, incr_lon
    )

    # calculate coordinate system
    dataframe_wildfire[["COORDINATES_LAT", "COORDINATES_LON"]] = (round((dataframe_wildfire[["LATITUDE", "LONGITUDE"]]
                                                            - [lat_min,lon_min]) / [incr_lat, incr_lon])).astype(int)

    dataframe_wildfire[["CELL_LAT", "CELL_LON"]] = (round((dataframe_wildfire[["LATITUDE", "LONGITUDE"]] / [incr_lat, incr_lon]))*[incr_lat, incr_lon])
    dataframe_wildfire["AREA_HA"] = vectorized_cell_area(
       dataframe_wildfire["CELL_LAT"], dataframe_wildfire["CELL_LON"], incr_lat, incr_lon
     )

    return dataframe_weather, dataframe_wildfire, incr_lat, incr_lon

def aggregate_fire(dataframe_wildfire):
    # regroup the data
    dataframe_wildfire = dataframe_wildfire.groupby(
        ["REP_DATE", "COORDINATES_LAT", "COORDINATES_LON"], as_index=False
    ).agg({
        **{col: "first" for col in dataframe_wildfire.columns if col not in ["SIZE_HA", "CAUSE", "FID"]},
        "SIZE_HA": "sum",
        "CAUSE": lambda x: x.iloc[0] if x.nunique() == 1 else "U",
        "FID": "min"
    })
    return dataframe_wildfire
def get_cells_chebyshev_range(x, y, N):
    cells = []
    for dx in range(-N, N + 1):
        for dy in range(-N, N + 1):
            if max(abs(dx), abs(dy)) == N:
                cells.append((x + dx, y + dy))
    return cells
def excess_fire_distribution(dataframe_wildfire,incr_lat,incr_lon):

    new_fid = dataframe_wildfire["FID"].max()+1
    excess_row = dataframe_wildfire[dataframe_wildfire["SIZE_HA"] > dataframe_wildfire["AREA_HA"]]
    for index, row in excess_row.iterrows():
        date = row["REP_DATE"]
        same_date_cells = dataframe_wildfire[dataframe_wildfire["REP_DATE"] == date]
        dist = 1
        #repartition of burned area among closest cells if the burned area is greater than the actual area due to fire aggregation
        while row["SIZE_HA"] > row["AREA_HA"]:
            neighbors = get_cells_chebyshev_range(row["COORDINATES_LAT"], row["COORDINATES_LON"], dist)

            for cell_neigh in neighbors:
                if row["SIZE_HA"] > row["AREA_HA"] :
                    cell_to_change = same_date_cells[(same_date_cells["COORDINATES_LAT"] == cell_neigh[0]) &
                                                     (same_date_cells["COORDINATES_LON"] == cell_neigh[1])]
                    if not cell_to_change.empty:

                        if cell_to_change.iloc[0]["AREA_HA"] > cell_to_change.iloc[0]["SIZE_HA"]:
                            if row["SIZE_HA"] - row["AREA_HA"] > cell_to_change.iloc[0]["AREA_HA"] - cell_to_change.iloc[0]["SIZE_HA"]:

                                row["SIZE_HA"] -= cell_to_change.iloc[0]["AREA_HA"] - cell_to_change.iloc[0]["SIZE_HA"]

                                dataframe_wildfire.loc[cell_to_change.index[0], "SIZE_HA"] = cell_to_change.iloc[0]["AREA_HA"]


                            else :
                                dataframe_wildfire.loc[cell_to_change.index[0], "SIZE_HA"] += row["SIZE_HA"]-row["AREA_HA"]
                                row["SIZE_HA"] = row["AREA_HA"]
                    else:
                        new_cell = {"LATITUDE" : 0 , "LONGITUDE" : 0 , "REP_DATE" : date ,
                        'PROTZONE':  row["PROTZONE"], "ECOZ_NAME" : row["ECOZ_NAME"],
                        'COORDINATES_LAT' : cell_neigh[0],'COORDINATES_LON' : cell_neigh[1],
                        'CELL_LAT' : row["CELL_LAT"]+(row["COORDINATES_LAT"]-cell_neigh[0])*incr_lat,
                        'CELL_LON' : row["CELL_LON"]+(row["COORDINATES_LON"]-cell_neigh[1])*incr_lon,
                        "AREA_HA" :0,'SIZE_HA' : 0 , 'CAUSE' : row["CAUSE"],"FID" : new_fid
                        }
                        new_cell["AREA_HA"] = vectorized_cell_area(new_cell["CELL_LAT"], new_cell["CELL_LON"],incr_lat,incr_lon)
                        if row["SIZE_HA"] - new_cell["AREA_HA"] < row["AREA_HA"]:
                            new_cell["SIZE_HA"] = row["SIZE_HA"] - row["AREA_HA"]
                            row["SIZE_HA"] = row["AREA_HA"]
                        else :
                            row["SIZE_HA"] -= new_cell["AREA_HA"]
                            new_cell["SIZE_HA"] =  new_cell["AREA_HA"]
                        dataframe_wildfire.loc[len(dataframe_wildfire)] = new_cell
                        new_fid+=1
                else :
                    break
            dist+=1
        dataframe_wildfire.loc[index, :] = row

    return dataframe_wildfire
def add_burned_density_isfire(dataframe_wildfire):
    dataframe_wildfire["BURNED_DENSITY"] = dataframe_wildfire["SIZE_HA"]/dataframe_wildfire["AREA_HA"]
    dataframe_wildfire["IS_FIRE"]=1
    return dataframe_wildfire
def add_rh_vpd(dataframe_weather):
    #kelvin to celcius
    t_c = dataframe_weather["T2M"] - 273.15
    td_c = dataframe_weather["D2M"] - 273.15

    # computation of  saturation vapor pressure (e_s) and real (e_d) in hPa with Tentens equation
    e_s = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e_d = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))

    # relative humidity (%)
    dataframe_weather['RH'] = 100 * (e_d / e_s)

    # Vapour-pressure deficit (VPD in hPa)
    dataframe_weather['VPD'] = e_s - e_d

    return dataframe_weather

def merge_fire_weather(dataframe_wildfire,dataframe_weather):
    dataframe_wildfire.rename(columns={'REP_DATE': 'DATE'}, inplace=True)
    dataframe_weather.rename(columns={'TIME': 'DATE',"LATITUDE":"CELL_LAT","LONGITUDE":"CELL_LON"}, inplace=True)
    dataframe_wildfire = dataframe_wildfire.drop(columns=['LATITUDE','LONGITUDE'])
    dataframe_weather = dataframe_weather.drop(columns=['VALID_TIME'])
    dataframe_wildfire['DATE'] = pd.to_datetime(dataframe_wildfire['DATE'])
    dataframe_weather['DATE'] = pd.to_datetime(dataframe_weather['DATE'])
    # merge on the date
    df_final = dataframe_weather.merge(dataframe_wildfire, on=['DATE', 'COORDINATES_LAT', 'COORDINATES_LON',"CELL_LAT","CELL_LON","AREA_HA"], how='left')

    # Replace Nan by 0
    df_final.fillna({'SIZE_HA': 0, 'BURNED_DENSITY': 0, 'PROTZONE': 'Inconnu', 'ECOZ_NAME' : 'Inconnu',
                     'CAUSE':'N',"FID" : 0,"IS_FIRE":0}, inplace=True)

    return df_final


def vectorized_cell_width_length(latitudes, longitudes, incr_lat, incr_lon):
    origine_lat = latitudes - incr_lat / 2
    origine_lon = longitudes - incr_lon / 2
    point1_lat = latitudes + incr_lat / 2
    point2_lon = longitudes + incr_lon / 2

    largeur = round(vectorized_haversine_formula(origine_lat, origine_lon, point1_lat, origine_lon), 3)
    longueur = round(vectorized_haversine_formula(origine_lat, origine_lon, origine_lat, point2_lon), 3)
    return np.column_stack((largeur, longueur))