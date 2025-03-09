import cdsapi
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir,'..', '..', '..', 'storage', 'dataset','wildfires','weather.grib')
dataset_path = os.path.abspath(dataset_path)

dataset = "reanalysis-era5-single-levels"
key = '684ab03f-dd8e-4c19-a669-b465462ef088'
if key == 'your_key' :
    print("You need to replace the variable \'key\' with your API token on the website: https://cds.climate.copernicus.eu")
    sys.exit()



request = {
    "product_type": ["reanalysis"],
    "variable": [
        "2m_dewpoint_temperature",
        "2m_temperature",
        "soil_temperature_level_1",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_pressure",
        "leaf_area_index_high_vegetation",
        "leaf_area_index_low_vegetation",
        "high_vegetation_cover"
    ],
    "year": [
        "2010", "2011", "2012",
        "2013", "2014", "2015",
        "2016", "2017", "2018",
        "2019", "2020", "2021"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": ["12:00"],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [63, -80, 45, -57]
}


client = cdsapi.Client(url="https://cds.climate.copernicus.eu/api/",key = key)
client.retrieve(dataset, request,str(dataset_path))
