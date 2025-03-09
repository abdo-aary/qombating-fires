import cdsapi
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir,'..', '..', '..', 'storage', 'dataset','wildfires','precipitation.grib')
dataset_path = os.path.abspath(dataset_path)

dataset = "reanalysis-era5-single-levels"
key = '684ab03f-dd8e-4c19-a669-b465462ef088'
if key == 'your_key' :
    print("You need to replace the variable \'key\' with your API token on the website: https://cds.climate.copernicus.eu")
    sys.exit()

request = {
    "product_type": ["reanalysis"],
    "variable": [
        "total_precipitation"
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
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [63, -80, 45, -57]
}

client = cdsapi.Client(url="https://cds.climate.copernicus.eu/api/",key = key)
client.retrieve(dataset, request,str(dataset_path))
