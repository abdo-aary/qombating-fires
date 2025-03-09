import cfgrib
import os
import xarray as xr
from data.utils.data_management import *
import gc
current_dir = os.path.dirname(os.path.abspath(__file__))
dfweather_path = os.path.join(current_dir, '..', '..', 'storage', 'dataset','weather.grib')
dfwildfire_path = os.path.join(current_dir, '..', '..', 'storage', 'dataset','CANADA_WILDFIRES.csv')
generate_path = os.path.join(current_dir, '..', '..', 'storage', 'dataset',"dataset_pre_analysis.csv")
generate_fire_path = os.path.join(current_dir, '..', '..', 'storage', 'dataset',"dataset_fire_cleaned.csv")
precip_file = os.path.join(current_dir, '..', '..', 'storage', 'dataset',"precipitation.grib")
dfweather_path  = os.path.abspath(dfweather_path)
dfwildfire_path = os.path.abspath(dfwildfire_path)

with cfgrib.open_dataset(dfweather_path) as ds:
    df_weather = ds.to_dataframe().reset_index()

del ds
gc.collect()

df_precip = xr.open_dataset(precip_file, engine="cfgrib")
df_precip["tp"] *= 1000

# (YYYY-MM-DD HH:MM:SS) -> (YYYY-MM-DD)
df_precip["valid_time"] = df_precip["valid_time"].dt.floor("D")


# Group by valid_time and sum tp to get 24h total precipitation
df_precip_grouped = df_precip.groupby("valid_time").sum()
df_precip_grouped = df_precip_grouped.to_dataframe().reset_index()
df_precip_grouped.columns = df_precip_grouped.columns.str.upper()
df_precip_grouped.drop(columns=["NUMBER","SURFACE"],inplace=True)
df_precip_grouped.rename(columns={'VALID_TIME': 'DATE','LATITUDE':'CELL_LAT','LONGITUDE':'CELL_LON'}, inplace=True)

del df_precip
gc.collect()

df_weather.columns = df_weather.columns.str.upper()
df_fire = pd.read_csv(dfwildfire_path)
df_fire["REP_DATE"] = pd.to_datetime(df_fire["REP_DATE"])
df_fire = df_fire[(df_fire["REP_DATE"] > "2009-12-31") & (df_fire["SRC_AGENCY"] == "QC")]
df_fire = df_fire.drop(columns=["SRC_AGENCY"])
df_weather["TIME"] = pd.to_datetime(df_weather["TIME"])
df_weather["TIME"] = df_weather["TIME"].dt.date
df_weather, df_fire ,incr_lat,incr_lon= add_coo_area(df_weather, df_fire)
df_weather=add_rh_vpd(df_weather)
df_fire = aggregate_fire(df_fire)
df_fire = excess_fire_distribution(df_fire,incr_lat,incr_lon)
df_fire = add_burned_density_isfire(df_fire)
df_fire.to_csv(generate_fire_path, index=False)
df_intermediate = merge_fire_weather(df_fire,df_weather)
df_intermediate = df_intermediate.drop(columns=['STEP','PROTZONE','ECOZ_NAME','CAUSE','FID','NUMBER','SURFACE'])
df_final = df_intermediate.merge(df_precip_grouped, on=['DATE',"CELL_LAT","CELL_LON"], how='left')
df_final.to_csv(generate_path, index=False)
