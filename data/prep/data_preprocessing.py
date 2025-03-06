import cfgrib
import os
from data.utils.data_management import *
current_dir = os.path.dirname(os.path.abspath(__file__))
dfweather_path = os.path.join(current_dir, '..', '..', 'storage', 'dataset','weather.grib')
dfwildfire_path = os.path.join(current_dir, '..', '..', 'storage', 'dataset','CANADA_WILDFIRES.csv')
generate_path = os.path.join(current_dir, '..', '..', 'storage', 'dataset',"dataset_pre_analysis.csv")
generate_fire_path = os.path.join(current_dir, '..', '..', 'storage', 'dataset',"dataset_fire_cleaned.csv")
dfweather_path  = os.path.abspath(dfweather_path)
dfwildfire_path = os.path.abspath(dfwildfire_path)

with cfgrib.open_dataset(dfweather_path) as ds:
    df_weather = ds.to_dataframe().reset_index()


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
df_final = merge_fire_weather(df_fire,df_weather)
df_final = df_final.drop(columns=['STEP','PROTZONE','ECOZ_NAME','CAUSE','FID','NUMBER','SURFACE'])
df_final.to_csv(generate_path, index=False)
