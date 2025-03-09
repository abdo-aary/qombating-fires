import pandas as pd
import os
import warnings
from data.view.pixel_label import *
import geopandas as gpd

current_dir  = os.getcwd()
dataset_path = os.path.join(current_dir, '..','..', 'storage', 'dataset','wildfires',"dataset_pre_analysis.csv")
date = '2019-08-01'
dataset_analysis_path = os.path.join(current_dir, '..','..', 'storage', 'dataset','wildfires',"wildfires_data.csv")

url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces.zip"
canada = gpd.read_file(url)
quebec = canada[canada["name"] == "Québec"]




dataset = pd.read_csv(dataset_path)

# convert IS_FIRE to float
dataset["IS_FIRE"] = dataset["IS_FIRE"].astype(float)
filtered_dataset = dataset[(dataset["CELL_LON"] >= -79.5) & (dataset["CELL_LON"] <= -64) &
                 (dataset["CELL_LAT"] >= 45.25) & (dataset["CELL_LAT"] <= 52)]
filtered_dataset[["COORDINATES_LAT","COORDINATES_LON" ]] = ((filtered_dataset[["CELL_LAT", "CELL_LON"]]
                                                            - [45.25,-79.5]) / [0.25,0.25]).astype(int)



filtered_dataset_date = filtered_dataset[filtered_dataset['DATE'] == date]
# Créer une géométrie de points avec latitudes et longitudes
filtered_dataset_date['IN_OCEAN'] = filtered_dataset_date.apply(lambda row: is_cell_in_ocean(row['CELL_LAT'], row['CELL_LON'], ocean_gdf), axis=1)


# Extraire uniquement les points situés dans l'océan
points_in_ocean = filtered_dataset_date[filtered_dataset_date['IN_OCEAN']]

# Extraire la liste des latitudes et longitudes des points dans l'océan
lat_lon_list_ocean = points_in_ocean[['CELL_LAT', 'CELL_LON']].values.tolist()
lat_lon_set_ocean = set(map(tuple, lat_lon_list_ocean))

# Supprimer les lignes de filtered_dataset qui ont des coordonnées dans l'océan
filtered_dataset_no_ocean = filtered_dataset[~filtered_dataset[['CELL_LAT', 'CELL_LON']].apply(tuple, axis=1).isin(lat_lon_set_ocean)]

filtered_dataset_no_ocean_date = filtered_dataset_no_ocean[filtered_dataset_no_ocean['DATE'] == date]
# Créer une géométrie de points avec latitudes et longitudes
filtered_dataset_no_ocean_date['IN_QC'] = filtered_dataset_no_ocean_date.apply(lambda row: is_cell_in_quebec(row['CELL_LAT'], row['CELL_LON'], quebec), axis=1)


# Extraire uniquement les points situés en dehors du québec
points_out_quebec = filtered_dataset_no_ocean_date[filtered_dataset_no_ocean_date['IN_QC']==False]

# Extraire la liste des latitudes et longitudes des points dans l'océan
lat_lon_list_notqc = points_out_quebec[['CELL_LAT', 'CELL_LON']].values.tolist()
lat_lon_set_notqc = set(map(tuple, lat_lon_list_notqc))

# Supprimer les lignes de filtered_dataset qui ont des coordonnées dans l'océan
filtered_dataset_no_ocean_qconly = filtered_dataset_no_ocean[~filtered_dataset_no_ocean[['CELL_LAT', 'CELL_LON']].apply(tuple, axis=1).isin(lat_lon_set_notqc)]



filtered_dataset_no_ocean_qconly["DATE"] = pd.to_datetime(filtered_dataset_no_ocean_qconly["DATE"], errors='coerce')

# Créer un masque pour exclure les dates entre le 7 décembre et le 31 décembre
mask_december_exclude = (filtered_dataset_no_ocean_qconly["DATE"].dt.month == 12) & (filtered_dataset_no_ocean_qconly["DATE"].dt.day > 7)

# Créer un masque pour exclure les dates entre le 1er janvier et le 11 mars (y compris le 1er janvier, mais excluant le 11 mars)
mask_jan_march_exclude = ((filtered_dataset_no_ocean_qconly["DATE"].dt.month == 1) | (filtered_dataset_no_ocean_qconly["DATE"].dt.month == 3)) & \
                          ((filtered_dataset_no_ocean_qconly["DATE"].dt.day >= 1) & (filtered_dataset_no_ocean_qconly["DATE"].dt.day <= 10))

# Créer un masque pour exclure toutes les dates avant le 12 mars
mask_before_march_12 = (filtered_dataset_no_ocean_qconly["DATE"].dt.month == 1) | \
                        ((filtered_dataset_no_ocean_qconly["DATE"].dt.month == 2) & (filtered_dataset_no_ocean_qconly["DATE"].dt.day <= 28)) | \
                        ((filtered_dataset_no_ocean_qconly["DATE"].dt.month == 3) & (filtered_dataset_no_ocean_qconly["DATE"].dt.day <= 10))

# Filtrer les données selon ces critères
filtered_data = filtered_dataset_no_ocean_qconly[~(mask_december_exclude | mask_jan_march_exclude | mask_before_march_12)]


filtered_data = filtered_data.sort_values(by=['CELL_LAT', 'CELL_LON', 'DATE'])

# Ajouter une colonne indiquant si le pixel sera en feu au prochain temps
filtered_data['IS_FIRE_NEXT_DAY'] = filtered_data.groupby(['CELL_LAT', 'CELL_LON'])['IS_FIRE'].shift(-1)

# Remplacer les NaN (qui apparaissent pour le dernier temps de chaque pixel) par 0
filtered_data['IS_FIRE_NEXT_DAY'] = filtered_data['IS_FIRE_NEXT_DAY'].fillna(0).astype(int)


filtered_data = filtered_data.sort_values(by=['DATE','CELL_LAT', 'CELL_LON'])
filtered_data.to_csv(dataset_analysis_path, index=False)