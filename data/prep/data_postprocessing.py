from data.utils.data_management import *
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
dfsample= os.path.join(current_dir, '..', '..', 'storage', 'output','test_samples_predictions.csv')
dfsample_measure = os.path.join(current_dir, '..', '..', 'storage', 'output','test_samples_measure.csv')
df = pd.read_csv(dfsample)
df[["CELL_WIDTH","CELL_LENGTH"]] = vectorized_cell_width_length(df["CELL_LAT"],df["CELL_LON"],0.25,0.25)
df.to_csv(dfsample_measure,index=False)