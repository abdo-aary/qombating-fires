import gdown
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_to = os.path.join(current_dir,'..', '..', '..', 'storage', 'dataset','wildfires')
dataset_path = os.path.join(current_dir,'..', '..', '..', 'storage', 'dataset','wildfires','wildfires_data.csv')
dataset_path = os.path.abspath(dataset_path)

url = 'https://drive.google.com/uc?id=19ZqBzA1zCKM_YtOA16sy6rHTqw0RmVzz'
gdown.download(url, dataset_path, quiet=False)

