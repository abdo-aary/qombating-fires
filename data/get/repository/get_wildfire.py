import gdown
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_to = os.path.join(current_dir,'..', '..', '..', 'storage', 'dataset','wildfires')
dataset_path = os.path.join(current_dir,'..', '..', '..', 'storage', 'dataset','wildfires','CANADA_WILDFIRES.csv')
dataset_path = os.path.abspath(dataset_path)

url = "https://drive.google.com/uc?id=1kjIsv7JOjmLfK841b86NsrxnxavSD2T8"
gdown.download(url, dataset_path, quiet=False)

