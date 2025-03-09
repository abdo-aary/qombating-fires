import gdown
import rarfile
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
extract_to = os.path.join(current_dir,'..', '..', '..', 'storage', 'dataset')
dataset_path = os.path.join(current_dir,'..', '..', '..', 'storage', 'dataset','data.rar')  # Remplacez le chemin du ZIP par celui du RAR
dataset_path = os.path.abspath(dataset_path)


url = 'https://drive.google.com/uc?export=download&id=1e6azaEtipR-tW2PdITl7ePkFYiD54KTu'
gdown.download(url, dataset_path, quiet=False)


with rarfile.RarFile(dataset_path) as rar_ref:
    rar_ref.extractall(extract_to)


os.remove(dataset_path)
