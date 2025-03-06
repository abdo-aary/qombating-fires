import gdown
import zipfile
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_to = os.path.join(current_dir,'..', '..', '..', 'storage', 'dataset')
dataset_path = os.path.join(current_dir,'..', '..', '..', 'storage', 'dataset','wildfire.zip')
dataset_path = os.path.abspath(dataset_path)
# Télécharger le fichier
url = 'https://drive.google.com/uc?id=1MuJQw3biKt-ChE1tp1Fv1NjNp8VnV8Xx'
gdown.download(url, dataset_path, quiet=False)

# Dézipper le fichier
zip_path = dataset_path


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

# Supprimer le fichier ZIP après extraction
os.remove(zip_path)