import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Charger les données
data_train = pd.read_csv('./eyes-dataset/Training_Set/Training_Set/RFMiD_Training_Labels.csv')
data_test = pd.read_csv('./eyes-dataset/Test_Set/Test_Set/RFMiD_Testing_Labels.csv')
data_evaluation = pd.read_csv('./eyes-dataset/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv')

# Définir le chemin du dossier contenant les images
image_folder_train = './eyes-dataset/Training_Set/Training_Set/Training/'  # Mettez le chemin approprié
image_folder_eval = './eyes-dataset/Evaluation_Set/Evaluation_Set/Evaluation/'  #
image_folder_test= './eyes-dataset/Test_Set/Test_Set/Test/'
# Transformer le nom du fichier pour qu'il corresponde aux noms réels des fichiers
data_train['ID'] = data_train['ID'].apply(lambda x: str(x) + '.png')
data_test['ID'] = data_test['ID'].apply(lambda x: str(x) + '.png')
data_evaluation['ID'] = data_evaluation['ID'].apply(lambda x: str(x) + '.png')

# Générer les chemins complets vers les images pour l'ensemble d'entraînement, de test et d'évaluation
image_paths_train = [os.path.join(image_folder, filename) for filename in data_train['ID'].values]
image_paths_test = [os.path.join(image_folder, filename) for filename in data_test['ID'].values]
image_paths_evaluation = [os.path.join(image_folder, filename) for filename in data_evaluation['ID'].values]

# Fonction pour prétraiter les images
def preprocess_images(image_paths, image_dimensions=(224, 224)):
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=image_dimensions)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    return np.array(images)

# Utilisation de la fonction pour prétraiter les images
X_train = preprocess_images(image_paths_train)
X_test = preprocess_images(image_paths_test)
X_evaluation = preprocess_images(image_paths_evaluation)

# Maintenant, vous avez X_train, X_test et X_evaluation contenant les images prétraitées pour chaque ensemble de données.
