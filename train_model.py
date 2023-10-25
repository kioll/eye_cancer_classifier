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
image_folder_eval = './eyes-dataset/Evaluation_Set/Evaluation_Set/Validation/'  #
image_folder_test= './eyes-dataset/Test_Set/Test_Set/Test/'
# Transformer le nom du fichier pour qu'il corresponde aux noms réels des fichiers
data_train['ID'] = data_train['ID'].apply(lambda x: str(x) + '.png')
data_test['ID'] = data_test['ID'].apply(lambda x: str(x) + '.png')
data_evaluation['ID'] = data_evaluation['ID'].apply(lambda x: str(x) + '.png')

# Générer les chemins complets vers les images pour l'ensemble d'entraînement, de test et d'évaluation
image_paths_train = [os.path.join(image_folder_train, filename) for filename in data_train['ID'].values]
image_paths_test = [os.path.join(image_folder_eval, filename) for filename in data_test['ID'].values]
image_paths_evaluation = [os.path.join(image_folder_test, filename) for filename in data_evaluation['ID'].values]

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

# Pour les étiquettes, il est important de les convertir en une forme qui peut être traitée par le modèle, généralement un array numpy.
y_train = data_train['Disease_Risk'].values
y_test = data_test['Disease_Risk'].values
y_evaluation = data_evaluation['Disease_Risk'].values



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()

# Première couche de convolution
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Deuxième couche de convolution
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Troisième couche de convolution
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Quatrième couche de convolution
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Couches entièrement connectées et Flatten pour préparer la sortie du réseau
model.add(Flatten())  # cette couche convertit les caractéristiques 2D en un vecteur 1D
model.add(Dense(512))  # couche entièrement connectée
model.add(Activation('relu'))
model.add(Dropout(0.5))  # pour réduire le surajustement
model.add(Dense(1))  # couche de sortie, NOMBRE_DE_CLASSES doit être le nombre de catégories que vous voulez prédire
model.add(Activation('sigmoid'))

# Compilation du modèle. Vous pouvez expérimenter avec différents optimiseurs et paramètres.
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# Entraînement du modèle
history = model.fit(
    X_train, y_train,  # Assurez-vous que y_train contient les étiquettes de vos données d'entraînement
    batch_size=32,
    epochs=25,  # Le nombre d'époques dépend de votre jeu de données, vous devrez peut-être l'ajuster
    validation_data=(X_test, y_test),
    shuffle=True
)

# Évaluer la performance du modèle sur votre ensemble de données d'évaluation
scores = model.evaluate(X_evaluation, y_evaluation, verbose=1)
print("Performance du modèle sur l'ensemble d'évaluation:", scores)
