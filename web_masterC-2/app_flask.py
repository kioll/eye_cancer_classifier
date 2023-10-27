from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf  # Assurez-vous d'avoir TensorFlow 2.x installé
import os

app = Flask(name)

# Get the absolute path of the directory where the Flask application is located
app_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = '/Users/enzo/Documents/efrei/eye_cancer_classifier/my_model_ml.h5'
model = tf.keras.models.load_model(MODEL_PATH)


noms_des_classes = ['Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'LS', 'MS', 'CSR', 'ODC', 'CRVO', 'AH', 'ODP', 'ODE', 'AION', 'RS', 'CRS', 'EDN', 'RPEC']

def predict_image(image_path):
    # Prétraitement de l'image pour qu'elle corresponde à l'entrée attendue par votre modèle
    image = Image.open(image_path)
    image = image.resize((536, 356))  # ou toute autre taille attendue par votre modèle
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    print("Prédictions :", predictions) 
    print("Prédictions reçues: ", predictions)  # Ceci affichera les prédictions.

    # Autre débogage pour vérifier le type et la forme des prédictions, cela peut aider à résoudre les problèmes.
    print("Type des prédictions: ", type(predictions))
    print("Forme des prédictions: ", predictions.shape) # pour le débogage

    return predictions

@app.route('/', methods=['GET'])
def index():
    # Page d'accueil
    return render_template('index.html')  # assurez-vous que 'index.html' existe dans le dossier 'templates'

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Aucun fichier envoyé')  # Utiliser flash pour les messages d'erreur
        return redirect(request.url)  # Redirige vers la page de téléchargement si aucun fichier n'est là

    file = request.files['file']

    if file.filename == '':
        flash('Aucun fichier sélectionné')
        return redirect(request.url)

    if file :  # Vous pourriez avoir une fonction allowed_file pour vérifier le type de fichier
        filename = secure_filename(file.filename)
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')

        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)

        try:
            predictions = predict_image(file_path)
            results = {}

            if predictions is not None and len(predictions[0]) == len(noms_des_classes):
                for i, prob in enumerate(predictions[0]):
                    results[noms_des_classes[i]] = prob  # Conservez les probabilités comme des nombres pour le formatage dans le template
            else:
                flash("Erreur: les dimensions des prédictions ne correspondent pas aux attentes")
                return redirect(request.url)

            # Rendre le template 'result.html' en passant les 'results'
            return render_template('result.html', predictions=results)  # Assurez-vous que 'result.html' est le bon nom de votre template

        except Exception as e:
            flash('Une erreur est survenue lors du traitement de votre fichier.')
            return redirect(request.url)

    flash('Une erreur est survenue lors du téléchargement du fichier')
    return redirect(request.url)

@app.route('/results')
def results(prediction):
    # Affichage de la page de résultats avec la prédiction
    return render_template('results.html', prediction=prediction)

if name == 'main':
    app.run(debug=True)  # L'exécution en mode debug permettra une mise au point plus facile
