from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)

MODEL_PATH = 'my_model_ml.h5'
model = tf.keras.models.load_model(MODEL_PATH)


noms_des_classes = ['Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS',
                    'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST', 'AION', 'PT',
                    'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS', 'CB', 'ODPM', 'PRH',
                    'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCR', 'CF', 'VH', 'MCA', 'VS', 'BRAO',
                    'PLQ', 'HPED', 'CL']
labels_significations = {
    'Disease_Risk': 'Présence de pathologie',
    'DR': 'Risque de rétinopathie diabétique',
    'ARMD': 'Dégénérescence maculaire liée à l\'âge',
    'MH': 'Opacités du vitré',
    'DN': 'Identification de drusen',
    'MYA': 'Signes de myopie',
    'BRVO': 'Occlusion veineuse rétinienne',
    'TSLN': 'Lignes tessellées',
    'ERM': 'Membrane épirétinienne',
    'LS': 'Cicatrices au laser',
    'MS': 'Cicatrices maculaires',
    'CSR': 'Rétinopathie séreuse centrale',
    'ODC': 'Importante excavation du disque optique',
    'CRVO': 'Occlusion veineuse rétinienne centrale',
    'TV': 'Vaisseaux rétiniens tortueux',
    'AH': 'Hyalose astéroïdienne',
    'ODP': 'Paleur du disque optique',
    'ODE': 'Œdème du disque optique',
    'ST': 'Shunt optociliaire',
    'AION': 'Neuropathie optique ischémique antérieure',
    'PT': 'Télangiectasies parafovéales',
    'RT': 'Traction rétinienne',
    'RS': 'Signes de rétinite',
    'CRS': 'Choroïdorétinite',
    'EDN': 'Exsudation rétinienne',
    'RPEC': 'Changements de l\'épithélium pigmentaire rétinien',
    'MHL': 'Trou maculaire',
    'RP': 'Rétinite pigmentaire',
    'CWS': 'Cotton wool spots',
    'CB': 'Colobome',
    'ODPM': 'Maculopathie due à une excavation du disque optique',
    'PRH': 'Hémorragie prérétinienne',
    'MNF': 'Fibres nerveuses myélinisées',
    'HR': 'Rétinopathie hémorragique',
    'CRAO': 'Occlusion de l\'artère rétinienne centrale',
    'TD': 'Disque optique incliné',
    'CME': 'Œdème maculaire cystoïde',
    'PTCR': 'Rupture choroïdienne post-traumatique',
    'CF': 'Pli choroidal',
    'VH': 'Hémorragie du vitré',
    'MCA': 'Macroanévrisme',
    'VS': 'Signes de vascularite',
    'BRAO': 'Occlusion de l\'artère rétinienne branche',
    'PLQ': 'Plaque',
    'HPED': 'Détachement pigmentaire hémorragique',
    'CL': 'Collatérales vasculaires'
}


def predict_image(image_path):
    image = Image.open(image_path)
    image = image.resize((356, 536))
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'erreur': 'Aucun fichier envoyé'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'erreur': 'Aucun fichier sélectionné'})

    if file:
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
                    results[noms_des_classes[i]] = float(prob)
                     
                    top_classes = sorted(results, key=results.get, reverse=True)[:2] 
                    
                  
                   
                   

                if 'Disease_Risk' in top_classes and results['Disease_Risk'] < 0.5:
                        # Si la classe 'Disease_Risk' a une probabilité inférieure à 0.5, indiquer qu'il n'y a pas de problème
                        return render_template('result.html', predictions=results, top_classes=["No Disease"], top_probabilities=[100.0])
                else:
                        top_probabilities = [results[class_name] for class_name in top_classes]
                        return render_template('result.html', predictions=results, top_classes=top_classes, top_probabilities=top_probabilities, labels_significations= labels_significations)
            else:
                return jsonify({'erreur': 'Erreur interne dans les prédictions'})

        except Exception as e:
            return jsonify({'erreur': str(e)})

    return jsonify({'erreur': 'Une erreur est survenue lors du téléchargement du fichier'})

@app.route('/results', methods=['POST'])
def results():
    if request.is_json:
        data = request.get_json()
        
        sorted_classes = sorted(data, key=data.get, reverse=True)
        
        top_classes = sorted_classes[:2]
        top_probabilities = [data[class_name] for class_name in top_classes]
        
        return render_template('result.html', top_classes=top_classes, top_probabilities=top_probabilities,labels_significations= labels_significations)

    return jsonify({'erreur': 'Format de données non pris en charge'})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
