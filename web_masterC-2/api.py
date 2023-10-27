from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import os

model = load_model('my_model.h5')
app = Flask(__name__)

# Get the absolute path of the directory where the Flask application is located
app_dir = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    
    if file and file.filename.endswith('.png'):
        image_filename = file.filename  # Garde le même nom de fichier
        image_path = os.path.join(app_dir, 'static/uploads', image_filename)
        file.save(image_path)
        data = "Test réussi"  # Message de test réussi
    else:
        data = "Le fichier n'est pas un PNG"
        image_filename = None
    
    return redirect(url_for('result', data=data, image_filename=image_filename))

@app.route('/result')
def result():
    data = request.args.get('data', '')
    image_filename = request.args.get('image_filename', '')
    return render_template('result.html', data=data, image_filename=image_filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
