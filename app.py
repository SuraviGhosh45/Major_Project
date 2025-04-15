from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from predict_model import predict

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('research.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Check if an image was uploaded
    file = request.files.get('file')
    symptoms = request.form.get('symptoms', '').strip()

    # If only image is provided
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = predict(image_path=filepath)
            return render_template('research.html', prediction=result, img_path=filepath)
        except Exception as e:
            return render_template('research.html', error=str(e))

    # If only 30 features are provided
    elif symptoms:
        try:
            features = [float(x.strip()) for x in symptoms.split(',')]
            if len(features) != 30:
                raise ValueError("⚠️ Please provide exactly 30 numerical features separated by commas.")
            result = predict(features=features)
            return render_template('research.html', prediction=result)
        except Exception as e:
            return render_template('research.html', error=str(e))

    # If neither provided
    else:
        return render_template('research.html', error="⚠️ Please upload an image or enter 30 features.")

if __name__ == '__main__':
    app.run(debug=True)