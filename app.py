import os
from flask import Flask, render_template, request, send_from_directory
from keras_preprocessing import image
from keras.models import load_model
import numpy as np
import tensorflow as tf

app = Flask(__name__)

STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
# Path to the folder where we store the different models
MODEL_FOLDER = STATIC_FOLDER + '/models'


def load__model():
    """Load model once at running time for all the predictions"""
    print('[INFO] : Model loading ................')
    global model
    # model = tf.keras.models.load_model(MODEL_FOLDER + '/catsVSdogs.h5')
    model = load_model(MODEL_FOLDER + '/cat_dog_classifier.h5')
    global graph
    graph = tf.get_default_graph()
    print('[INFO] : Model loaded')


def predict(fullpath):
    data = image.load_img(fullpath, target_size=(128, 128, 3))
    # (150,150,3) ==> (1,150,150,3)
    data = np.expand_dims(data, axis=0)
    # Scaling
    data = data.astype('float') / 255

    # Prediction

    with graph.as_default():
        result = model.predict(data)

    return result


# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Process file and predict his label
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        result = predict(fullname)

        pred_prob = result.item()

        if pred_prob > .5:
            label = 'Dog'
            accuracy = round(pred_prob * 100, 2)
        else:
            label = 'Cat'
            accuracy = round((1 - pred_prob) * 100, 2)

        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def create_app():
    load__model()
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
