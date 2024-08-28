import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, load_model
import efficientnet.tfkeras as efn

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Function to rebuild the model architecture
def build_model():
    input_shape = (224, 224, 3)
    base_model = efn.EfficientNetB0(
        input_shape=input_shape,
        weights=None,
        include_top=False
    )
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(5, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build the model
model = build_model()

# Load the weights from the pre-trained model file
model.load_weights('efficientnetb0_model.h5')

# Label mapping
class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and preprocess the image
            image = load_img(filepath, target_size=(224, 224))
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            
            # Predict
            preds = model.predict(image)
            pred_class = class_names[np.argmax(preds)]
            
            return render_template('index.html', filename=filename, pred_class=pred_class)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
