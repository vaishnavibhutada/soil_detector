import os
import uuid
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
# Load  trained image model
model = load_model('..\dataset\image_model.h5')
  

def prepare_image(img_path):
    img = Image.open(img_path)
    img = img.resize((128, 128)) 
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join('static', filename)
            
          
            file.save(filepath)

           
            img_array = prepare_image(filepath)

            
            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            class_label = ['Alluvial soil', 'Black soil', 'Clay soil', 'Red soil']
            result = class_label[class_index]
            confidence = prediction[0][class_index] * 100
            

           
            return render_template("result.html", result=result, confidence=confidence, image_filename=filename)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
