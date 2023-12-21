import os
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"]=set(['png', 'jpg', 'jpeg'])
app.config["UPLOAD_FOLDER"] = "static/uploads/"

def allowed_file(filename):
    return "." in filename and \
        filename.split(".", 1) [1] in app.config["ALLOWED_EXTENSIONS"]

model = load_model("model_u2net.h5", compile = False)
# with open("label.txt", "r") as file:
#     labels = file.read().splitlines()


@app.route("/")
def index():
    return jsonify({
        "status":{
            "code":200,
            "message":"API berjalan"
        },
        "data":None
    }),200

@app.route("/prediction", methods=("GET", "POST"))
def prediction():
    if request.method == "POST":
        image = request.files["image"]
        if image and secure_filename(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            # result = model.predict(processed_image)
            img = Image.open(image_path).convert("RGB")
            img = img.resize((512,512))
            img_array = np.asanyarray(img)
            img_array = np.expand_dims(img_array, axis=0)
            nomalized_image_array = (img_array.astype(np.float32)/ 127.5) - 1
            data = np.ndarray(shape=(1, 512, 512, 3), dtype=np.float32)
            data[0]= nomalized_image_array

            prediction = model.predict(data)
            # index = np.argmax(prediction)
            # class_names = labels[index]
            # confidence_score = prediction[0][index]

            # print("Prediction array:", prediction)
            # class_index = np.argmax(prediction)
            
            # confidence_score = prediction[0][class_index]
            result_image = post_process_prediction(image_path, prediction[0][0])

            return jsonify({
                "status":{
                    "code":200,
                    "message":"Succes prediction",
                },
                "data":{
                    # "prediksi_beras":float(class_index)
                    # "confidence_score":float(confidence_score)
                    "result_image_path": result_image
                }
            }),200
        else:
            return jsonify({
                "status":{
                    "code":400,
                    "message":"Client side error"
                },
                "data":None
            }), 400  
    else:
        return jsonify({
            "status":{
                "code":405,
                "message":"Method now Allowed"
            },
            "data":None,
        }), 405

def post_process_prediction(original_image_path, prediction):
    image = cv2.imread(original_image_path)
    image_h, image_w, _ = image.shape

    # Resize prediction to match original image dimensions
    y0 = cv2.resize(prediction, (image_w, image_h))
    y0 = np.expand_dims(y0, axis=-1)
    y0 = np.concatenate([y0, y0, y0], axis=-1)

    # Concatenate original image, prediction, and a white line
    line = np.ones((image_h, 10, 3)) * 255
    result_image = np.concatenate([image * y0], axis=1)

    # Save the result image
    result_image_path = os.path.join('static/uploads', secure_filename(original_image_path.split('/')[-1]))
    cv2.imwrite(result_image_path, result_image)
    return result_image_path

if __name__ == "__main__":
    app.run()