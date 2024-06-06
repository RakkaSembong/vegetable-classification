import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vegetable', methods=['POST'])
def vegetable():
    image_request = request.files['image']
    image_pil = Image.open(image_request)
    expected_size = (150, 150)
    resized_image_pil = image_pil.resize(expected_size)
    image_array = np.array(resized_image_pil)
    rescaled_image_array = image_array / 255.
    batched_rescaled_image_array = np.array([rescaled_image_array])

    loaded_model = tf.keras.models.load_model('vegetable_classification.h5')
    result = loaded_model.predict(batched_rescaled_image_array)
    return jsonify({'prediction': get_formated_predict_result(result)})

# def get_formated_predict_result(predict_result) :
#     class_indices = {
#         'Radish': 0,
#         'Tomato': 1,
#         'Cauliflower': 2,
#         'Cucumber': 3,
#         'Bitter_Gourd': 4,
#         'Papaya': 5,
#         'Bottle_Gourd': 6,
#         'Broccoli': 7,
#         'Potato': 8,
#         'Brinjal': 9,
#         'Bean': 10,
#         'Pumpkin': 11,
#         'Cabbage': 12,
#         'Capsicum': 13,
#         'Carrot': 14
#     }
#     inverted_class_indices = {}
#
#     for key in class_indices:
#         class_indices_key = key
#         class_indices_value = class_indices[key]
#
#         inverted_class_indices[class_indices_value] = class_indices_key
#
#     procesed_predict_result = predict_result[0]
#     maxIndex = 0
#     maxValue = 0
#
#     for index in range(len(procesed_predict_result)):
#         if procesed_predict_result[index] > maxValue:
#             maxValue = procesed_predict_result[index]
#             maxIndex = index
#
#     return inverted_class_indices[maxIndex]


def get_formated_predict_result(predict_result):
    class_indices = {'Bean': 0, 'Bitter_Gourd': 1, 'Bottle_Gourd': 2, 'Brinjal': 3, 'Broccoli': 4, 'Cabbage': 5, 'Capsicum': 6, 'Carrot': 7, 'Cauliflower': 8, 'Cucumber': 9, 'Papaya': 10, 'Potato': 11, 'Pumpkin': 12, 'Radish': 13, 'Tomato': 14}

    inverted_class_indices = {v: k for k, v in class_indices.items()}

    max_index = np.argmax(predict_result)

    return inverted_class_indices[max_index]


if __name__ == '__main__':
    app.run(debug=True)
