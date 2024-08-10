import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import io
import smtplib
import tempfile
from PIL import Image  # type: ignore # Asegúrate de usar PIL para la manipulación de imágenes
from flask import Flask, request, jsonify, send_from_directory # type: ignore
from flask_cors import CORS # type: ignore

from psutil import virtual_memory # type: ignore
from pathlib import Path
import os
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

import datetime
import cv2 # type: ignore
import json


import urllib.parse
import tflite_runtime.interpreter as tflite


import tflite_runtime.interpreter as tflite


app = Flask(__name__, static_folder='agraria_ivan/AgrarIA-Front/dist/agrar-ia')
CORS(app)  # Habilita CORS para todas las rutas

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')



@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

plagas = [
    {"id": 1, "title": "Sarna del Manzano", "imgSrc": "../../../assets/img/plagas/apple scab.jpg", "altText": "Apple Scab", "category": "Manzano"},
    {"id": 2, "title": "Pudrición negra del Manzano", "imgSrc": "../../../assets/img/plagas/apple black rot.jpg", "altText": "Apple Black rot", "category": "Manzano"},
    {"id": 3, "title": "Roya del cedro y del manzano", "imgSrc": "../../../assets/img/plagas/roya cedro.jpg", "altText": "Roya", "category": "Manzano"},
    {"id": 4, "title": "Manzano sano", "imgSrc": "../../../assets/img/plagas/manzano sana.jpg", "altText": "manzano sano", "category": "Manzano"},
    {"id": 5, "title": "Arandano", "imgSrc": "../../../assets/img/plagas/Arandano.png", "altText": "Arandano", "category": "Arandano"},
    {"id": 6, "title": "Cereza", "imgSrc": "../../../assets/img/plagas/cereza.jpg", "altText": "cereza", "category": "Cereza"},
    {"id": 7, "title": "Oidio Cerezo", "imgSrc": "../../../assets/img/plagas/oidio cerezo.jpg", "altText": "oidio cerezo", "category": "Cereza"},
    {"id": 8, "title": "Prodedumbre negra de la Uva", "imgSrc": "../../../assets/img/plagas/Prodedumbre negra de la Uva.jpg", "altText": "Prodedumbre negra de la Uva", "category": "Uva"},
    {"id": 9, "title": "Madera de la Vid", "imgSrc": "../../../assets/img/plagas/Madera de la vid.jpg", "altText": "Madera de la Vid", "category": "Uva"},
    {"id": 10, "title": "Hoja de Vid sana", "imgSrc": "../../../assets/img/plagas/Hoja de Vid sana.jpg", "altText": "Hoja de Vid sana", "category": "Uva"},
    {"id": 11, "title": "Tizon de la Vid", "imgSrc": "../../../assets/img/plagas/Tizon de la Vid.png", "altText": "Tizon de la Vid", "category": "Uva"},
    {"id": 12, "title": "Enfermedad del Dragon Amarillo", "imgSrc": "../../../assets/img/plagas/Enfermedad del Dragon Amarillo.jpg", "altText": "Enfermedad del Dragon Amarillo", "category": "Cítricos"},
    {"id": 13, "title": "Mancha bacteriana del Melocoton", "imgSrc": "../../../assets/img/plagas/Mancha bacteriana del Melocoton.jpg", "altText": "Mancha bacteriana del Melocoton", "category": "Melocotón"},
    {"id": 14, "title": "Melocoton", "imgSrc": "../../../assets/img/plagas/Melocoton.jpeg", "altText": "Melocoton", "category": "Melocotón"},
    {"id": 15, "title": "Mancha bacteriana Pimiento", "imgSrc": "../../../assets/img/plagas/Mancha bacteriana Pimiento.jpg", "altText": "Mancha bacteriana Pimiento", "category": "Pimiento"},
    {"id": 16, "title": "Pimiento", "imgSrc": "../../../assets/img/plagas/Pimiento sano.jpg", "altText": "Pimiento", "category": "Pimiento"},
    {"id": 17, "title": "Tizon temprano de la Patata", "imgSrc": "../../../assets/img/plagas/Tizon temprano de la Patata.png", "altText": "Tizon temprano de la Patata", "category": "Patata"},
    {"id": 18, "title": "Patata", "imgSrc": "../../../assets/img/plagas/Patata.jpg", "altText": "Patata", "category": "Patata"},
    {"id": 19, "title": "Tizon tardio de Patata", "imgSrc": "../../../assets/img/plagas/Tizon tardio de Patata.jpeg", "altText": "Tizon tardio de Patata", "category": "Patata"},
    {"id": 20, "title": "Frambuesa", "imgSrc": "../../../assets/img/plagas/Frambuesa.jpg", "altText": "Frambuesa", "category": "Frambuesa"},
    {"id": 21, "title": "Frijol", "imgSrc": "../../../assets/img/plagas/Frijol.jpeg", "altText": "Frijol", "category": "Frijol"},
    {"id": 22, "title": "Oidio de Calabaza", "imgSrc": "../../../assets/img/plagas/Oidio de Calabaza.jpg", "altText": "Oidio de Calabaza", "category": "Calabaza"},
    {"id": 23, "title": "Fresa", "imgSrc": "../../../assets/img/plagas/Fresa.jpg", "altText": "Fresa", "category": "Fresa"},
    {"id": 24, "title": "Quemadura de Fresa", "imgSrc": "../../../assets/img/plagas/Quemadura de Fresa.jpg", "altText": "Quemadura de Fresa", "category": "Fresa"},
    {"id": 25, "title": "Mancha bacteriana del Tomate", "imgSrc": "../../../assets/img/plagas/Mancha bacteriana del Tomate.jpg", "altText": "Mancha bacteriana del Tomate", "category": "Tomate"},
    {"id": 26, "title": "Tizon temprano del Tomate", "imgSrc": "../../../assets/img/plagas/Tizon temprano del Tomate.jpeg", "altText": "Tizon temprano del Tomate", "category": "Tomate"},
    {"id": 27, "title": "Tomate", "imgSrc": "../../../assets/img/plagas/Tomate.jpg", "altText": "Tomate", "category": "Tomate"},
    {"id": 28, "title": "Tizon tardio del Tomate", "imgSrc": "../../../assets/img/plagas/Tizon tardio del Tomate.jpg", "altText": "Tizon tardio del Tomate", "category": "Tomate"},
    {"id": 29, "title": "Mildiu del Tomate", "imgSrc": "../../../assets/img/plagas/Mildiu del Tomate.jpg", "altText": "Mildiu del Tomate", "category": "Tomate"},
    {"id": 30, "title": "Septoriosis del Tomate", "imgSrc": "../../../assets/img/plagas/Septoriosis del Tomate.jpeg", "altText": "Septoriosis del Tomate", "category": "Tomate"},
    {"id": 31, "title": "Araña roja del Tomate", "imgSrc": "../../../assets/img/plagas/Araña roja del Tomate.jpg", "altText": "Araña roja del Tomate", "category": "Tomate"},
    {"id": 32, "title": "Mancha de diana del Tomate", "imgSrc": "../../../assets/img/plagas/Mancha de diana del Tomate.jpg", "altText": "Mancha de diana del Tomate", "category": "Tomate"},
    {"id": 33, "title": "Virus del mosaico del tomate", "imgSrc": "../../../assets/img/plagas/Virus del mosaico del tomate.jpg", "altText": "Virus del mosaico del tomate", "category": "Tomate"},
    {"id": 34, "title": "Virus del rizado amarillo del tomate", "imgSrc": "../../../assets/img/plagas/Virus del rizado amarillo del tomate.jpeg", "altText": "Virus del rizado amarillo del tomate", "category": "Tomate"}
]



classes_train = ['Apple scab', 'Apple Black Rot', 'Manzano Roya del manzano y del cedro', 'Manzano saludable', 'Arándano saludable', 'Cereza (incluyendo ácida) Oidio', 'Cereza (incluyendo ácida) saludable', 'Vid Podredumbre negra', 'Vid Esca (Sarampión negro)', 'Vid Tizón de la hoja (Mancha de la hoja de Isariopsis)', 'Vid saludable', 'Naranja Huanglongbing (Greening de los cítricos)', 'Melocotón Mancha bacteriana', 'Melocotón saludable', 'Pimiento, morrón Mancha bacteriana', 'Pimiento, morrón saludable', 'Patata Tizón temprano', 'Patata Tizón tardío', 'Patata saludable', 'Frambuesa saludable', 'Soja saludable', 'Calabaza Oidio', 'Fresa Chamusco de la hoja', 'Fresa saludable', 'Tomate Mancha bacteriana', 'Tomate Tizón temprano', 'Tomate Tizón tardío', 'Tomate Moho de la hoja', 'Tomate Mancha foliar de Septoria', 'Tomate Araña roja (Ácaro de dos manchas)', 'Tomate Mancha diana', 'Tomate Virus del rizado amarillo de la hoja del tomate', 'Tomate Virus del mosaico del tomate', 'Tomate saludable']


model_path = 'models/converted_model.tflite'


if not os.path.exists(model_path):
    raise FileNotFoundError(f"El archivo de modelo en la ruta {model_path} no existe.")

try:
   # Cargar el modelo TFLite
    interpreter = tflite.Interpreter(model_path)
    interpreter.allocate_tensors()

        # Obtener detalles del tensor de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

print(classes_train)  # Muestra los nombres de todas las clases
#model.summary()  # Verificar el resumen del modelo


@app.route('/plagas', methods=['GET'])
def get_plagas():
    filter_category = request.args.get('filter')
    if filter_category:
        filtered_plagas = [plaga for plaga in plagas if plaga['category'] == filter_category]
        return jsonify(filtered_plagas)
    return jsonify(plagas)




@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        if img is not None:
            img_resize = cv2.resize(img, (200, 200))
            
            # Aquí es donde debes normalizar la imagen
            input_data = np.array([img_resize], dtype=np.float32) / 255.0  # Normaliza a [0, 1]

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            confidence = np.max(output_data, axis=-1)[0] * 100
            result = np.argmax(output_data, axis=-1)

            predicted_class = classes_train[result[0]]
            final_json = {
                "filename": file.filename,
                "class": predicted_class,
                "confidence": f"{confidence:.2f}"
            }

            print(final_json)
            return jsonify(final_json)
        else:
            return jsonify({"error": "Failed to process the image F"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/send-email', methods=['POST'])
def send_email():
    try:
        data = request.get_json()
        print(f"Datos recibidos: {data}")
        
        user_name = data.get('name')
        user_email = data.get('email')
        user_phone = data.get('phone')
        user_message = data.get('message')

        recipient_email = "tfgivan34@gmail.com"
        subject = "Nuevo mensaje de contacto"
        body = f"Nombre: {user_name}\nEmail: {user_email}\nTeléfono: {user_phone}\n\nMensaje:\n{user_message}"

        msg = MIMEMultipart()
        msg['From'] = user_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        # Usar utf-8 para el contenido del mensaje
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login('tfgivan34@gmail.com', 'x h h i r c o s q z o f d a l n')
            server.sendmail(user_email, recipient_email, msg.as_string())
            server.quit()
        except smtplib.SMTPException as smtp_err:
            print(f"Error SMTP: {smtp_err}")
            return jsonify({"error": f"Error SMTP: {smtp_err}"}), 500
        except Exception as e:
            print(f"Error al enviar el correo: {e}")
            return jsonify({"error": f"Error al enviar el correo: {e}"}), 500

        return jsonify({"message": "Correo enviado exitosamente"}), 200

    except Exception as e:
        print(f"Error en la solicitud: {e}")
        return jsonify({"error": f"Error en la solicitud: {e}"}), 500
if __name__ == '__main__':
     app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))