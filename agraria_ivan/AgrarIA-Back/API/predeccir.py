import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import io
import smtplib
import tempfile
from PIL import Image  # type: ignore # Asegúrate de usar PIL para la manipulación de imágenes
from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore

from psutil import virtual_memory # type: ignore
from pathlib import Path
import os
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard # type: ignore
import cv2 # type: ignore
import json
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

root_path = Path("C:\\Users\\ivan2\\OneDrive\\Escritorio\\newplantvillage\\newplantvillage")
root_path_str = str(root_path)
classes_train = ['Apple scab', 'Apple Black Rot', 'Manzano Roya del manzano y del cedro', 'Manzano saludable', 'Arándano saludable', 'Cereza (incluyendo ácida) Oidio', 'Cereza (incluyendo ácida) saludable', 'Vid Podredumbre negra', 'Vid Esca (Sarampión negro)', 'Vid Tizón de la hoja (Mancha de la hoja de Isariopsis)', 'Vid saludable', 'Naranja Huanglongbing (Greening de los cítricos)', 'Melocotón Mancha bacteriana', 'Melocotón saludable', 'Pimiento, morrón Mancha bacteriana', 'Pimiento, morrón saludable', 'Patata Tizón temprano', 'Patata Tizón tardío', 'Patata saludable', 'Frambuesa saludable', 'Soja saludable', 'Calabaza Oidio', 'Fresa Chamusco de la hoja', 'Fresa saludable', 'Tomate Mancha bacteriana', 'Tomate Tizón temprano', 'Tomate Tizón tardío', 'Tomate Moho de la hoja', 'Tomate Mancha foliar de Septoria', 'Tomate Araña roja (Ácaro de dos manchas)', 'Tomate Mancha diana', 'Tomate Virus del rizado amarillo de la hoja del tomate', 'Tomate Virus del mosaico del tomate', 'Tomate saludable']

# Cargar el modelo base ResNet50 preentrenado en ImageNet sin la parte superior
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
base_model.trainable = True

# Crear un nuevo modelo
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes_train), activation='softmax')  # Asegúrate de que el número de clases es correcto
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

model_path = 'C:\\Users\\ivan2\\OneDrive\\Escritorio\\best_model.h5'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"El archivo de modelo en la ruta {model_path} no existe.")

try:
    model = tf.keras.models.load_model(model_path)
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

print(classes_train)  # Muestra los nombres de todas las clases
#model.summary()  # Verificar el resumen del modelo



import tensorflow as tf # type: ignore
print(tf.__version__)

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
            dataset = np.array([img_resize]) / 255.0

            # Realizar la predicción
            predictions = model.predict(dataset)
            confidence = np.max(predictions, axis=-1)[0] * 100
            result = np.argmax(predictions, axis=-1)

            # Construcción del resultado en formato JSON
            predicted_class = classes_train[result[0]]
            final_json = {
                "filename": file.filename,
                "class": predicted_class,
                "confidence": f"{confidence:.2f}"
            }

            print(final_json)
            return jsonify(final_json)
        else:
            return jsonify({"error": "Failed to process the image"}), 500

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
    app.run(debug=True)