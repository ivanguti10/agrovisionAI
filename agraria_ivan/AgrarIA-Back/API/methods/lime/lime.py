# Dentro de methods/lime/lime.py
import time
from . import lime_bp
from flask import Flask, jsonify, request
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.offline as pyo
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

tf.compat.v1.disable_v2_behavior

from tensorflow import keras
from keras import Sequential
from keras import layers
from keras.layers import LSTM, Dense, Bidirectional, Dropout, Flatten
from keras import utils

import lime
import shap
import lime.lime_tabular as lt
from lime.lime_tabular import LimeTabularExplainer
import math
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import base64
import imgkit
import os

from flask_cors import CORS  # Importa CORS desde Flask-CORS
from prophet import Prophet
from joblib import dump, load

from shap.plots._force import save_html_return

from methods.lime import lime_bp
from methods.python_conv_lime import python_conv_lime_bp
from methods.pytorch_conv2 import pytorch_conv2_bp
from methods.shap import shap_bp
from methods.time_series_analysis import time_series_analysis_bp
from methods.time_series_prophet import time_series_prophet_bp
from methods.yuca import yuca_bp

import datetime
import json

#import agentes
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, TimeoutBehaviour
from spade.message import Message

from api import son_agentes_activos, detener_agentes, agentes_activos, graficos_recibidos

#Agente que solo envia mensajes a Lime
class PeriodicSenderagentLime(Agent):
    class InformBehav(TimeoutBehaviour):  
        async def run(self):
            
            graficos = await generate_graficos_Lime()
            msg = Message(to=self.get("receiver_jid_Lime"))
            msg.body = json.dumps(graficos)  # Convierte la lista de gráficos a formato JSON
            await self.send(msg)
       
    async def setup(self):
        #print(f"PeriodicSenderagentTimeSeriesAnalysis started at {datetime.datetime.now().time()}")
        print("El agente va a iniciar la generación de gráficos")
        start_at = datetime.datetime.now() + datetime.timedelta(seconds=5)
        b = self.InformBehav(start_at=start_at)
        self.add_behaviour (b)
        
class ReceiverAgent(Agent):
    class RecvBehav(CyclicBehaviour):
        async def run(self):
            global graficos_recibidos
            msg = await self.receive(timeout=300)  # Espera un mensaje durante 180 segundos (ajusta según tus necesidades)

            if msg:
                try:
                   
                    graficos_recibidos = json.loads(msg.body)
                    #print(graficos)
                 
                except json.JSONDecodeError as e:
                    print(f"Error al analizar el mensaje como JSON: {str(e)}")

    async def setup(self):
        b = self.RecvBehav()
        self.add_behaviour(b)
        
async def generate_graficos_Lime():
        archivos = request.files["archivo"]
        graficos = []

        # crop_recommendation.csv
        datos = pd.read_csv(archivos)
        X = datos.iloc[:, 0:6]
        y = datos.iloc[:, 7]
        feature_names = X.columns
        class_names = y.unique()
        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        y_m = utils.to_categorical(y, num_classes=22)

        X = normalize(X, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y_m)

        model = tf.keras.Sequential(
            [
                layers.Dense(units=512, activation="relu", input_shape=[6]),
                layers.Dense(units=512, activation="relu"),
                layers.Dense(units=512, activation="relu"),
                layers.Dense(units=22, activation="softmax"),
            ]
        )

        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        def entrenarModelo():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                  Entrenando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                batch_size=20,
                epochs=50,
            )
            model.save(
                "Modelos/lime.h5"
            )

        def inferencia():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                    Cargando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            model = keras.models.load_model(
                "Modelos/lime.h5"
            )
            return model

        # IF para crear modelo o cargarlo
        if os.path.isfile(
            "Modelos/lime.h5"
            ):
            model = inferencia()
        else:
            entrenarModelo()

        explainer = lt.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True,
        )

        explanation = explainer.explain_instance(
            X_test[1], model.predict, num_features=6, top_labels=1
        )

        # Asegúrate de que grafic sea una cadena de texto
        grafic = str(explanation.as_html())

        # Convierte la imagen en base64
        base64_image = base64.b64encode(grafic.encode("utf-8")).decode("utf-8")
        graficos.append({"tipo": "interactivo", "base64": base64_image, "text": ""})

        model.predict(X_test[1:2])
        # graficos.append(explanation_content)

        return graficos
    

def limeFunction():
    if request.method == "POST":

        #-----------------------------------AGENTES----------------------------------------
        global graficos_recibidos
        graficos_recibidos = None
        try:
            
                detener_agentes()    
                #Credenciales de cada agente
                jid_creds = [
                        ("lime@chalec.org", "hola1234"),
                        ("manageworkflowagent@chalec.org", "hola1234")
                    ]

                if not son_agentes_activos():

                    
                    # Inicializa los agentes aquí
                    agentes_activos["agentLime"] = ReceiverAgent(jid_creds[0][0], jid_creds[0][1])
                    agentes_activos["agentLime"].start(auto_register=True)
                    
                    agentes_activos["agentManager"] = PeriodicSenderagentLime(jid_creds[1][0], jid_creds[1][1])                   
                    agentes_activos["agentManager"].start(auto_register=True)
                    
                    agentes_activos["agentManager"].set("receiver_jid_Lime", jid_creds[0][0])

                    agentes_activos["agentLime"].web.start(hostname="127.0.0.1", port="1007")
                    agentes_activos["agentManager"].web.start(hostname="127.0.0.1", port="1006")           

       
        except ZeroDivisionError as e:
                # Manejar la excepción específica
                print(f"Ocurrió un error: {e}")  
            

        timeout = 180  # Ajustar según sea necesario
        while timeout > 0:
            if graficos_recibidos is not None:
                return jsonify(graficos_recibidos)  # Devolver los gráficos si han sido recibidos
            time.sleep(5)  # Espera un breve periodo antes de volver a verificar
            timeout -= 5

    return None

