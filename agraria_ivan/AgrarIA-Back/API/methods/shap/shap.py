import time
from . import shap_bp

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

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.keras.backend.get_session
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

class PeriodicSenderagentShap(Agent):
    class InformBehav(TimeoutBehaviour):  
        async def run(self):
            
            graficos = await generate_graficos_Shap()
            msg = Message(to=self.get("receiver_jid_Shap"))
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
    
async def generate_graficos_Shap():
        archivos = request.files["archivo"]
        graficos = []

        # crop_recommendation.csv
        datos = pd.read_csv(archivos)
        datos = datos.loc[
            (datos.iloc[:, 7] == "cotton")
            | (datos.iloc[:, 7] == "kidneybeans")
            | (datos.iloc[:, 7] == "watermelon")
            | (datos.iloc[:, 7] == "papaya")
            | (datos.iloc[:, 7] == "maize"),
            :,
        ]
        X = datos.iloc[:, 0:6]
        y = datos.iloc[:, 7]
        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        y_m = utils.to_categorical(y, num_classes=5)
        X = normalize(X, axis=0)
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y_m, range(len(X)), test_size=0.25)




        model = tf.keras.Sequential(
            [
                layers.Dense(units=512, activation="relu", input_shape=[6]),
                layers.Dense(units=512, activation="relu"),
                layers.Dense(units=512, activation="relu"),
                layers.Dense(units=5, activation="softmax"),
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
                "Modelos/shap.h5"
            )

        def inferencia():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                    Cargando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            model = keras.models.load_model(
                "Modelos/shap.h5"
            )
            return model

        # IF para crear modelo o cargarlo
        if os.path.isfile(
            "Modelos/shap.h5"
        ):
            model = inferencia()
        else:
            entrenarModelo()

        explainer_2 = shap.DeepExplainer(model, X_train)
        shap_values_2 = explainer_2.shap_values(X_test)
        plt.figure()
        shap.summary_plot(
            shap_values_2,
            title="Impacto de cada variable en la predicción de una clase",
            plot_size=(16, 9),
            class_names=datos["label"].unique(),
            feature_names=datos.columns[0:6],
        )

        
        
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
        graficos.append({"tipo": "foto", "base64": img_base64, "text": "Sumario"})


    #Grafico 2
        
        i = 0
        plot = shap.force_plot(
            explainer_2.expected_value[0],
            np.array(
                [
                    list(shap_values_2[0][i]),
                    list(shap_values_2[1][i]),
                    list(shap_values_2[2][i]),
                    list(shap_values_2[3][i]),
                    list(shap_values_2[4][i]),
                ]
            ),
            feature_names=datos.columns[0:6],
            matplotlib=False,
        )
        ploti = str(save_html_return(plot))

        graficos.append(
            {
                "tipo": "interactivo",
                "base64": base64.b64encode(ploti.encode("utf-8")).decode("utf-8"),
                "text": "Probabilidad de clasificar la instancia 0 en cada una de las clases",
            }
        )
        
    #Grafico 3

        plot = shap.force_plot(
            explainer_2.expected_value[1],
            shap_values_2[1][i],
            feature_names=datos.columns[0:6],
            matplotlib=False,
        )
        ploti = str(save_html_return(plot))

        graficos.append(
            {
                "tipo": "interactivo",
                "base64": base64.b64encode(ploti.encode("utf-8")).decode("utf-8"),
                "text": "Probabilidad de clasificar la instancia 0 en la clase 1",
            }
        )
                # Obtén el índice de la primera instancia del conjunto de prueba
        indice_instancia_0 = indices_test[0]

        # Usa el índice para obtener el registro original sin normalizar
        instancia_0_original = datos.iloc[indice_instancia_0]
        print(instancia_0_original)

        

        plot = shap.force_plot(
            explainer_2.expected_value[1],
            shap_values_2[3][i],
            feature_names=datos.columns[0:6],
            matplotlib=False,
        )
        ploti = str(save_html_return(plot))

        graficos.append(
            {
                "tipo": "interactivo",
                "base64": base64.b64encode(ploti.encode("utf-8")).decode("utf-8"),
                "text": "Probabilidad de clasificar la instancia 0 en la clase 3",
            }
        )
      
        plot = shap.force_plot(
            explainer_2.expected_value[1], shap_values_2[0], matplotlib=False
        )
        ploti = str(save_html_return(plot))

        graficos.append(
            {
                "tipo": "interactivo",
                "base64": base64.b64encode(ploti.encode("utf-8")).decode("utf-8"),
                "text": "Probabilidad de clasificar las instancias como la clase 0",
            }
        )

        predict_x = model.predict(X_test)
        classes_x = np.argmax(predict_x, axis=1)

        classes_x
       
        return graficos


def shapFunction():
    if request.method == "POST":
    
        global graficos_recibidos
        graficos_recibidos = None
        try:
            
                detener_agentes()    
                #Credenciales de cada agente
                jid_creds = [
                        ("shap@movim.eu", "hola1234"),
                        ("manageworkflowagent@movim.eu", "hola1234")
                    ]

                if not son_agentes_activos():

                    
                    # Inicializa los agentes aquí
                    agentes_activos["agentShap"] = ReceiverAgent(jid_creds[0][0], jid_creds[0][1])
                    agentes_activos["agentShap"].start(auto_register=True)
                    
                    agentes_activos["agentManager"] = PeriodicSenderagentShap(jid_creds[1][0], jid_creds[1][1])                   
                    agentes_activos["agentManager"].start(auto_register=True)
                    
                    agentes_activos["agentManager"].set("receiver_jid_Shap", jid_creds[0][0])

                    agentes_activos["agentShap"].web.start(hostname="127.0.0.1", port="1005")
                    agentes_activos["agentManager"].web.start(hostname="127.0.0.1", port="1006")           

       
        except ZeroDivisionError as e:
                # Manejar la excepción específica
                print(f"Ocurrió un error: {e}")  
            

        timeout = 180  # Ajustar según sea necesario
        while timeout > 0:
            if graficos_recibidos is not None:
                return jsonify(graficos_recibidos)  # Devolver los gráficos si han sido recibidos
            time.sleep(5)
            timeout -= 5
    
    return None