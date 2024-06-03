# Dentro de methods/lime/lime.py
import asyncio
import datetime
from pathlib import Path
import time
from . import time_series_prophet_bp

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
import json


#import agentes
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, TimeoutBehaviour
from spade.message import Message

from api import son_agentes_activos, detener_agentes, agentes_activos, graficos_recibidos

#Agente que solo envia mensajes a TimeSeriesProphet
class PeriodicSenderAgentTimeSeriesProphet(Agent):
    class InformBehav(TimeoutBehaviour):  
        async def run(self):
            
            graficos = await generate_graficos()
            msg = Message(to=self.get("receiver_jid"))
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
    
async def generate_graficos():
    
        archivos = request.files["archivo"]
        graficos = []
        # Corn.csv
        CornData = pd.read_csv(archivos)[["Date", "High", "Volume", "Open", "Low"]]

        CornData["Date"] = pd.to_datetime(CornData["Date"], format="%Y-%m-%d")
        Volume = np.array(CornData["Volume"])
        scaler = StandardScaler().fit(Volume.reshape(-1, 1))
        CornData["Volume"] = scaler.transform(Volume.reshape(-1, 1))
        CornData.head()

        dates = pd.date_range(start="7/17/2000", end="06/09/2021")

        dataFilled = pd.DataFrame(None, columns=["ds", "y", "Volume", "Open", "Low"])
        dataFilled["ds"] = dates
        dataFilled.head()

        for i in range(len(CornData)):
            iter = dataFilled[dataFilled["ds"] == CornData["Date"][i]].index.values[0]
            dataFilled["y"][iter] = CornData["High"][i]
            dataFilled["Volume"][iter] = CornData["Volume"][i]
            dataFilled["Open"][iter] = CornData["Open"][i]
            dataFilled["Low"][iter] = CornData["Low"][i]

            # Recorte desde 2006
        dataFilled = dataFilled.iloc[1994:, :]

        plt.figure(figsize=(500, 250))
        figures, axes = plt.subplots(nrows=2, ncols=2)
        axes[0, 0].plot(dataFilled["ds"], dataFilled["y"], label="High")
        axes[0, 1].plot(dataFilled["ds"], dataFilled["Volume"], label="Volume")
        axes[1, 0].plot(dataFilled["ds"], dataFilled["Open"], label="Open")
        axes[1, 1].plot(dataFilled["ds"], dataFilled["Low"], label="Low")
        plt.legend()
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
        graficos.append({"tipo": "foto", "base64": img_base64, "text": ""})

        pyo.init_notebook_mode(connected=True)

        m = Prophet()

        def entrenarModelo():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                  Entrenando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            m.fit(dataFilled)
            dump(
                m,
                "Modelos/tsp/tsp0.joblib",
            )

        def inferencia():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                    Cargando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            current_dir = Path(__file__).parent  
            models_path = current_dir / '../../Modelos'
            models_path_absolute_path = models_path.resolve()
            model = load(
                f"{models_path_absolute_path}\\tsp\\tsp0.joblib"
            )
            
            return model

        m = inferencia()


        future = m.make_future_dataframe(periods=365)
        future.tail()

        forecast = m.predict(future)
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()

        fig1 = m.plot(forecast)
        fig1_img_buf = BytesIO()
        fig1.savefig(fig1_img_buf, format="png")
        fig1_img_buf.seek(0)
        fig1_img_base64 = base64.b64encode(fig1_img_buf.read()).decode("utf-8")
        graficos.append(
            {
                "tipo": "foto",
                "base64": fig1_img_base64,
                "text": "Análisis univariante de High",
            }
        )

        fig2 = m.plot_components(forecast)
        fig2_img_buf = BytesIO()
        fig2.savefig(fig2_img_buf, format="png")
        fig2_img_buf.seek(0)
        fig2_img_base64 = base64.b64encode(fig2_img_buf.read()).decode("utf-8")
        graficos.append(
            {
                "tipo": "foto",
                "base64": fig2_img_base64,
                "text": "Vemos que la predicción se ajusta decentemente a los datos, sin embargo la predicción fiera de la muestra no parece que represente el comportamiento futuro real de la serie.",
            }
        )

        # Separate in Train and test
        dataTrain = dataFilled.iloc[0 : (7267 - 1994), :]
        dataTest = dataFilled.iloc[(7267 - 1994) :, :].reset_index()
        dataTest = dataTest.drop(columns="index")

        # Fill the NaNs of test
        dataTest = dataTest.fillna(method="bfill", axis=0)

        model = Prophet(interval_width=0.9)
        model.add_regressor("Volume", standardize=True)
        model.add_regressor("Open", standardize=False)
        model.add_regressor("Low", standardize=False)

        def entrenarModelo():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                  Entrenando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            model.fit(dataTrain)
            dump(
                model,
                "Modelos/tsp/tsp1.joblib",
            )

        def inferencia():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                    Cargando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            current_dir = Path(__file__).parent  
            models_path = current_dir / '../../Modelos'
            models_path_absolute_path = models_path.resolve()
            model = load(
                f"{models_path_absolute_path}\\tsp\\tsp1.joblib"
            )
            return model

        model = inferencia()


        forecast = model.predict(dataTest)
        forecast = forecast[["ds", "yhat"]]
        forecast.head()

        finalData = pd.concat((forecast["yhat"], dataTest), axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(finalData["ds"], finalData["y"], color="red", label="actual")
        plt.plot(finalData["ds"], finalData["yhat"], color="blue", label="forecast")
        plt.legend()
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
        graficos.append(
            {"tipo": "foto", "base64": img_base64, "text": "Analisis usando regresores"}
        )

        model2 = Prophet(interval_width=0.9)
        model2.add_regressor("Volume", standardize=True)

        def entrenarModelo():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                  Entrenando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            model2.fit(dataTrain)
            dump(
                model2,
                "Modelos/tsp/tsp2.joblib",
            )

        def inferencia():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                    Cargando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            current_dir = Path(__file__).parent  
            models_path = current_dir / '../../Modelos'
            models_path_absolute_path = models_path.resolve()
            model = load(
                f"{models_path_absolute_path}\\tsp\\tsp2.joblib"
            )
            return model

        model2 = inferencia()


        forecast2 = model2.predict(dataTest)
        forecast2 = forecast2[["ds", "yhat"]]

        finalData2 = pd.concat((forecast2["yhat"], dataTest), axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(finalData2["ds"], finalData2["y"], color="red", label="actual")
        plt.plot(finalData2["ds"], finalData2["yhat"], color="blue", label="forecast")
        plt.legend()
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
        graficos.append(
            {
                "tipo": "foto",
                "base64": img_base64,
                "text": "Modelo sin usar las otras variables de precios como regresores",
            }
        )

        dataTrain.columns = ["ds", "High", "y", "Open", "Low"]
        dataTest.columns = ["ds", "High", "y", "Open", "Low"]

        model3 = Prophet(interval_width=0.9)
        model3.add_regressor("High", standardize=False)
        model3.add_regressor("Open", standardize=False)
        model3.add_regressor("Low", standardize=False)

        def entrenarModelo():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                  Entrenando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            model3.fit(dataTrain)
            dump(
                model3,
                "Modelos/tsp/tsp3.joblib",
            )

        def inferencia():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                    Cargando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            current_dir = Path(__file__).parent  
            models_path = current_dir / '../../Modelos'
            models_path_absolute_path = models_path.resolve()
            model = load(
                f"{models_path_absolute_path}\\tsp\\tsp3.joblib"
            )
            return model

        model3 = inferencia()


        forecast3 = model3.predict(dataTest)
        forecast3 = forecast3[["ds", "yhat"]]

        finalData3 = pd.concat((forecast3["yhat"], dataTest), axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(finalData3["ds"], finalData3["y"], color="red", label="actual")
        plt.plot(finalData3["ds"], finalData3["yhat"], color="blue", label="forecast")
        plt.legend()
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
        graficos.append(
            {
                "tipo": "foto",
                "base64": img_base64,
                "text": "Probando ahora con Volume como variable respuesta",
            }
        )

        dataTrain.columns = ["ds", "y", "Volume", "Open", "Low"]
        dataTest.columns = ["ds", "y", "Volume", "Open", "Low"]

        model4 = Prophet(interval_width=0.9)
        model4.add_regressor("Volume", standardize=True)
        model4.add_seasonality(name="monthly", period=30, fourier_order=3)

        def entrenarModelo():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                  Entrenando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            model4.fit(dataTrain)
            dump(
                model4,
                "Modelos/tsp/tsp4.joblib",
            )

        def inferencia():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                    Cargando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            current_dir = Path(__file__).parent  
            models_path = current_dir / '../../Modelos'
            models_path_absolute_path = models_path.resolve()
            model = load(
                f"{models_path_absolute_path}\\tsp\\tsp4.joblib"
            )
            return model

        model4 = inferencia()


        forecast4 = model4.predict(dataTest)
        forecast4 = forecast4[["ds", "yhat"]]

        finalData4 = pd.concat((forecast4["yhat"], dataTest), axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(finalData4["ds"], finalData4["y"], color="red", label="actual")
        plt.plot(finalData4["ds"], finalData4["yhat"], color="blue", label="forecast")
        plt.legend()
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
        graficos.append(
            {
                "tipo": "foto",
                "base64": img_base64,
                "text": "Añadiendo la estacionalidad",
            }
        )

        return graficos       


def time_series_prophet():
    if request.method == "POST":
        
        try:
                
                detener_agentes()    
                #Credenciales de cada agente
                jid_creds = [
                        ("agentTimeSeriesProphet@chalec.org", "hola1234"),
                        ("manageworkflowagent@chalec.org", "hola1234")
                    ]

                global graficos_recibidos
                graficos_recibidos = None

                if not son_agentes_activos():

                    
                    # Inicializa los agentes aquí
                    agentes_activos["agentTimeSeriesProphet"] = ReceiverAgent(jid_creds[0][0], jid_creds[0][1])
                    agentes_activos["agentTimeSeriesProphet"].start(auto_register=True)
                    
                    agentes_activos["agentManager"] = PeriodicSenderAgentTimeSeriesProphet(jid_creds[1][0], jid_creds[1][1])                   
                    agentes_activos["agentManager"].start(auto_register=True)
                    
                    agentes_activos["agentManager"].set("receiver_jid", jid_creds[0][0])

                    agentes_activos["agentTimeSeriesProphet"].web.start(hostname="127.0.0.1", port="1009")
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
        detener_agentes() 


    return None

