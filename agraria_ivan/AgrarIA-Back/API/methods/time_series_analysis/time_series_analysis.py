# Dentro de methods/lime/lime.py
import asyncio
from pathlib import Path
import time
from . import time_series_analysis_bp

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

#Agente que solo envia mensajes a TimeSeriesAnalysis
class PeriodicSenderagentTimeSeriesAnalysis(Agent):
    class InformBehav(TimeoutBehaviour):  
        async def run(self):
            
            graficos = await generate_graficos_TSA()
            msg = Message(to=self.get("receiver_jid_TSA"))
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
    
async def generate_graficos_TSA():
    
        archivos = request.files["archivo"]
        archivos1 = request.files["archivo1"]

        graficos = []
        # Soybean.csv
        
        dataOG = pd.read_csv(archivos)[["Date", "Adj Close", "Volume"]]
        dataOG.head()
        dataOG.isnull().sum()
        data1 = dataOG.fillna(0)
        data1 = data1.drop([0], axis=0)  # La primera fila es la unica de esa semana

        def weekSum(data):
            newData = pd.DataFrame(
                None, columns=["Week", "Date", "Adj Close", "Volume"]
            )
            for i in range(0, len(data) - 5, 5):
                AdjCloseSum = sum(data.iloc[i : i + 5, 1])
                VolumeSum = sum(data.iloc[i : i + 5, 2])
                newData = newData.append(
                    {
                        "Week": ((i / 5) + 1),
                        "Date": data.iloc[i, 0],
                        "Adj Close": AdjCloseSum,
                        "Volume": VolumeSum,
                    },
                    ignore_index=True,
                )
            return newData

        def monthMean(data):
            i = 0
            newData = pd.DataFrame(columns=["Date", "Adj Close", "Volume"])
            while i < len(data):
                count = 0
                AdjCloseSum = 0
                VolumeSum = 0
                month = data.iloc[i, 0].split("-")[1]
                date = "-".join(data.iloc[i, 0].split("-")[0:2])
                while i < len(data) and data.iloc[i, 0].split("-")[1] == month:
                    count += 1
                    AdjCloseSum += data.iloc[i, 1]
                    VolumeSum += data.iloc[i, 2]
                    i += 1
                new_data = pd.DataFrame(
                    {
                        "Date": [date],
                        "Adj Close": [round(AdjCloseSum / count, 2)],
                        "Volume": [round(VolumeSum / count, 2)],
                    }
                )
                newData = pd.concat([newData, new_data], ignore_index=True)
            return newData

        data1 = monthMean(data1)
        data1.to_csv("Soybean1.csv", header=True, index=False)
        dataTemp = data1.drop(columns="Date")
        scaler = MinMaxScaler()
        scaler = scaler.fit(dataTemp)
        dataScaled = scaler.transform(dataTemp)
        month = []
        year = []
        for i in data1["Date"]:
            year.append(int(i.split("-")[0]))
            month.append(int(i.split("-")[1]))
        data1["month"] = month
        data1["year"] = year
        data1.head(10)
        data1Final = data1.drop(columns=["Date", "Adj Close", "Volume"])
        data1Final["Adj Close"] = dataScaled[:, 0]
        data1Final["Volume"] = dataScaled[:, 1]
        data1Final.head()
        X = data1Final.drop(columns="Volume").to_numpy().reshape(-1, 3, 1)
        Y = data1Final["Volume"].to_numpy().reshape(-1, 1)
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(
                    64,
                    activation="relu",
                    input_shape=(X.shape[1], X.shape[2]),
                    return_sequences=True,
                )
            )
        )
        model.add(Bidirectional(LSTM(32, activation="relu", return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(Y.shape[1]))
        model.compile(optimizer="adam", loss="mse")

        # history
        def entrenarModelo():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                  Entrenando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            model.fit(X, Y, epochs=12, batch_size=8, verbose=1)
            
            model.save(
                "Modelos/tsa/tsa1.h5"
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
            model = keras.models.load_model(
                f"{models_path_absolute_path}\\tsa\\tsa1.h5"
            )
            return model


        model = inferencia()


        plt.figure()
        # plt.plot(history.history["loss"], label="Training loss")
        plt.legend()
        data1["Date"] = pd.to_datetime(data1["Date"], format="%Y-%m")
        data1.index = data1.Date
        data1 = data1.iloc[72:, :]
        train = data1[: int(0.8 * (len(data1)))].drop("Date", axis=1)
        valid = data1[int(0.8 * (len(data1))) :].drop("Date", axis=1)
        model = VAR(endog=train.to_numpy())
        model_fit = model.fit()

        # make prediction on validation

        prediction = model_fit.forecast(model.y, steps=len(valid))
        prediction

        cols = data1.columns[1:3]
        pred = pd.DataFrame(index=range(0, len(prediction)), columns=[cols])
        for j in range(0, 2):
            for i in range(0, len(prediction)):
                pred.iloc[i][j] = prediction[i][j]

        # check rmse
        for i in cols:
            print(
                "rmse value for",
                i,
                "is : ",
                np.sqrt(mean_squared_error(pred[i], valid[i])),
            )

        # HistoricalData_1665746821248.csv
        stock_data = pd.read_csv(archivos1)
        closeLast = []
        open = []
        high = []
        low = []
        for index, row in stock_data.iterrows():
            closeLast.append(row["Close/Last"].split("$")[1])
            open.append(row["Open"].split("$")[1])
            high.append(row["High"].split("$")[1])
            low.append(row["Low"].split("$")[1])

        stock_data["Close/Last"] = closeLast
        stock_data["Close/Last"] = stock_data["Close/Last"].astype(float)
        stock_data["Open"] = open
        stock_data["Open"] = stock_data["Open"].astype(float)
        stock_data["High"] = high
        stock_data["High"] = stock_data["High"].astype(float)
        stock_data["Low"] = low
        stock_data["Low"] = stock_data["Low"].astype(float)
        stock_data.head()

        stock_data["average"] = (stock_data["High"] + stock_data["Low"]) / 2
        stock_data.head(2)
        input_feature = stock_data.iloc[:, [2, 6]].values
        input_data = input_feature

        plt.plot(input_feature[:, 0])
        plt.title("Volume of stocks sold")
        plt.xlabel("Time (latest-> oldest)")
        plt.ylabel("Volume of stocks traded")
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
        graficos.append(
            {"tipo": "foto", "base64": img_base64, "text": "Volume of stocks sold"}
        )
        plt.figure()
        plt.plot(input_feature[:, 1], color="blue")
        plt.title("Google Stock Prices")
        plt.xlabel("Time (latest-> oldest)")
        plt.ylabel("Stock Opening Price")
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
        graficos.append(
            {"tipo": "foto", "base64": img_base64, "text": "Google Stock Prices"}
        )

        sc = MinMaxScaler(feature_range=(0, 1))
        input_data[:, 0:2] = sc.fit_transform(input_feature[:, :])

        lookback = 50

        test_size = int(0.3 * len(stock_data))
        X = []
        y = []
        for i in range(len(stock_data) - lookback - 1):
            t = []
            for j in range(0, lookback):
                t.append(input_data[[(i + j)], :])
            X.append(t)
            y.append(input_data[i + lookback, 1])

        X, y = np.array(X), np.array(y)
        X_test = X[: test_size + lookback]
        X = X.reshape(X.shape[0], lookback, 2)
        X_test = X_test.reshape(X_test.shape[0], lookback, 2)

        model2 = Sequential()
        model2.add(LSTM(units=30, return_sequences=True, input_shape=(X.shape[1], 2)))
        model2.add(LSTM(units=30, return_sequences=True))
        model2.add(LSTM(units=30))
        model2.add(Dense(units=1))
        model2.summary()

        model2.compile(optimizer="adam", loss="mean_squared_error")

        def entrenarModelo():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                  Entrenando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            model2.fit(X, y, epochs=200, batch_size=32)
            model2.save(
                "Modelos/tsa/tsa2.h5"
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
            model2 = keras.models.load_model(
                f"{models_path_absolute_path}\\tsa\\tsa2.h5"
            )
            return model2


        model2 = inferencia()


        predicted_value = model2.predict(X_test)
        plt.figure()
        plt.plot(predicted_value, color="red")
        plt.plot(input_data[lookback : test_size + (2 * lookback), 1], color="green")
        plt.title("Opening price of stocks sold")
        plt.xlabel("Time (latest-> oldest)")
        plt.ylabel("Stock Opening Price")
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
        graficos.append(
            {
                "tipo": "foto",
                "base64": img_base64,
                "text": "Opening price of stocks sold",
            }
        )
        """"""
        return graficos

def time_series_analysis():
    if request.method == "POST":
            #-----------------------------------AGENTES----------------------------------------
        global graficos_recibidos
        graficos_recibidos = None
        try:
            
                detener_agentes()    
                #Credenciales de cada agente
                jid_creds = [
                        ("agenttimeSeriesAnalysis@chalec.org", "hola1234"),
                        ("manageworkflowagent@chalec.org", "hola1234")
                    ]
                
                if not son_agentes_activos():
  
                    # Inicializa los agentes aquí
                    agentes_activos["agentTimeSeriesAnalysis"] = ReceiverAgent(jid_creds[0][0], jid_creds[0][1])
                    agentes_activos["agentTimeSeriesAnalysis"].start(auto_register=True)
                    
                    agentes_activos["agentManager"] = PeriodicSenderagentTimeSeriesAnalysis(jid_creds[1][0], jid_creds[1][1])                   
                    agentes_activos["agentManager"].start(auto_register=True)
                    
                    agentes_activos["agentManager"].set("receiver_jid_TSA", jid_creds[0][0])

                    agentes_activos["agentTimeSeriesAnalysis"].web.start(hostname="127.0.0.1", port="1008")
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