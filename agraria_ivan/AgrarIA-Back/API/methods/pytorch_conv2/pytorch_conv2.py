import time
from . import pytorch_conv2_bp
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

import tensorflow as tf
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import shap
import numpy as np

class Data(Dataset):
    def __init__(self, data, transform=None):
        self.dat = list(data.values)
        self.transform = transform
        label = []
        image = []
        for i in self.dat:
            label.append(i[-1])
            image.append(i[:-1])
        self.labels = np.asarray(label)
        self.images = np.array(image).reshape(-10, 3, 60, 60).astype("float32")

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )

        self.fc1 = nn.Linear(in_features=864, out_features=400)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=400, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


#Agente que solo envia mensajes a pytorch_conv2
class PeriodicSenderagentPytorch(Agent):
    class InformBehav(TimeoutBehaviour):  
        async def run(self):
            
            graficos = await generate_graficos_pytorch()
            msg = Message(to=self.get("receiver_jid_pytorch"))
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

async def generate_graficos_pytorch():

        
        # Potatoes.csv
        archivos = request.files["archivo"]
        graficos = []
        imagenes = pd.read_csv(archivos, header=None)
        print(imagenes.shape)
        imagenes["label"] = imagenes[10800]
        imagenes = imagenes.drop(columns=10800)
        # print(imagenes)
        print(imagenes.size)
        print(np.__version__)
        X_train, X_test, y_train, y_test = train_test_split(
            imagenes.iloc[:, :-1], imagenes.iloc[:, -1], test_size=1 / 3, random_state=1
        )

        le = preprocessing.LabelEncoder()

        le.fit(imagenes["label"])
        classes = list(le.classes_)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

        X_train["label"] = y_train
        X_test["label"] = y_test

        train_set = Data(X_train)
        test_set = Data(X_test)

        train_loader = DataLoader(train_set)
        test_loader = DataLoader(test_set)

        model = CNN()
        # device = torch.device("cuda:0")
        # model.to(device)

        error = nn.CrossEntropyLoss()

        learning_rate = 0.001
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(10):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                inputs = inputs.to(torch.float32)
                # forward + backward + optimize
                outputs = model(inputs.reshape(-1, 3, 60, 60))
                labels = labels.type(torch.LongTensor)
                loss = criterion(outputs, torch.tensor(labels))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

        print("Finished Training")

        correct = 0
        total = len(X_test)
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                # calculate outputs by running images through the network
                images = inputs.to(torch.float32)
                outputs = model(images.reshape(-1, 3, 60, 60))
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                # total += torch.tensor(labels).size(0)
                correct += sum(predicted == labels)

                # correct += int(predicted == labels[0])

        print(
            "Accuracy of the network on the test images: %d %%"
            % (100 * correct / total)
        )
        accuracy = "Accuracy of the network on the test images: %d %%" % (
            100 * correct / total
        )
        batch = next(iter(test_loader))
        batch[0].shape

        batch = next(iter(test_loader))
        batch2 = next(iter(test_loader))

        images, _ = batch
        images = images.view(-1, 3, 60, 60).to(torch.float32)

        test_images = images

        e = shap.DeepExplainer(
            model,
            torch.Tensor(X_train.iloc[0:500, :-1].values)
            .view(-1, 3, 60, 60)
            .to(torch.float32),
        )
        shap_values = e.shap_values(test_images)

        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

        shap.image_plot(shap_numpy, test_numpy)

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Convierte la imagen en base64
        image_base64 = base64.b64encode(buffer.read()).decode()

        graficos.append({"tipo": "foto", "base64": image_base64, "text": accuracy})

        return graficos

def pytorch_conv2():
  
    if request.method == "POST":
                    #-----------------------------------AGENTES----------------------------------------
            global graficos_recibidos
            graficos_recibidos = None
            try:
                
                    detener_agentes()    
                    #Credenciales de cada agente
                    jid_creds = [
                            ("pytorch_conv2@chalec.org", "hola1234"),
                            ("manageworkflowagent@chalec.org", "hola1234")
                        ]

                    if not son_agentes_activos():

                        
                        # Inicializa los agentes aquí
                        agentes_activos["pytorch_conv2"] = ReceiverAgent(jid_creds[0][0], jid_creds[0][1])
                        agentes_activos["pytorch_conv2"].start(auto_register=True)
                        
                        agentes_activos["agentManager"] = PeriodicSenderagentPytorch(jid_creds[1][0], jid_creds[1][1])                   
                        agentes_activos["agentManager"].start(auto_register=True)
                        
                        agentes_activos["agentManager"].set("receiver_jid_pytorch", jid_creds[0][0])

                        agentes_activos["pytorch_conv2"].web.start(hostname="127.0.0.1", port="1004")
                        agentes_activos["agentManager"].web.start(hostname="127.0.0.1", port="1006")           

        
            except ZeroDivisionError as e:
                    # Manejar la excepción específica
                    print(f"Ocurrió un error: {e}")  
                

            timeout = 300  # Ajustar según sea necesario
            while timeout > 0:
                if graficos_recibidos is not None:
                    return jsonify(graficos_recibidos)  # Devolver los gráficos si han sido recibidos
                time.sleep(5)  # Espera un breve periodo antes de volver a verificar
                timeout -= 5

    return None