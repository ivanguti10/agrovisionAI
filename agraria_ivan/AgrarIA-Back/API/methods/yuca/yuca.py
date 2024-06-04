import asyncio
import base64
import datetime
import io
from pathlib import Path
import time

from django.http import JsonResponse
import requests
from . import yuca_bp

# Preliminaries
import os
from pathlib import Path
import glob
from tqdm import tqdm
tqdm.pandas()
import json
import pandas as pd
import numpy as np

## Image hash
#import imagehash

# Visuals and CV2
import seaborn as sn
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf


# albumentations for augs
#import albumentations
#from albumentations.pytorch.transforms import ToTensorV2

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from flask import Flask, request, jsonify
import pandas as pd
import io

#torch
import torch
#import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

from flask import Flask, app, jsonify, request

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


#######################################################################

#Agente que solo envia mensajes a TimeSeriesProphet
class PeriodicSenderAgentYuca(Agent):
    class InformBehav(TimeoutBehaviour):  
        async def run(self):
            
            graficos = await mostrar_healthies()
            msg = Message(to=self.get("receiver_jid"))
            msg.body = json.dumps(graficos, indent=4, sort_keys=True, default=str)  # Convertir los gráficos a formato JSON
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

def obtener_ruta_escritorio():
    onedrive_path = os.environ.get('ONEDRIVE')
    if not onedrive_path:
        raise EnvironmentError("No se encontró la variable de entorno 'ONEDRIVE'")
    desktop_path = Path(onedrive_path) / 'Escritorio' / 'proyecto'
    return desktop_path

# Ejemplo de uso:
ruta_del_escritorio = obtener_ruta_escritorio()
print(f"La ruta del escritorio es: {ruta_del_escritorio}")

def get_images_base64(class_id, images_number, verbose=0):
    plot_list = train[train["label"] == class_id].sample(images_number)['image_id'].tolist()
    if verbose:
        print(plot_list)

    images_base64 = []

    for image_id in plot_list:
        if not image_id.endswith(".jpg"):
            image_id += ".jpg"


        ruta_del_escritorio = Path(obtener_ruta_escritorio())
        image_path = str(ruta_del_escritorio / 'train_images' / image_id)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: No se pudo cargar la imagen en la ruta {image_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', image)
        base64_string = base64.b64encode(buffer).decode('utf-8')
        images_base64.append(base64_string)

    return plot_list, images_base64

async def mostrar_imagenes(class_id, label, images_number, verbose=0):
    global train
    global BASE_DIR

    BASE_DIR = Path(__file__).resolve().parent
    train_file = request.files["archivo"]

    print("Archivo recibido:", train_file.filename)
    train = pd.read_csv(train_file)

    plot_list, images_base64 = get_images_base64(class_id, images_number, verbose)

    labels = [label for _ in range(len(plot_list))]
    size = int(np.sqrt(images_number))
    if size * size < images_number:
        size += 1

    plt.figure(figsize=(20, 20))

    for ind, (image_id, label) in enumerate(zip(plot_list, labels)):
        if not image_id.endswith(".jpg"):
            image_id += ".jpg"

        ruta_del_escritorio = Path(obtener_ruta_escritorio())
        image_path = str(ruta_del_escritorio / 'train_images' / image_id)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: No se pudo cargar la imagen en la ruta {image_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(size, size, ind + 1)
        plt.imshow(image)
        plt.title(label, fontsize=12)
        plt.axis("off")

    return images_base64

def mostrar_healthies():
    return mostrar_imagenes(class_id=4, label='Healthy', images_number=6, verbose=1)

#######################################################
    

def yuca():

    if request.method == "POST":
        
        try:
                
                detener_agentes()    
                #Credenciales de cada agente
                jid_creds = [
                        ("yuca@chalec.org", "hola1234"),
                        ("manageworkflowagent@chalec.org", "hola1234")
                    ]

                global graficos_recibidos
                graficos_recibidos = None

                if not son_agentes_activos():

                    
                    # Inicializa los agentes aquí
                    agentes_activos["yuca"] = ReceiverAgent(jid_creds[0][0], jid_creds[0][1])
                    agentes_activos["yuca"].start(auto_register=True)
                    
                    agentes_activos["agentManager"] = PeriodicSenderAgentYuca(jid_creds[1][0], jid_creds[1][1])                   
                    agentes_activos["agentManager"].start(auto_register=True)
                    
                    agentes_activos["agentManager"].set("receiver_jid", jid_creds[0][0])

                    agentes_activos["yuca"].web.start(hostname="127.0.0.1", port="1002")
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


def yuca_CBB():

    if request.method == "POST":
        
        result = asyncio.run(mostrar_imagenes(class_id=0, label='CBB', images_number=6, verbose=1))
        return result

    return None
  

def yuca_CGM():

    if request.method == "POST":
        
        result = asyncio.run(mostrar_imagenes(class_id=2, label='CGM', images_number=6, verbose=1))
        return result

    return None

def yuca_CDM():

    if request.method == "POST":
        
        result = asyncio.run(mostrar_imagenes(class_id=3, label='CDM', images_number=6, verbose=1))
        return result

    return None

def yuca_CBSD():

    if request.method == "POST":
        
        result = asyncio.run(mostrar_imagenes(class_id=1, label='CBSD', images_number=6, verbose=1))
        return result

    return None

def yucamodeloIA():
    if request.method == "POST":
        try:
            archivo1 = request.files["archivo"]
            features = pd.read_hdf(archivo1, 'healthy')
            summary = features.describe().to_dict()
            print(summary)
            return jsonify({"message": "Modelo cargado correctamente"}), 200
        except Exception as e:
            return jsonify({"message": f"Error al cargar el modelo: {str(e)}"}), 500

    return jsonify({"message": "Método no permitido"}), 405