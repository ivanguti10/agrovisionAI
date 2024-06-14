import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import datetime
import io
from multiprocessing import Pool
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

from keras.preprocessing.image import load_img
import pickle
#######################################################################

groups = {}  # Definir groups en un ámbito global

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

#############################################################################

def get_images_base64_from_cluster(cluster, groups, images_number, verbose=0):
    plot_list = groups[cluster]
    if len(plot_list) > images_number:
        plot_list = np.random.choice(plot_list, images_number, replace=False).tolist()
    
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

async def mostrar_imagenes_cluster(cluster, images_number, groups, verbose=0):
    plot_list, images_base64 = get_images_base64_from_cluster(cluster, groups, images_number, verbose)
    
    size = int(np.sqrt(images_number))
    if size * size < images_number:
        size += 1
    
    plt.figure(figsize=(20, 20))
    
    for ind, image_id in enumerate(plot_list):
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
        plt.title(f"Cluster {cluster}", fontsize=12)
        plt.axis("off")
    
    return images_base64


def agrupar_imagenes_por_cluster(image_ids, kmeans):
    global groups
    for file, cluster in zip(image_ids, kmeans.labels_):
        if cluster not in groups:
            groups[cluster] = [file]
        else:
            groups[cluster].append(file)
    return groups




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

def yuca_cargarModelo():

    if request.method == "POST":
        
        features_file = request.files["archivo"]

        # Guardar temporalmente el archivo recibido
        temp_file_path = "archivo"
        features_file.save(temp_file_path)

        print("Archivo recibido:", features_file.filename)

        # Leer desde el archivo temporal
        loaded_features = pd.read_hdf(temp_file_path, )

        features = np.array(loaded_features['features'].values.tolist()).reshape(-1, 2048)
        image_ids = np.array(loaded_features['image_id'].values.tolist())

        # Clustering
        kmeans = KMeans(n_clusters=5, random_state=22)
        kmeans.fit(features)

        print("Cargando modelo")
        print(kmeans)

        agrupar_imagenes_por_cluster(image_ids, kmeans)
                

        resultado = {"mensaje": "Modelo cargado con éxito"}


        return jsonify(resultado), 200
    
    return None

def cluster1():

    if request.method == "POST":
        
       result = asyncio.run(mostrar_imagenes_cluster(0, 100, groups, verbose=1))
       return result
    
    return None

def cluster2():

    if request.method == "POST":
        
       result = asyncio.run(mostrar_imagenes_cluster(1, 100, groups, verbose=1))
       return result
    
    return None

def cluster3():

    if request.method == "POST":
        
       result = asyncio.run(mostrar_imagenes_cluster(2, 100, groups, verbose=1))
       return result
    
    return None

def cluster4():

    if request.method == "POST":
        
       result = asyncio.run(mostrar_imagenes_cluster(3, 100, groups, verbose=1))
       return result
    
    return None

def cluster5():

    if request.method == "POST":
        
       result = asyncio.run(mostrar_imagenes_cluster(4, 100, groups, verbose=1))
       return result
    
    return None

def compute_similarity_numpy(hashes_all):
    # Calcular la similitud entre todos los hashes usando operaciones vectorizadas de NumPy
    # Esta es una forma simplificada y puede necesitar ajustes basados en la definición exacta de "similitud"
    sims = np.dot(hashes_all, hashes_all.T) / 256
    return sims


import numpy as np
import torch
from flask import jsonify, request
import pickle

import time

def buscarDuplicados():
    start_time = time.time()  # Iniciar el temporizador

    if request.method == "POST":
        # Cargar .npy
        hashes_loaded = request.files["archivo"]
        temp_file_path = "archivo.npy"
        hashes_loaded.save(temp_file_path)
        print("Archivo recibido:", hashes_loaded.filename)

        # Cargar los datos del archivo .npy
        hashes_all = np.load(temp_file_path)



        
        if torch.cuda.is_available():
            hashes_all = torch.Tensor(hashes_all.astype(int)).cuda()
            sims = np.array([(hashes_all[i] == hashes_all).sum(dim=1).cpu().numpy()/256 for i in range(hashes_all.shape[0])])
        else:
            hashes_all = torch.Tensor(hashes_all.astype(int))
            sims = np.array([(hashes_all[i] == hashes_all).sum(dim=1).numpy()/256 for i in range(hashes_all.shape[0])])
       
        image_ids_loaded = request.files["archivo1"]
        temp_file_path = "archivo1.pkl"  # Asegúrate de que la extensión sea .pkl para claridad
        image_ids_loaded.save(temp_file_path)
        print("Archivo recibido:", image_ids_loaded.filename)

        # Cargar los identificadores de las imágenes
        with open(temp_file_path, 'rb') as f:
            image_ids_loaded = pickle.load(f)

        
        indices1 = np.where(sims > 0.9)
        indices2 = np.where(indices1[0] != indices1[1])
        image_ids1 = [image_ids_loaded[i] for i in indices1[0][indices2]]
        image_ids2 = [image_ids_loaded[i] for i in indices1[1][indices2]]
        global dups
        dups = {tuple(sorted([image_ids1,image_ids2])):True for image_ids1, image_ids2 in zip(image_ids1, image_ids2)}
           
        #dups = 2
        #mensaje = 'found %d duplicates' %dups
        #resultado = { "duplicados": mensaje}

        mensaje = 'found %d duplicates' % len(dups)
        resultado = {"mensaje": "Modelo cargado con éxito", "duplicados": mensaje}

       
        end_time = time.time()  # Finalizar el temporizador
        print(f"Tiempo de ejecución: {(end_time - start_time)/60} minutos")
        return jsonify(resultado), 200

    return None

def mostrarDuplicados():
    if request.method == "POST":

        images_base64 = []
        ruta_del_escritorio = Path(obtener_ruta_escritorio())
        #dups = {('1562043567.jpg', '3551135685.jpg'): True, ('2252529694.jpg', '911861181.jpg'): True}
        duplicate_image_ids = sorted(list(dups))  # Asegúrate de que dups sea una lista de listas con IDs de imágenes duplicadas.

        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)  # Ajusta el espacio aquí

        # Leer el archivo train.csv
        train = pd.read_csv(ruta_del_escritorio / 'train.csv')

        for row in range(2):
            for col in range(2):
                try:
                    image_id = duplicate_image_ids[row][col]
                    image_path = str(ruta_del_escritorio / 'train_images' / image_id)
                    label = str(train.loc[train['image_id'] == image_id].label.values[0])

                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Error: No se pudo cargar la imagen en la ruta {image_path}")
                        continue

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Añadir texto a la imagen
                    fuente = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, f"ID: {image_id} - Label: {label}", (10, 30), fuente, 1, (255, 255, 255), 2, cv2.LINE_AA)


                    # Convert image to base64
                    _, buffer = cv2.imencode('.jpg', image)
                    base64_string = base64.b64encode(buffer).decode('utf-8')
                    images_base64.append(base64_string)
                except IndexError:
                    axs[row, col].axis('off')  # Si no hay suficientes imágenes, desactiva el eje

        return images_base64

    return None

def cargarPlanta():
    if request.method == "POST":
        BASE_DIR = Path(__file__).resolve().parent
        prueba_file = request.files["archivo"]
        print("Archivo recibido:", prueba_file.filename)
        prueba = pd.read_csv(prueba_file)
        print(prueba)
        
        # Obtener los nombres de las columnas
        column_names = prueba.columns.tolist()
        
        # Construir un string multilinea con los nombres de las columnas y los valores
        mensaje = "      ".join(column_names) + "\n"
        for index, row in prueba.iterrows():
            mensaje += f"{row['image_id']}  {row['label']}\n"
        
        resultado = {"mensaje": mensaje}
 
        return jsonify(resultado), 200
    return None