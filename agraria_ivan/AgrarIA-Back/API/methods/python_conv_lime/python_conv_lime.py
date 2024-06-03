import time
from flask import jsonify, request
from matplotlib.colors import ListedColormap
import matplotlib

matplotlib.use("Agg")
import tensorflow as tefo

tefo.compat.v1.disable_v2_behavior()
tefo.compat.v1.keras.backend.get_session
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


#!pip install skimage
from skimage.segmentation import mark_boundaries
import lime

#!pip install opencv-python
import cv2
from PIL import Image
import io
from matplotlib import cm
import PIL
import keras
import numpy as numpy
from keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as ploty
from lime import lime_image
from keras.preprocessing import image
from torchvision import transforms
import os

from . import python_conv_lime_bp

import datetime
import json

#import agentes
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, TimeoutBehaviour
from spade.message import Message

from pathlib import Path



archivos = None
graficos_recibidos = None

                
#Agente que solo envia mensajes a python_conv_lime
class PeriodicSenderagentPython_conv_lime(Agent):
    class InformBehav(TimeoutBehaviour):  
        async def run(self):
            
            graficos = await generate_graficos_python_conv_lime()
            msg = Message(to=self.get("receiver_jid_python_conv_lime"))
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
    
async def generate_graficos_python_conv_lime():
    
        graficos = []
        # Obtiene el directorio donde se encuentra archivo.py

        current_dir = Path(__file__).parent 
        plant_village_path = current_dir / '../../../PlantVillage'
        plant_village_absolute_path = plant_village_path.resolve()
        
        datos_train = tefo.keras.utils.image_dataset_from_directory(f"{plant_village_absolute_path}",
            labels="inferred",
            label_mode="int",
            batch_size=2,
            validation_split=0.25,
            subset="training",
            # shuffle=False,
            seed=0,
        )
        
        datos_val = tefo.keras.preprocessing.image_dataset_from_directory(f"{plant_village_absolute_path}",
            labels="inferred",
            label_mode="int",
            batch_size=64,
            validation_split=0.25,
            subset="validation",
            # shuffle=True,
            seed=0,
        )

        model = tefo.keras.Sequential(
            [
                tefo.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255),
                tefo.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(64, 256, 256, 3)),
                tefo.keras.layers.MaxPooling2D((2, 2)),
                tefo.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tefo.keras.layers.MaxPooling2D((2, 2)),
                tefo.keras.layers.Conv2D(192, (3, 3), activation="relu"),
                tefo.keras.layers.MaxPooling2D((2, 2)),
                tefo.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tefo.keras.layers.MaxPooling2D((2, 2)),
                tefo.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tefo.keras.layers.MaxPooling2D((2, 2)),
                tefo.keras.layers.Flatten(),
                tefo.keras.layers.Dense(64, activation="relu"),
                tefo.keras.layers.Dense(15, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tefo.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        def entrenarModelo():
            print("╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮")
            print("┃                                                                                    ┃")
            print("┃                                  Entrenando modelo                                 ┃")
            print("┃                                                                                    ┃")
            print("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯")
            model.fit(datos_train, validation_data=datos_val, epochs=1)
            model.save(
                "Modelos/conv_lime.h5"
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
            model = tefo.keras.models.load_model(
                f"{models_path_absolute_path}/conv_lime.h5"
            )
            return model

        # IF para crear modelo o cargarlo
        model = inferencia()

        class_names = datos_train.class_names

        img = PIL.Image.open(archivo)

        # Creacion del explainer LIME para la imagen que hemos seleccionado
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image.img_to_array(img).astype("double"), model.predict, top_labels=1
        )

        # INTRODUCIMOS UN BATCH PARA QUE LO ANALICE Y EXTRAEMOS LOS RESULTADOS CORRESPONDIENTES

        x = image.img_to_array(img)
        x = numpy.expand_dims(x, axis=0)
        x = preprocess_input(x)
        n = numpy.zeros((10, 256, 256, 3))
        n[0] = x[0]
        temp, mask = explanation.get_image_and_mask(
            numpy.argmax(model.predict(n)[0]),
            positive_only=False,
            negative_only=True,
            hide_rest=True,
        )
        ploty.figure()
        ploty.imshow(img)
        ploty.axis("off")
        buffer = BytesIO()
        ploty.savefig(
            buffer, format="png", bbox_inches="tight", pad_inches=0, transparent=True
        )
        buffer.seek(0)

        # Convierte la imagen en base64
        image_base64 = base64.b64encode(buffer.read()).decode()
        background = Image.open(io.BytesIO(base64.b64decode(image_base64)))

        graficos.append(
            {"tipo": "foto", "base64": image_base64, "text": "Imagen normal"}
        )

        print(class_names[numpy.argmax(model.predict(n)[0])])

        # REPRESENTACIÓN DE LAS PARTES DE LA IMAGEN QUE AFECTAN NEGATIVAMENTE A LA CLASIFICACION PRINCIPAL
        ploty.figure()
        cmap = ListedColormap(["white", "black"])
        ploty.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        ploty.axis("off")
        buffer = BytesIO()
        ploty.savefig(
            buffer, format="png", bbox_inches="tight", pad_inches=0, transparent=True
        )
        buffer.seek(0)

        # Convierte la imagen en base64
        image_base64 = base64.b64encode(buffer.read()).decode()
        overlay = Image.open(io.BytesIO(base64.b64decode(image_base64)))

        overlay = overlay.convert("RGBA")
        background = background.convert("RGBA")
        overlay = overlay.resize(background.size)

        transparent_overlay = overlay.copy()
        for x in range(overlay.width):
            for y in range(overlay.height):
                # Reemplazar el punto con una versión más transparente
                current_color = overlay.getpixel((x, y))
                # Asumiendo que el fondo de la imagen superior es gris y queremos hacerlo transparente
                if current_color[:3] == (128, 128, 128):
                    transparent_overlay.putpixel(
                        (x, y), (255, 255, 255, 0)
                    )  # Hacer el pixel completamente transparente
                else:
                    transparent_overlay.putpixel(
                        (x, y), current_color[:-1] + (128,)
                    )  # 128 es el nivel de transparencia

        # Superponer la imagen transparente sobre el fondo
        buffer = BytesIO()
        ploty.imshow(Image.alpha_composite(background, transparent_overlay))
        ploty.axis("off")
        ploty.savefig(
            buffer, format="png", bbox_inches="tight", pad_inches=0, transparent=True
        )
        buffer.seek(0)
        image_combined_base64 = base64.b64encode(buffer.read()).decode()

        graficos.append(
            {
                "tipo": "foto",
                "base64": image_base64,
                "combined_base64": image_combined_base64,
                "text": "Representación de las partes de la imagen que afectan negativamente a la clasificación principal.",
            }
        )

        temp, mask = explanation.get_image_and_mask(
            numpy.argmax(model.predict(n)[0]),
            positive_only=True,
            negative_only=False,
            hide_rest=True,
        )

        # REPRESENTACIÓN DE LAS PARTES DE LA IMAGEN QUE AFECTAN POSITIVAMENTE A LA CLASIFICACION PRINCIPAL
        ploty.figure()
        cmap = ListedColormap(["white", "black"])
        ploty.imshow(
            mark_boundaries(temp / 2 + 0.5, mask)
        )  # impacta positivamente a la clasificación
        ploty.axis("off")
        buffer = BytesIO()
        ploty.savefig(
            buffer, format="png", bbox_inches="tight", pad_inches=0, transparent=True
        )
        buffer.seek(0)

        # Convierte la imagen en base64
        image_base64 = base64.b64encode(buffer.read()).decode()

        overlay = Image.open(io.BytesIO(base64.b64decode(image_base64)))

        overlay = overlay.convert("RGBA")
        background = background.convert("RGBA")
        overlay = overlay.resize(background.size)

        transparent_overlay = overlay.copy()
        for x in range(overlay.width):
            for y in range(overlay.height):
                # Reemplazar el punto con una versión más transparente
                current_color = overlay.getpixel((x, y))
                # Asumiendo que el fondo de la imagen superior es gris y queremos hacerlo transparente
                if current_color[:3] == (128, 128, 128):
                    transparent_overlay.putpixel(
                        (x, y), (255, 255, 255, 0)
                    )  # Hacer el pixel completamente transparente
                else:
                    transparent_overlay.putpixel(
                        (x, y), current_color[:-1] + (128,)
                    )  # 128 es el nivel de transparencia

        # Superponer la imagen transparente sobre el fondo
        buffer = BytesIO()
        ploty.imshow(Image.alpha_composite(background, transparent_overlay))
        ploty.axis("off")
        ploty.savefig(
            buffer, format="png", bbox_inches="tight", pad_inches=0, transparent=True
        )
        buffer.seek(0)
        image_combined_base64 = base64.b64encode(buffer.read()).decode()

        graficos.append(
            {
                "tipo": "foto",
                "base64": image_base64,
                "combined_base64": image_combined_base64,
                "text": "Representación de las partes de la imagen que afectan positivamente a la clasificación principal.",
            }
        )

        EFFICIENTNET_VERSION = {
            "B0": {"model": tefo.keras.applications.EfficientNetB0, "img_size": 224},
            "B1": {"model": tefo.keras.applications.EfficientNetB1, "img_size": 240},
            "B2": {"model": tefo.keras.applications.EfficientNetB2, "img_size": 260},
            "B3": {"model": tefo.keras.applications.EfficientNetB3, "img_size": 300},
            "B4": {"model": tefo.keras.applications.EfficientNetB4, "img_size": 380},
            "B5": {"model": tefo.keras.applications.EfficientNetB5, "img_size": 456},
            "B6": {"model": tefo.keras.applications.EfficientNetB6, "img_size": 528},
            "B7": {"model": tefo.keras.applications.EfficientNetB7, "img_size": 600},
        }

        """
        Implementation of Local interpretable model-agnostic explanations.
        Source: Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin.
        "Why should I trust you?: Explaining the predictions of any classifier."
        Proceedings of the 22nd ACM SIGKDD international conference on knowledge
        discovery and data mining. ACM (2016).
        """
        import tensorflow as tf
        import numpy as np
        from typing import Tuple
        from sklearn.metrics import pairwise_distances
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        import skimage
        import matplotlib.pyplot as plt

        EXPLAINABLE_MODELS = {
            'linear_regression': LinearRegression,
            'decision_tree_regressor': DecisionTreeRegressor
        }


        class LIME:

            def __init__(self, image: tf.Tensor, model: tf.keras.Model, random_seed: int = 9):
                """
                Parameters
                ----------
                image: tf.Tensor; Image for which the explanation should be made
                model: tf.keras.Model; Base model
                """
                self.image = image
                self.model = model
                self.random_seed = random_seed
                self.super_pixels, self.super_pixel_count = self.create_super_pixels()
                self.perturbation_vectors = self.generate_pertubation_vectors()

            def create_super_pixels(
                    self, kernel_size: int = 6, max_dist: int = 1000, ratio: float = 0.2
            ) -> Tuple[np.ndarray, int]:
                """
                Parameters
                ----------
                kernel_size, max_dist, ratio: parameters for skimage.segmentation.quickshift function.
                See https://scikit-image.org/docs/stable/api/skimage.segmentation.html for more info
                Returns
                -------
                super_pixels: np.ndarray (shape==self.image.shape); Contains
                integers to which super pixel area the location belongs
                super_pixel_count: int; total number of different superpixel areas
                """
                super_pixels = skimage.segmentation.quickshift(
                    self.image, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio
                )
                super_pixel_count = len(np.unique(super_pixels))
                return super_pixels, super_pixel_count

            def plot_super_pixel_boundary(self):
                """ Plots the boundaries of the superpixel areas """
                super_pixel_boundaries = skimage.segmentation.mark_boundaries(
                    np.array(self.image).astype(int), self.super_pixels
                )
                plt.imshow(super_pixel_boundaries)
                plt.title('Superpixel boundaries')

            def generate_pertubation_vectors(self, num_perturbations: int = 100) -> np.ndarray:
                """
                Generates a number of perturbation vectors. These are binary vectors of length
                num_super_pixels, which define if a superpixel is perturbed
                Parameters
                ----------
                num_perturbations: int; total number of perturbations
                Returns
                -------
                np.ndarray (shape=(num_perturbations, super_pixel_count); binary array defining if a
                superpixel area should be perturbed
                """
                if self.random_seed is not None:
                    np.random.seed(self.random_seed)
                return np.random.binomial(1, 0.5, size=(num_perturbations, self.super_pixel_count))

            def predict_perturbed_images(self) -> np.ndarray:
                """
                Generates predictions for all perturbed_images
                Returns
                -------
                np.ndarray (shape=(num_perturbations, num_output_classes)); contains predictions for all
                perturbed images
                """
                perturbed_images = self.create_perturbed_images()
                print(perturbed_images)
                #return np.array(self.model(perturbed_images))
                return "HOLA"

            def create_perturbed_images(self) -> np.ndarray:
                """ Creates perturbed images based on all pertubation vectors """
                self.generate_pertubation_vectors()
                return np.apply_along_axis(
                    lambda x: self._create_perturbed_image(x), 1, self.perturbation_vectors
                )

            def _create_perturbed_image(self, perturbation_vector: np.ndarray) -> np.ndarray:
                """
                Creatas a single perturbed image
                Parameters
                ----------
                perturbation_vector; np.ndarray (shape=(num_perturbations, super_pixel_count); binary array
                defining if a superpixel area should be perturbed
                Returns
                -------
                np.ndarray (shape==self.image.shape); contains original image info or perturbed image
                dependent on the perturbation_vector
                """
                perturbation_mask = np.isin(self.super_pixels, np.argwhere(perturbation_vector == 1))
                return np.where(np.expand_dims(perturbation_mask, -1), self.image, 0)

            def plot_perturbed_image(self):
                """ Plots a single perturbed image """
                if self.perturbation_vectors is None:
                    self.generate_pertubation_vectors()

                if self.random_seed is not None:
                    np.random.seed(self.random_seed)
                idx = np.random.randint(len(self.perturbation_vectors))
                perturbed_img = self._create_perturbed_image(self.perturbation_vectors[idx])

                plt.imshow(perturbed_img.astype(int))
                plt.title('Perturbed Image')

            def calculate_perturbation_weights(self, kernel_width: float = 0.25) -> np.ndarray:
                """
                Calculates the perturbation weights. First the distance between the perturbed images
                and the original image is calculated. A kernel function is used to map these distances
                to weights. The smaller the distance to the original image, the larger the weight. The
                intuition behind this is that if a perturbed image is very close to the original image,
                (let's say only 1 perturbed superpixel area), there is a lot of information in this
                sample, as it tells a lot on the 1 perturbed superpixel area importance.
                Parameters
                ----------
                kernel_width: float; defines the width of the kernel that is used
                Returns
                -------
                np.ndarray (shape=(num_perturbations,)); weights for each perturbation
                """
                non_perturbed_vector = np.ones((1, self.super_pixel_count))
                distances = pairwise_distances(
                    self.perturbation_vectors, non_perturbed_vector, metric='cosine'
                )
                return np.squeeze(np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2)))

            def _fit_explainable_model(
                    self,
                    predictions: np.ndarray,
                    weights: np.ndarray,
                    explainable_model_type: str = 'decision_tree_regressor'
            ) -> np.ndarray:
                """
                Function to fit an exmplainable model and define the importance of each superpixel area
                Parameters
                ----------
                predictions: np.ndarray (shape=(num_perturbations, num_output_classes)); contains
                predictions for all perturbed images
                weights: np.ndarray (shape=(num_perturbations,)); weights for each perturbation
                explainable_model_type: str containing the type of explainable model
                Returns
                -------
                feature_importances: np.ndarray (shape=(super_pixel_count,)); importance of each superpixel
                for predicting a specific class
                """
                if explainable_model_type not in EXPLAINABLE_MODELS.keys():
                    raise ValueError(
                        f"Please specify one of the following model_types: {EXPLAINABLE_MODELS.keys()}"
                    )

                model = EXPLAINABLE_MODELS[explainable_model_type]()
                model.fit(X=self.perturbation_vectors, y=predictions, sample_weight=weights)
                if 'regression' in explainable_model_type:
                    feature_importance = model.coef_
                elif 'tree' in explainable_model_type:
                    feature_importance = model.feature_importances_
                else:
                    raise ValueError(
                        f"Please specify one of the following model_types: {EXPLAINABLE_MODELS.keys()}"
                    )

                return feature_importance

            def plot_explainable_image(self, class_to_explain: int = None, num_superpixels: int = 4,
                                    explainable_model_type: str = 'decision_tree_regressor'):
                """
                Plots the most important super pixel areas for predicting a specific class
                Parameters
                ----------
                class_to_explain: int; which class to plot the explanations for. If not specified, will
                default to the class with the highest probability
                num_superpixels: int; defines how many superpixel areas will be plotted
                explainable_model_type: str; which explainable model to use to generate explainability plot
                """
                # get perturbed image predictions & weights
                perturbed_image_predictions = self.predict_perturbed_images()
                weights = self.calculate_perturbation_weights()

                if class_to_explain is None:
                    class_to_explain = np.argmax(np.array(self.model(np.expand_dims(self.image, 0))))

                # fit simple interpretable model
                feature_importance = self._fit_explainable_model(
                    predictions=perturbed_image_predictions[:, class_to_explain],
                    weights=weights,
                    explainable_model_type=explainable_model_type
                )

                # Define which superpixel areas should be plotted
                superpixels_to_plot = np.argsort(feature_importance)[-num_superpixels:]
                superpixel_vector = np.zeros(self.super_pixel_count)
                np.put(superpixel_vector, superpixels_to_plot, v=1)

                # Create the image
                perturbed_img = self._create_perturbed_image(superpixel_vector)
                plt.imshow(perturbed_img.astype(int))
                plt.title('LIME explanation')

                

        img = PIL.Image.open(archivo)

        L = LIME(img, model, random_seed=10)

        ploty.figure(figsize=(14, 7))
        ploty.subplot(141)
        ploty.imshow(numpy.array(img).astype(int))
        ploty.axis("off")
        ploty.title("Original image")
        ploty.subplot(142)
        L.plot_super_pixel_boundary()
        ploty.subplot(143)
        # L.plot_explainable_image(explainable_model_type='linear_regression')
        ploty.subplot(144)
        # L.plot_explainable_image(explainable_model_type='decision_tree_regressor')
        ploty.suptitle("LIME for Images", size=18)
        buffer = BytesIO()
        ploty.savefig(
            buffer, format="png", bbox_inches="tight", pad_inches=0, transparent=True
        )
        buffer.seek(0)

        # Convierte la imagen en base64
        image_base64 = base64.b64encode(buffer.read()).decode()

        graficos.append({"tipo": "foto", "base64": image_base64, "text": ""})
        """"""
        return graficos

def python_conv_lime():

    from api import son_agentes_activos, detener_agentes, agentes_activos

    global graficos_recibidos 
    global archivo 
    archivo = request.files["archivo"]

    if request.method == "POST":
                    #-----------------------------------AGENTES----------------------------------------
            graficos_recibidos = None
            try:
                
                    detener_agentes()    
                    #Credenciales de cada agente
                    jid_creds = [
                            ("python_conv_lime@chalec.org", "hola1234"),
                            ("manageworkflowagent@chalec.org", "hola1234")
                        ]

                    if not son_agentes_activos():

                        
                        # Inicializa los agentes aquí
                        agentes_activos["python_conv_lime"] = ReceiverAgent(jid_creds[0][0], jid_creds[0][1])
                        agentes_activos["python_conv_lime"].start(auto_register=True)
                        
                        agentes_activos["agentManager"] = PeriodicSenderagentPython_conv_lime(jid_creds[1][0], jid_creds[1][1])                   
                        agentes_activos["agentManager"].start(auto_register=True)
                        
                        agentes_activos["agentManager"].set("receiver_jid_python_conv_lime", jid_creds[0][0])

                        agentes_activos["python_conv_lime"].web.start(hostname="127.0.0.1", port="1003")
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