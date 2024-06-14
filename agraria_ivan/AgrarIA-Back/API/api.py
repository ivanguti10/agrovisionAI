from flask import Flask, jsonify, request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flask_cors import CORS  # Importa CORS desde Flask-CORS

from methods.lime import lime as li
from methods.python_conv_lime import python_conv_lime as pcl
from methods.pytorch_conv2 import pytorch_conv2 as pc2
from methods.shap import shap as sh
from methods.time_series_analysis import time_series_analysis as tsa
from methods.time_series_prophet import time_series_prophet as tsp
from methods.yuca import yuca as yca




app = Flask(__name__)
CORS(app)
plt.rcParams["text.usetex"] = False


graficos_recibidos = None

agentes_activos = {
    "agentManager": None,    
    "agentRecept": None
}

# Función para verificar si los agentes están activos
def son_agentes_activos():
    return all(agentes_activos[agente] is not None for agente in agentes_activos)

# Función para detener agentes
def detener_agentes():
    if son_agentes_activos():
        for agente in agentes_activos:
            # Detiene cada agente si está activo
            if agentes_activos[agente] is not None:
                agentes_activos[agente].stop()
                agentes_activos[agente] = None
                

@app.route('/lime', methods=["GET", "POST"])
def lime_route():
    return li.limeFunction()

@app.route('/python_conv_lime', methods=["GET", "POST"])
def python_conv_lime_route():
    return pcl.python_conv_lime()

@app.route('/pytorch_conv2', methods=["GET", "POST"])
def pytorch_conv2_route():
    return pc2.pytorch_conv2()

@app.route('/shap', methods=["GET", "POST"])
def shap_route():
    return sh.shapFunction()

@app.route('/timeseriesanalysis', methods=["GET", "POST"])
def time_series_analysis_route():
    return tsa.time_series_analysis()

@app.route('/timeseriesprophet', methods=["GET", "POST"])
def time_series_prophet_route():
    return tsp.time_series_prophet()

@app.route('/yuca', methods=["GET", "POST"])
def yuca_route():
    return yca.yuca()

@app.route('/yuca_CBB', methods=["GET", "POST"])
def yuca_route_CBB():
    return yca.yuca_CBB()

@app.route('/yuca_CGM', methods=["GET", "POST"])
def yuca_route_CGM():
    return yca.yuca_CGM()

@app.route('/yuca_CDM', methods=["GET", "POST"])
def yuca_route_CDM():
    return yca.yuca_CDM()

@app.route('/yuca_CBSD', methods=["GET", "POST"])
def yuca_route_CBSD():
    return yca.yuca_CBSD()

@app.route('/cargarModelo', methods=["GET", "POST"])
def yuca_route_CargarModelo():
    return yca.yuca_cargarModelo()

@app.route('/cluster1', methods=["GET", "POST"])
def yuca_route_cluster1():
    return yca.cluster1()

@app.route('/cluster2', methods=["GET", "POST"])
def yuca_route_cluster2():
    return yca.cluster2()

@app.route('/cluster3', methods=["GET", "POST"])
def yuca_route_cluster3():
    return yca.cluster3()

@app.route('/cluster4', methods=["GET", "POST"])
def yuca_route_cluster4():
    return yca.cluster4()

@app.route('/cluster5', methods=["GET", "POST"])
def yuca_route_cluster5():
    return yca.cluster5()

@app.route('/buscarDuplicados', methods=["GET", "POST"])
def yuca_route_buscarDuplicados():
    return yca.buscarDuplicados()

@app.route('/mostrarDuplicados', methods=["GET", "POST"])
def yuca_route_mostrarDuplicados():
    return yca.mostrarDuplicados()

if __name__ == "__main__":
    app.run(debug=True)
