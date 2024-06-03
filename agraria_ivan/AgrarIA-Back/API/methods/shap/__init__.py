from flask import Blueprint

shap_bp = Blueprint('shap', __name__)

from . import shap