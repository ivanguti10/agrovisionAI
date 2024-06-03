from flask import Blueprint

time_series_analysis_bp = Blueprint('time_series_analysis', __name__)

from . import time_series_analysis