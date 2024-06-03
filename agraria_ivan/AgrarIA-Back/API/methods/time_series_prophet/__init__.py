from flask import Blueprint

time_series_prophet_bp = Blueprint('time_series_prophet', __name__)

from . import time_series_prophet