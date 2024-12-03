from flask import request, jsonify, Blueprint, current_app, send_file
import torch
from flcore.models.basic import HARSModel

bp = Blueprint("training", __name__, url_prefix="/training")

