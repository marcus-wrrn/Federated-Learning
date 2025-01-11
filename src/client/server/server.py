from flask import request, jsonify, Blueprint, current_app, send_file, Flask
import torch
from flcore.models.basic import HARSModel
import threading

bp = Blueprint("training", __name__, url_prefix="/training")

def coordinate_with_server(server_ip: str):
    try:
        ...
    except Exception as e:
        print(f"Failed to ping coordination server: {e}")
    finally:
        threading.Timer(30.0, coordinate_with_server, args=[server_ip]).start()

def start_scheduler(app: Flask):
    server_ip = app.config.get("COORDINATION_IP")
    coordinate_with_server(server_ip)

