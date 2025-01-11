import os
from flask import Flask
from flcore.models.basic import HARSModel
import threading
import requests

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    # Add global values 
    # TODO: Move to config.py file
    app.config.from_mapping(
        DB_PATH = os.path.join(app.instance_path, "db.sqlite"),
        KEY_PATH = os.path.join(app.instance_path, "client_hash.txt"),
        MODEL_PATH = os.path.join(app.instance_path, "model.pth"),
        DATA_PATH = os.path.join(app.instance_path, "train.csv"),
        COORDINATION_IP = "http://127.0.0.1:8000"
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    # Load the directory that stores all important file data
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Add routes to app
    from . import server
    app.register_blueprint(server.bp)

    # Start pinging server
    with app.app_context():
        start_scheduler(app)

    return app


def coordinate_with_server(server_ip: str):
    try:
        response = requests.post(server_ip)
    except Exception as e:
        print(f"Failed to ping coordination server: {e}")
    finally:
        threading.Timer(30.0, coordinate_with_server, args=[server_ip]).start()

def start_scheduler(app: Flask):
    server_ip = app.config.get("COORDINATION_IP")
    coordinate_with_server(server_ip)