import os
from flask import Flask
from flcore.models.basic import HARSModel
import threading

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    # Add global values 
    # TODO: Move to config.py file
    app.config.from_mapping(
        DB_PATH = os.path.join(app.instance_path, ""),
        MODEL_PATH = "model.pth",
        DATA_PATH = "train.csv"
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    # TBH not sure if we need this but could be useful for later
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Add routes to app
    from . import server
    app.register_blueprint(server.bp)
    
    return app