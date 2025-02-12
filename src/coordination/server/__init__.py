import os
from flask import Flask
from flcore.models.basic import HARSModel
import threading
#import check_database
from server.check_database import check_database 

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
    # Add global values 
    # TODO: Move to config.py file
    # TODO: This method of global variables could potentially lead to bugs. We should be using a file system or database to store information SQLite may be a good choice
    app.config.from_mapping(
        DATAPATH = os.path.join(app.instance_path, "db.sqlite"),
        CLIENT_UPDATES = [],
        CURRENT_ROUND = 0,
        ROUND_COMPLETED = threading.Event(),
        NUM_CLIENTS = 2, # TODO: Should eventually be removed to allow for a dynamic number of clients
        GLOBAL_BIN_PATH = os.path.join(app.instance_path, "global_model.bin")
    )
    print("Database path : ")
    print(app.config["DATAPATH"])

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    # Load the directory that stores all important file data
    if not os.path.exists(app.instance_path):
        os.makedirs(app.instance_path)

    # Add routes to app
    from . import server
    app.register_blueprint(server.bp)
    print(app.url_map)
    # Setup

    # Threading
    # Threading with app context
    def run_check_database():
        with app.app_context():
            check_database()
    thread = threading.Thread(target=run_check_database,daemon=True)    
    thread.start()
    
    return app