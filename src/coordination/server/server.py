from flask import request, jsonify, Blueprint, current_app, send_file
import torch
import threading
from flcore.models.basic import HARSModel
import os
import sqlite3

bp = Blueprint("training", __name__, url_prefix="/training")

@bp.route('/get_model', methods=['GET'])
def get_model():
    path = current_app.config["GLOBAL_BIN_PATH"]

    if not os.path.exists(path):
        return 500, "Model uninitialized"

    return send_file(path)

@bp.route('/send_update', methods=['POST'])
def receive_update():
    client_updates: list = current_app.config["CLIENT_UPDATES"]
    num_clients: int = current_app.config["NUM_CLIENTS"]
    global_model: HARSModel = current_app.config["GLOBAL_MODEL"]
    round_completed: threading.Event = current_app.config["ROUND_COMPLETED"]

    # Receive updated model parameters from client
    client_id = request.args.get('client_id', 'Unknown')
    client_state = request.data

    # Saving then loading from a file is not neccessary we can just save the data to the client updates directly
    # with open(f'client_model_{client_id}.pt', 'wb') as f:
    #     f.write(update)
    # client_state = torch.load(f'client_model_{client_id}.pt', map_location='cpu', weights_only=True)
    client_updates.append((client_state, client_id))
    
    print(f"Received update from client {client_id}")

    # If not currently aggregating 
    if len(client_updates) == num_clients and not round_completed.is_set():
        round_completed.set()
        # Aggregate updates
        states = [cs[0] for cs in client_updates]
        global_model_state = aggregate_models(states)
        
        global_model = HARSModel(device='cpu').load_state_dict(global_model_state)
        global_binary = global_model.export_binary()

        # Overwrite the global binary
        with open(current_app.config["GLOBAL_BIN_PATH"], "wb") as fp:
            fp.write(global_binary)

        current_app.config["CLIENT_UPDATES"] = []
        current_app.config["CURRENT_ROUND"] += 1

        round_completed.clear()
        
    return 'Update received', 200

@bp.route('/init_connection', methods=['POST'])
def add_client():
    client_key = request.data
    # will do the number of clients here
    # check to see if the key and the ip are in use
    # store info in a db?
    # need to get client list
    client_list = []
    ip_address = request.remote_addr
    db = sqlite3.connect(current_app.config["DATAPATH"])
    max_id = db.execute("SELECT MAX(id) FROM clients")
    existing_client = db.execute("SELECT * FROM clients WHERE ip_address = ? AND client_id = ?",(ip_address,client_key)).fetchone
    if not existing_client:
        db.add_client(client_key,ip_address)
        current_app.config["NUM_CLIENTS"] = current_app.config["NUM_CLIENTS"] + 1 # add 1 to the number of clients # might be able to calculate this number from the database 

@bp.route('/is_aggregated', methods=['GET'])
def is_aggregated():
    round_completed: threading.Event = current_app.config["ROUND_COMPLETED"]
    return jsonify({'aggregated': round_completed.is_set()})


# TODO: Improve aggregation logic and move to seperate file
def aggregate_models(client_states: dict) -> dict:
    # Simple average of model parameters
    global_model_state: dict = current_app.config["GLOBAL_MODEL"].state_dict()

    new_state = {}
    for key in global_model_state.keys():
        new_state[key] = sum([client_state[key] for client_state in client_states]) / len(client_states)
    return new_state

