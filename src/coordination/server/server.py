from flask import request, jsonify, Blueprint, current_app, send_file
import torch
import threading
from flcore.models.basic import HARSModel
import os
import sqlite3
from server.database_orm import CoordinationDB
from server.data_classes import ClientRequest, CoordinationResponse
from dataclasses import asdict

bp = Blueprint("training", __name__, url_prefix="/training")

@bp.route('/get_model/<model_id>', methods=['GET'])
def get_model(model_id):
    with CoordinationDB(current_app.config["DATAPATH"]) as db:

        path = db.get_model_path(current_app.instance_path, model_id)
        print(path)
        if not path:
            return "Model does not exist", 404
        if not os.path.exists(path):
            return "Model has been deleted", 500

    return send_file(path)

@bp.route('/display_models', methods=['GET'])
def display():
    with CoordinationDB(current_app.config["DATAPATH"]) as db:
        db.cursor.execute("SELECT * FROM model")
        results = db.cursor.fetchall()
        return results, 200

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
    client_key = request.get_json()
    # will do the number of clients here
    # check to see if the key and the ip are in use
    # store info in a db?
    # need to get client list
    client_list = []
    ip_address = request.remote_addr
    print(f"Following device has been connected : ip address : {ip_address}, client_key {client_key}")    
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

@bp.route('/ping', methods=['POST'])
def ping_server():
    data = request.get_json()
    try:
        client_resp = ClientRequest(data)
        with CoordinationDB(current_app.config["DATAPATH"]) as db:
            if not db.client_exists(client_resp.client_id):
                db.add_client(client_resp.client_id, client_resp.model_id, client_resp.state.value)
            
            # Get current round
            current_round = db.get_current_round()
            # If current round is none do not update the client script
            if current_round is None:
                client = db.get_client(client_resp.client_id)
                response = CoordinationResponse(client)
                return jsonify(asdict(response)), 200
            # Else get current model
            model_id = db.get_model_id(current_round.round_id)
            if model_id != client_resp.model_id:
                db.update_client_model(client_resp.client_id, model_id)

            client = db.get_client(client_resp.client_id)
            print(asdict(client))
            response = CoordinationResponse(
                client_id=client.client_id,
                model_id=client.model_id,
                next_state=client.next_state,
                hyperparameters=None
            ) 

        print(asdict(response))
        return jsonify(asdict(response)), 200
    
    except Exception as e:
        return f"Error processing request: {e}", 500


@bp.route('/initialize', methods=['POST'])
def init_training():
    data = request.get_json()
    try:
        if "max_rounds" not in data or "client_threshold" not in data or "learning_rate" not in data:
            raise Exception("Request missing required parameters")
        
        print("Initialization Started")
        with CoordinationDB(current_app.config["DATAPATH"]) as db:
            db.initialize_training(
                instance_path=current_app.instance_path,
                max_rounds=data["max_rounds"], 
                client_threshold=data["client_threshold"], 
                learning_rate= data["learning_rate"]
            )
            print("Round initialized")

            round = db.get_current_round()
            if round is None:
                raise Exception("Round is none")
            
            model = HARSModel("cpu")
            model_id = db.create_model(round.round_id)

            # create round directory and current model
        path = os.path.join(current_app.instance_path, f"training_round_{round.round_id}/{model_id}.pth")
        torch.save(model.state_dict(), path)

        return jsonify(asdict(round)), 200
    except Exception as e:
        return f"Error processing request: {e}", 500
    
@bp.route('/connect_test',methods=['POST','GET'])
def connected():
    print("The following device has connected to the network : "+request.remote_addr)
    print("End message")
    return"<p> YOU ARE CONNECTED ! <p>"


# TODO: Improve aggregation logic and move to seperate file
def aggregate_models(client_states: dict) -> dict:
    # Simple average of model parameters
    global_model_state: dict = current_app.config["GLOBAL_MODEL"].state_dict()

    new_state = {}
    for key in global_model_state.keys():
        new_state[key] = sum([client_state[key] for client_state in client_states]) / len(client_states)
    return new_state

