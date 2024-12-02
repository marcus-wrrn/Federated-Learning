from flask import request, jsonify, Blueprint, current_app, send_file
import torch
import threading
from flcore.models.basic import HARSModel
import os

bp = Blueprint("training", __name__, url_prefix="/training")

@bp.route('/get_model', methods=['GET'])
def get_model():
    path = os.path.join(current_app.instance_path, current_app.config["GLOBAL_BIN_PATH"])

    if not os.path.exists(path):
        global_model: HARSModel = current_app.config["GLOBAL_MODEL"]
        bin_data = global_model.export_binary(compress=True)
        with open(path, 'wb') as fp:
            fp.write(bin_data)

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
        
        global_model.load_state_dict(global_model_state)
        torch.save(global_model_state, "global_model.pt")
        current_app.config["CLIENT_UPDATES"] = []
        current_app.config["CURRENT_ROUND"] += 1

        # Save to file
        path = os.path.join(current_app.instance_path, current_app.config["GLOBAL_BIN_PATH"])
        global_binary = global_model.export_binary()

        with open(path, "wb") as fp:
            fp.write(global_binary)

        round_completed.clear()
        
    return 'Update received', 200

@bp.route('/is_aggregated', methods=['GET'])
def is_aggregated():
    round_completed: threading.Event = current_app.config["ROUND_COMPLETED"]
    return jsonify({'aggregated': round_completed.is_set()})


# TODO: Improve aggregation logic
def aggregate_models(client_states: dict) -> dict:
    # Simple average of model parameters
    global_model_state: dict = current_app.config["GLOBAL_MODEL"].state_dict()

    new_state = {}
    for key in global_model_state.keys():
        new_state[key] = sum([client_state[key] for client_state in client_states]) / len(client_states)
    return new_state

# def run_server():
#     app.run(host='0.0.0.0', port=8080, threaded=False, processes=1)

# if __name__ == '__main__':
#     run_server()
