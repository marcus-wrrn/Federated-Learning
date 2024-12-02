from flask import Flask, request, jsonify
import torch
import threading
import sys
from models.basic import HARSModel
import numpy as np

app = Flask(__name__)

# Global model
global_model = HARSModel(device='cpu')
global_model_state = global_model.state_dict()

# Parameters for synchronization
client_updates = []
num_clients = 2  # Adjust based on the number of clients
round_completed = threading.Event()

current_round = 0

@app.route('/get_model', methods=['GET'])
def get_model():
    global global_model_state, current_round, round_completed
    if round_completed.is_set():
        round_completed.clear()
        current_round += 1
    # Send the global model parameters to the client
    model_bytes = torch.save(global_model_state, 'global_model.pt')
    with open('global_model.pt', 'rb') as f:
        model_data = f.read()
    return model_data

@app.route('/send_update', methods=['POST'])
def receive_update():
    global client_updates, global_model_state
    # Receive updated model parameters from client
    client_id = request.args.get('client_id', 'Unknown')
    update = request.data
    with open(f'client_model_{client_id}.pt', 'wb') as f:
        f.write(update)
    client_state = torch.load(f'client_model_{client_id}.pt', map_location='cpu', weights_only=True)
    client_updates.append((client_state, client_id))
    
    print(f"Received update from client {client_id}")

    if len(client_updates) == num_clients:
        # Aggregate updates
        states = [cs[0] for cs in client_updates]
        global_model_state = aggregate_models(states)
        client_updates = []
        round_completed.set()
    return 'Update received', 200

@app.route('/is_aggregated', methods=['GET'])
def is_aggregated():
    return jsonify({'aggregated': round_completed.is_set()})

def aggregate_models(client_states):
    # Simple average of model parameters
    global global_model_state
    new_state = {}
    for key in global_model_state.keys():
        new_state[key] = sum([client_state[key] for client_state in client_states]) / len(client_states)
    return new_state

def run_server():
    app.run(host='0.0.0.0', port=8080, threaded=False, processes=1)

if __name__ == '__main__':
    run_server()
