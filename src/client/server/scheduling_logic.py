import threading
import requests
import key_generation as kg
from config import TrainingConfig, Hyperparameters, CoordinationServerResponse
import os

def initialize_client(cfg: TrainingConfig) -> CoordinationServerResponse:
    data = {
        "key": cfg.client_id,
        "state": cfg.current_state,
        "model_id": cfg.model_id,
    }
    response = requests.post(cfg.host_ip, data)
    response.raise_for_status()

    json_data = response.json()

    hyperparams = json_data.get("hyperparameters")
    if hyperparams:
        json_data["hyperparameters"] = Hyperparameters(**hyperparams)
    

    return CoordinationServerResponse(**json_data)


def coordinate_with_server(config: TrainingConfig):
    try:
        response = initialize_client(config)
        

    except Exception as e:
        print(f"Failed to ping coordination server: {e}")
    finally:
        threading.Timer(config.wait_time, coordinate_with_server, args=[config]).start()

def start_scheduler(config: TrainingConfig):
    coordinate_with_server(config)
