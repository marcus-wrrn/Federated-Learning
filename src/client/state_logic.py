import threading
import requests
from config import TrainingConfig, Hyperparameters, CoordinationServerResponse, ClientState
import torch
from torch.utils.data import DataLoader
from flcore.models.basic import HARSModel
from flcore.data_handling.datasets import HARSDataset
from flcore.logger import client_logger
#import json

def communicate_with_server(cfg: TrainingConfig) -> CoordinationServerResponse:
    data = {
        "client_id": cfg.client_id,
        "state": cfg.current_state.value,
        "model_id": cfg.model_id,
    }

    route = cfg.host_ip + "/training/ping"
    response = requests.post(route, json=data)
    response.raise_for_status()

    json_data = response.json()

    # If hyperparameters exist, replace json with dataclass
    # Allows us to load it more easily into the CoordinationServerResponse class
    hyperparams = json_data.get("hyperparameters")
    if hyperparams:
        json_data["hyperparameters"] = Hyperparameters(**hyperparams)
    
    return CoordinationServerResponse(**json_data)

def get_new_model(server_address: str, model_id: str) -> requests.Response:
    route = server_address + f"/training/get_model/{model_id}"
    client_logger.info("Retrieving new mode")
    #rint("Retrieving new model")
    response = requests.get(route)
    #print("Got model")
    client_logger.info("Updated model")
    response.raise_for_status()

    return response

def upload_model(cfg: TrainingConfig):
    route = cfg.host_ip + "/training/upload-model"
    metadata = {
        "client_id": cfg.client_id,
        "state": cfg.current_state.value,
        "model_id": cfg.model_id
    }

    with open(cfg.model_path, "rb") as fp:
        files = {"model": fp}
        response = requests.post(route, files=files, data=metadata)
    
    

def save_model(model_path: str, model: HARSModel):
    torch.save(model.state_dict(), model_path)

def cast_string_client_state(enum_class, value):
    try:
        return enum_class(value)
    except ValueError:
        return None

def coordinate_with_server(config: TrainingConfig):
    try:
        client_logger.info("Communicating with server")
        #print("Communication with server")
        response = communicate_with_server(config)
        if response.client_id != config.client_id:
            # change client id
            with open(config.client_id_path, "w") as fp:
                fp.write(response.client_id)
            config = response.client_id

        if response.model_id != config.model_id:
            # download new model
            config.model_id = response.model_id
            model_resp = get_new_model(config.host_ip, response.model_id)
            with open(config.model_path, "wb") as fp:
                fp.write(model_resp.content)
        
        # cast response state to Enum
        current_state = ClientState(response.state)
        config.current_state = current_state

        # If in training mode start training
        if config.current_state == ClientState.TRAIN:
            #print("Starting Training")
            client_logger.info("Starting to train")
            device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
            model = HARSModel(device)

            model.load_state_dict(torch.load(config.model_path, weights_only=True))
            optimizer = torch.optim.AdamW(model.parameters(), response.hyperparameters.learning_rate)
            dataloader = DataLoader(HARSDataset(config.train_path), batch_size=2, shuffle=True)
            model.fit(dataloader, optimizer, train=True,config_key = config.client_id)
            # save model
            torch.save(model.state_dict(), config.model_path)
            # Send model file back to server
            #print("Uploading Model")
            client_logger.info("Uploading model")
            upload_model(config)

        
    except Exception as e:
        #print(f"Printstatement : Failed to ping coordination server: {e}")
        client_logger.info(f"Failed to ping coordination server: {e}")
    finally:
        if config.current_state != ClientState.TEARDOWN:
            threading.Timer(config.wait_time, coordinate_with_server, args=[config]).start()

def start_scheduler(config: TrainingConfig):
    coordinate_with_server(config)