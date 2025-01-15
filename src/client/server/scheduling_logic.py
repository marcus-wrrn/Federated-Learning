import threading
import requests
from config import TrainingConfig, Hyperparameters, CoordinationServerResponse, ClientState

def initialize_client(cfg: TrainingConfig) -> CoordinationServerResponse:
    data = {
        "key": cfg.client_id,
        "state": cfg.current_state,
        "model_id": cfg.model_id,
    }
    response = requests.post(cfg.host_ip, data)
    response.raise_for_status()

    json_data = response.json()

    # If hyperparameters exist, replace json with dataclass
    # Allows us to load it more easily into the CoordinationServerResponse class
    hyperparams = json_data.get("hyperparameters")
    if hyperparams:
        json_data["hyperparameters"] = Hyperparameters(**hyperparams)
    
    return CoordinationServerResponse(**json_data)

def get_new_model(server_address: str, model_id: str) -> requests.Response:

    route = server_address + f"/get_model/{model_id}"
    response = requests.get(route)
    response.raise_for_status()

    return response

def cast_string_client_state(enum_class, value):
    try:
        return enum_class(value)
    except ValueError:
        return None

def coordinate_with_server(config: TrainingConfig):
    try:
        response = initialize_client(config)
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
        current_state = cast_string_client_state(ClientState, response.next_state)
        assert current_state != None
        config.current_state = current_state

        # If in training mode start training
        if config.current_state == ClientState.TRAIN:
            # train_model(config, hyperparameters)
            ...
    except Exception as e:
        print(f"Failed to ping coordination server: {e}")
    finally:
        if config.current_state != ClientState.TEARDOWN:
            threading.Timer(config.wait_time, coordinate_with_server, args=[config]).start()
        print("Client has closed successfully")

def start_scheduler(config: TrainingConfig):
    coordinate_with_server(config)
