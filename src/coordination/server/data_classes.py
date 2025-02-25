from dataclasses import dataclass
from enum import Enum
class ClientState(Enum):
    INITIALIZATION = "INITIALIZATION"
    IDLE = "IDLE"
    TRAIN = "TRAIN"
    TEARDOWN = "TEARDOWN"
@dataclass
class ClientRequest:
    def __init__(self, client_data):
        if not ("client_id" in client_data and "state" in client_data and "model_id" in client_data):
            raise Exception("Client Response is missing neccessary components")
        
        self.client_id = client_data["client_id"]
        self.state = ClientState(client_data["state"])
        self.model_id = client_data["model_id"]

@dataclass
class TrainRound:
    super_round_id: int
    round_id: int
    max_rounds: int
    client_threshold: int
    learning_rate: float
    is_aggregating: bool
    step_size: int
    gamma: float

@dataclass
class Hyperparameters:
    learning_rate: float

@dataclass
class Client:
    client_id: str
    model_id: str
    state: str
    has_trained: bool

@dataclass
class CoordinationResponse:
    client_id: str
    model_id: str
    state: str
    hyperparameters: Hyperparameters | None


