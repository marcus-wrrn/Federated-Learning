from dataclasses import dataclass

@dataclass
class ClientResponse:
    def __init__(self, client_data):
        assert "key" in client_data and "state" in client_data and "model_id" in client_data
        
        self.client_id = client_data["key"]
        self.state = client_data["state"]
        self.model_id = client_data["model_id"]

@dataclass
class TrainRound:
    round_id: int
    current_round: int
    max_rounds: int
    client_threshold: int
    learning_rate: float
    is_aggregating: bool

@dataclass
class Hyperparameters:
    learning_rate: float

@dataclass
class Client:
    client_id: str
    model_id: str
    next_state: str
    current_state: str

@dataclass
class CoordinationResponse:
    def __init__(self, client: Client, train_round: TrainRound):
        self.client_id = client.client_id
        self.model_id = client.model_id
        self.next_state = client.next_state
        self.hyperparameters = ...


