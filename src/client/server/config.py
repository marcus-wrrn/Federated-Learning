from dataclasses import dataclass
from enum import Enum
import os
import key_generation as kg
from requests import Response
from typing import Optional

class ClientState(Enum):
    INITIALIZATION = "INITIALIZATION"
    IDLE = "IDLE"
    TRAIN = "TRAIN"
    TEARDOWN = "TEARDOWN"

@dataclass
class TrainingConfig:
    train_path: str
    instance_path: str
    cuda: bool
    host_ip: str
    init_time: 15.0
    idle_time: 30.0
    model_id = None
    client_id = None
    current_state = ClientState.INITIALIZATION

    def __post_init__(self):
        # Initialize data dir
        os.makedirs(self.instance_path, exist_ok=True)
        
        # Initialize client_id
        self.client_id = kg.get_key(self._client_id_path)

        # Make sure the host ip is correct
        if not self.host_ip.startswith('http://') and not self.host_ip.startswith('https://'):
            self.host_ip = 'http://' + self.host_ip

    @property
    def _client_id_path(self) -> str:
        return self.instance_path + "/client_hash.txt"
    
    @property
    def model_path(self) -> str:
        return self.instance_path + "/model.pth"
    
    @property
    def wait_time(self) -> float:
        if self.current_state == ClientState.INITIALIZATION:
            return self.init_time
        return self.idle_time


@dataclass
class Hyperparameters:
    learning_rate: float

@dataclass
class CoordinationServerResponse:
    client_id: str
    model_id: str
    next_state: str
    hyperparameters: Optional[Hyperparameters]

