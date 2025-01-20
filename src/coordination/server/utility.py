from dataclasses import dataclass

@dataclass
class ClientResponse:
    def __init__(self, client_data):
        assert "key" in client_data and "state" in client_data and "model_id" in client_data
        
        self.key = client_data["key"]
        self.state = client_data["state"]
        self.model_id = client_data["model_id"]

