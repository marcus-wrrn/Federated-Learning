import client
import time
from flcore.logger import client_logger, setup_client_logger
import argparse
from config import TrainingConfig
from state_logic import upload_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10, help="Number of splits")
    parser.add_argument("--file", type=str, default="eve_split", help="Path to data")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5000")

    args = parser.parse_args()
    # Must be run in src/client
    TEST_MODEL_DIR = r"../../data/test/client_test/"

    cfg = TrainingConfig(
        train_path = "", 
        instance_path = client_dir,
        host_ip = args.server_url,
        cuda = False,
    )  
    response = communicate_with_server(cfg)
    cfg.model_id = response.model_id
    current_state = ClientState(response.state)
    cfg.current_state = current_state
    for i in range(1,3):
        client_dir = TEST_MODEL_DIR + "client{i}/"
        id_path = client_dir + "client_hash.txt"
        model_path =  client_dir + "model.pth"
        print("Uploading test {i}")
        with open(id_path, "r") as fp:
            client_id = fp.read(response.client_id)        
        cfg.client_id = client_id
        upload_model(cfg)
        # load the model 
        # create a blank client
        # create a cfg 
        # 
        
