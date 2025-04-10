import client
import time
from flcore.logger import client_logger, setup_client_logger
from state_logic  import communicate_with_server, get_new_model
from config import ClientState
import argparse
from config import TrainingConfig
from state_logic import upload_model
import os 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5000")

    args = parser.parse_args()
    # Must be run in src/client
    TEST_MODEL_DIR = r"data/test/client_test/"

    cfg = TrainingConfig(
        train_path = "", 
        instance_path = os.getcwd(),
        #assume os.getcwd is /src/client
        host_ip = args.server_url,
        cuda = False,
    )  
    current_path = os.path.dirname(cfg.instance_path)
    current_path = os.path.dirname(current_path)
    print(current_path)
    response = communicate_with_server(cfg)
    cfg.model_id = response.model_id
    current_state = ClientState(response.state)
    cfg.current_state = current_state
    for i in range(1,4):
        client_dir = os.path.join(current_path,TEST_MODEL_DIR)
        print(f"client Dir : {client_dir}")
        client_dir = os.path.join(client_dir,f"client{i}/")
        print(f"client Dir : {client_dir}")
        id_path = os.path.join(client_dir , "client_hash.txt")
        print(id_path)
        model_path =  os.path.join(client_dir , "model.pth")
        print(model_path)
        print(f"Uploading test {i}")
        print(id_path)
        print(model_path)
        with open(id_path, "r") as fp:
            client_id = fp.read()        
        cfg.client_id = client_id
        cfg.instance_path = client_dir

        upload_model(cfg)
        # load the model 
        # create a blank client
        # create a cfg 
        # 
        
