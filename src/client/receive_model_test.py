import client
import time
from flcore.logger import client_logger, setup_client_logger
from state_logic  import communicate_with_server, get_new_model
from config import ClientState
import argparse
from config import TrainingConfig
from state_logic import upload_model
import os
import hashlib
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5000")

    args = parser.parse_args()
    # Must be run in src/client    
    TEST_MODEL_DIR = r"../../data/test/"
    TEST_MODEL_FILE = r"agg_test/6793b484275ade3f8bb8f07e434d8842.pth"
    TEST_MODEL_SAVE_NAME = "file.pth"
    cfg = TrainingConfig(
        train_path = "", 
        instance_path = os.getcwd(),
        host_ip = args.server_url,
        cuda = False,
    )  

    # get_new_model(server_address: str, model_id: str) -> requests.Response:
    
    response = communicate_with_server(cfg)
    cfg.model_id = response.model_id
    current_state = ClientState(response.state)
    cfg.current_state = current_state

    model_resp = get_new_model(args.server_url, response.model_id)

    with open(os.path.join(TEST_MODEL_DIR,TEST_MODEL_SAVE_NAME), "wb") as fp:
        fp.write(model_resp.content)

    with open(os.path.join(TEST_MODEL_DIR,TEST_MODEL_SAVE_NAME), "rb") as f:
        receive_byte_size = f.read()

    with open(os.path.join(TEST_MODEL_DIR,TEST_MODEL_FILE) , "rb") as f:
        val_byte_size = f.read()
    receive_hash = hashlib.sha256(receive_byte_size).hexdigest()
    val_hash = hashlib.sha256(val_byte_size).hexdigest()
    if(receive_hash == val_hash):
        print("Upload successful")
    else:
        print("Upload failed. Model not identical")
    os.remove(os.path.join(TEST_MODEL_DIR,TEST_MODEL_SAVE_NAME))
