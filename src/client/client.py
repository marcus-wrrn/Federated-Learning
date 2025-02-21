# client.py
import requests
import torch
from torch.utils.data import DataLoader
from flcore.models.basic import HARSModel
from flcore.data_handling.datasets import HARSDataset
import argparse
import time
import os
from config import TrainingConfig
from state_logic import start_scheduler
import datetime
import string
import random
import hashlib
from flcore.logger import client_logger, setup_client_logger

def download_model(server_url: str, client_id: str):
    if not server_url.startswith('http://') and not server_url.startswith('https://'):
        server_url = 'http://' + server_url
    
    response = requests.get(f'{server_url}/get_model')
    client_logger.info("Recieved response, got model")
    #print("Got model")
    client_folder = f'client{client_id}'

    os.makedirs(client_folder, exist_ok=True)
    global_model_path = os.path.join(client_folder, 'global_model.pt')

    with open(global_model_path, 'wb') as f:
        f.write(response.content)
    #print("Saved model")
    client_logger.info("Saving model")
    global_model_state = torch.load(global_model_path, map_location='cpu', weights_only=True)

    return global_model_state

def upload_model(server_url, model_state, client_id):
    if not server_url.startswith('http://') and not server_url.startswith('https://'):
        server_url = 'http://' + server_url

    client_folder = f'client{client_id}'
    os.makedirs(client_folder, exist_ok=True)

    client_model_path = os.path.join(client_folder, 'client_model.pt')
    torch.save(model_state, client_model_path)

    with open(client_model_path, 'rb') as f:
        model_data = f.read()

    # Include client_id in the POST data
    response = requests.post(f'{server_url}/send_update', data=model_data, params={'client_id': client_id})

    return response

def train(model: HARSModel, train_loader, device):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    if not model.training:
        model.train()
    loss = model.fit(train_loader, optimizer, train=True)

    # for batch in train_loader:
    #     optimizer.zero_grad()
    #     data, target = batch  # Unpack the batch
    #     data = data.to(device)
    #     target = target.to(device)
    #     output = model(data)
    #     loss = torch.nn.functional.cross_entropy(output, target)
    #     loss.backward()
    #     optimizer.step()
    
    return model.state_dict()

def wait_for_aggregation(server_url):
    while True:
        response = requests.get(f'{server_url}/is_aggregated')
        if response.json().get('aggregated'):
            break
        time.sleep(1)  # Wait before checking again

def generate_random_key():
    now = datetime.datetime.now()
    strdate = now.isoformat()
    rand_key = ''
    characters = string.ascii_letters + string.digits
    length = 64-26
    rand_key = rand_key.join(random.choices(characters,k = length))
    key = rand_key + strdate
    key = key.encode()
    hash =  hashlib.md5(key).hexdigest()
    key_file = open("client_key.txt","w")
    key_file.write(hash)
    key_file.close()
    return hash

def upload_key(server_url):
    if not server_url.startswith('http://') and not server_url.startswith('https://'):
        server_url = 'http://' + server_url
    client_key = get_key()
    response = requests.post(f'{server_url}/init_connection', data=client_key)

def load_key():
    #locations for where more security could be added as the clientkey is currently stored as text file 
    if os.path.exists("client_key.txt"):
        key_file = open("client_key.txt","r")
        client_key = key_file.read()
        key_file.close()
    else: 
        client_key = None
    return client_key

def get_key():
    client_key = load_key()
    if not client_key:
        client_key = generate_random_key()
    return client_key

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_path", type=str, default="./train.csv")
#     parser.add_argument("--server_url", type=str, default="http://localhost:8080")
#     parser.add_argument("--cuda", type=str, default="y")
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() and args.cuda.lower() == 'y' else "cpu")

#     # Load data
#     train_data = HARSDataset(args.train_path)
#     train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

#     # Initialize model
#     model = HARSModel(device=device)

#     for rnd in range(args.rounds):
#         print(f"Client {args.client_id} - Round {rnd+1}")

#         # Download the global model
#         global_state = download_model(args.server_url, args.client_id)
#         model.load_state_dict(global_state)

#         # Train locally
#         local_state = train(model, train_loader, device)

#         # Upload the updated model
#         response = upload_model(args.server_url, local_state, args.client_id)
#         print(response.text)

#         # Wait for the server to complete aggregation
#         wait_for_aggregation(args.server_url)

def create_client(train_path_in,server_url_in,instance_path_in,cuda_in,log_path=None):
    print("Creating client")
    cfg = TrainingConfig(
        train_path = train_path_in, 
        instance_path = instance_path_in,
        host_ip = server_url_in,
        cuda = True if cuda_in.lower() == 'y' else False,
    )

    if not os.path.exists(cfg.instance_path):
        os.mkdir(cfg.instance_path)
    start_scheduler(cfg)    
    client_logger = setup_client_logger(log_path)
    client_logger.info("Creating new client")


def start_training(path,data=None):
    print("Starting training")
    if data is None:
        data = {
            "max_rounds" : 20,
            "client_threshold" : 10,
            "learning_rate" : 0.0000001
        }
    route = path +"/training/initialize"
    response = requests.post(route,json=data)
    response.raise_for_status()
    client_logger.info("Max rounds {}".format(data[max_rounds]))
    client_logger.info("Client threshold {}".format(data[client_threshold]))
    client_logger.info("Learning rate {}".format(data[learning_rate]))

if __name__ == "__main__":
    print("Creating a client")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./train.csv")
    parser.add_argument("--server_url", type=str, default="http://localhost:8080")
    parser.add_argument("--instance_path", type=str, default="./instance", help="Directory for storing model data, if running on same machine, this value must be unique")
    parser.add_argument("--cuda", type=str, default="y", help="Use Cuda: Y/n")
    #parser.add_argument("--log",type =int,default = 0,help="Do you want to log results")
    parser.add_argument("--log_path",type =str,default="./instance",help="Directory where you want to store the log")
    args = parser.parse_args()

    cfg = TrainingConfig(
        train_path = args.train_path, 
        instance_path = args.instance_path,
        host_ip = args.server_url,
        cuda = True if args.cuda.lower() == 'y' else False,
    )

    log_path = args.log_path
   # if(args.log == 1):
    client_logger = setup_client_logger(log_path)


    if not os.path.exists(cfg.instance_path):
        os.mkdir(cfg.instance_path)
    
    start_scheduler(cfg)

    



