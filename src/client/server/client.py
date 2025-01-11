# client.py
import requests
import torch
from torch.utils.data import DataLoader
from flcore.models.basic import HARSModel
from flcore.data_handling.datasets import HARSDataset
import argparse
import time
import os


def download_model(server_url, client_id):
    if not server_url.startswith('http://') and not server_url.startswith('https://'):
        server_url = 'http://' + server_url
    
    response = requests.get(f'{server_url}/get_model')
    client_folder = f'client{client_id}'

    os.makedirs(client_folder, exist_ok=True)
    global_model_path = os.path.join(client_folder, 'global_model.pt')

    with open(global_model_path, 'wb') as f:
        f.write(response.content)

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./train.csv")
    parser.add_argument("--server_url", type=str, default="http://localhost:8080")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cuda", type=str, default="y")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda.lower() == 'y' else "cpu")

    # Load data
    train_data = HARSDataset(args.train_path)
    train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

    # Initialize model
    model = HARSModel(device=device)

    for rnd in range(args.rounds):
        print(f"Client {args.client_id} - Round {rnd+1}")

        # Download the global model
        global_state = download_model(args.server_url, args.client_id)
        model.load_state_dict(global_state)

        # Train locally
        local_state = train(model, train_loader, device)

        # Upload the updated model
        response = upload_model(args.server_url, local_state, args.client_id)
        print(response.text)

        # Wait for the server to complete aggregation
        wait_for_aggregation(args.server_url)
