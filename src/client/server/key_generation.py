import random
import datetime
import hashlib
import string
import os
import requests

def generate_random_key() -> str:
    now = datetime.datetime.now()
    strdate = now.isoformat()
    rand_key = ''
    characters = string.ascii_letters + string.digits
    rand_key = rand_key.join(random.choices(characters, k=random.randint(0, len(characters) - 1)))
    key = rand_key + strdate
    key = key.encode()
    hash =  hashlib.md5(key).hexdigest()

    # with open("client_key.txt", "w") as key_file:
    #     key_file.write(hash)
    
    return hash

def load_key(key_path: str):
    client_key = None
    #locations for where more security could be added as the clientkey is currently stored as text file 
    if os.path.exists(key_path):
        with open(key_path, "r") as fp:
            client_key = fp.read()

    return client_key

def get_key(key_path: str):
    client_key = load_key(key_path)
    if not client_key:
        client_key = generate_random_key()

        # Save key
        with open(key_path, "w") as fp:
            fp.write(client_key)
        
    return client_key

def upload_key(server_url):
    if not server_url.startswith('http://') and not server_url.startswith('https://'):
        server_url = 'http://' + server_url
    client_key = get_key()
    response = requests.post(f'{server_url}/init_connection', data=client_key)