import client as client
import os
import argparse
# testing creating a brand new key 
if os.path.exists("client_key.txt"):
    os.remove("client_key.txt")
key1 = client.get_key()
print(key1)
# testing to see if loading a key works
key2 = client.get_key()

if key1 != key2:
    print("Error generated keys are not the same")
else:
    print("Generated keys are the same")

# test sending the key to the coordination setver 
parser = argparse.ArgumentParser(description="A script to start a local coordination server for testing")
parser.add_argument('-ip',type = str,help ="Ip address")
args = parser.parse_args()

if(args.ip=="josh"):
    ip_address = '192.168.2.24'
else:
    ip_address = args.ip

server_address = 'http://'+ip_address+':5000/training' 
temp2 = client.upload_key(server_address)
print(temp2)