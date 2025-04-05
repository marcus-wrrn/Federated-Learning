import requests
import argparse

if __name__ == "__main__":
    print("Creating a client")
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5000")
    args = parser.parse_args()

    route = args.server_url + "/training/connection_test"
    print(route)
    response = requests.post(route)