from server import create_app
import argparse
from flcore.logger import setup_server_logger
parser = argparse.ArgumentParser(description="A script to start a local coordination server for testing")
parser.add_argument('--mode', type=int, default=1)
parser.add_argument('--ip',type = str,help ="Ip address", default='0.0.0.0')

args = parser.parse_args()

ip_address = args.ip


app = create_app()
app.config["TEST_MODE"] = args.mode
#########

if __name__ == "__main__":
    print("Running coordination server")
    app.run(host=ip_address, port=5000)