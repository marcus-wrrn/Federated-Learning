from server import create_app
import argparse

parser = argparse.ArgumentParser(description="A script to start a local coordination server for testing")
parser.add_argument('-ip',type = str,help ="Ip address")
args = parser.parse_args()

if(args.ip=="josh"):
    ip_address = '192.168.2.24'
else:
    ip_address = args.ip


app = create_app()

#########

if __name__ == "__main__":
    print("Running coordination server")
    app.run(debug=True,host=ip_address)