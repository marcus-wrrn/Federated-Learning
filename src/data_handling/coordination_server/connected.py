from flask import Flask
from flask import request

app = Flask(__name__)
@app.route("/connected", methods=['POST', 'GET'])

def connected():
    print("The following device has connected to the network : " + request.remote_addr)
    print("The end")
    return"<p> Connected Devices</p?>"

########

if __name__ == '__main__':
    app.run(host='192.168.2.24', port=5000)

