from flask import Flask
from flask import (
    Blueprint,
    request
)

bp = Blueprint('connections', __name__, url_prefix='/training')

@bp.route("/register", methods=['POST', 'GET'])
def register():
    print("The following device has connected to the network : " + request.remote_addr)
    print("The end")
    return"<p> Connected Devices</p?>"

########

# if __name__ == '__main__':
#     app.run(host='192.168.2.24', port=5000)

