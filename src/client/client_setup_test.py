import client
import time
from flcore.logger import client_logger, setup_client_logger

client_logger = setup_client_logger("/home/ubu/git/Federated-Learning/src/models/client_test")
client_logger.info("New super round")

#print("Creating client 1")
client_logger.info("Creating client 1")
client1 = client.create_client(
    "/home/ubu/git/Federated-Learning/data/train.csv",
    #"http://192.168.2.71:5000",
    "http://127.0.0.1:5000",
    "/home/ubu/git/Federated-Learning/src/models/client_test/client1",
    "y"
    )
#print("Creating client 2")
client_logger.info("Creating client 2")
client2 = client.create_client(
    "/home/ubu/git/Federated-Learning/data/train.csv",
    #"http://192.168.2.71:5000",
    "http://127.0.0.1:5000",
    "/home/ubu/git/Federated-Learning/src/models/client_test/client2",
    "y"
)

#print("Creating client 3")
client_logger.info("Creating client 3")
client2 = client.create_client(
    "/home/ubu/git/Federated-Learning/data/train.csv",
    #"http://192.168.2.71:5000",
    "http://127.0.0.1:5000",
    "/home/ubu/git/Federated-Learning/src/models/client_test/client3",
    "y"
)

print("Delay 5")
time.sleep(5)
print("calling start training")
#client.start_training("http://192.168.2.71:5000")
client.start_training("http://127.0.0.1:5000")