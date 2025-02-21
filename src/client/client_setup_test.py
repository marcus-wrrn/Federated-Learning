import client
import time
from flcore.logger import client_logger, setup_client_logger

client_logger = setup_client_logger("/home/ubu/git/Federated-Learning/src/models/client_test")
client_logger.info("New super round")

#print("Creating client 1")
client_logger.info("Creating client 1")
client1 = client.create_client(
    #"/home/ubu/Documents/split_data/homogenous/1.csv",
    "/home/ubu/git/Federated-Learning/data/dataset_1.csv",
    #"http://192.168.2.71:5000",
    #"http://127.0.0.1:5000",
    "http://34.130.51.66:8080",
    "/home/ubu/git/Federated-Learning/src/models/client_test/client1",
    "y"
    )
#print("Creating client 2")
client_logger.info("Creating client 2")
client2 = client.create_client(
    #"/home/ubu/Documents/split_data/homogenous/3.csv",
    "/home/ubu/git/Federated-Learning/data/dataset_2.csv",
    #"http://192.168.2.71:5000",
    #"http://127.0.0.1:5000",
    "http://34.130.51.66:8080",
    "/home/ubu/git/Federated-Learning/src/models/client_test/client2",
    "y"
)

#print("Creating client 3")
client_logger.info("Creating client 3")
client2 = client.create_client(
    #"/home/ubu/Documents/split_data/homogenous/5.csv",
    "/home/ubu/git/Federated-Learning/data/dataset_3.csv",
    #"http://192.168.2.71:5000",
    "http://34.130.51.66:8080",    
    #"http://127.0.0.1:5000",
    "/home/ubu/git/Federated-Learning/src/models/client_test/client3",
    "y"
)

print("Creating client 4")
client4 = client.create_client(
    #"/home/ubu/Documents/split_data/homogenous/6.csv",
    "/home/ubu/git/Federated-Learning/data/dataset_4.csv",
    #"http://192.168.2.71:5000",
    #"http://127.0.0.1:5000",
    "http://34.130.51.66:8080",    
    "/home/ubu/git/Federated-Learning/src/models/client_test/client4",
    "y"
    )
print("Creating client 5")
client5 = client.create_client(
    #"/home/ubu/Documents/split_data/homogenous/7.csv",
    "/home/ubu/git/Federated-Learning/data/dataset_5.csv",
    #"http://192.168.2.71:5000",
    "http://34.130.51.66:8080",    
    #"http://127.0.0.1:5000",
    "/home/ubu/git/Federated-Learning/src/models/client_test/client5",
    "y"
)

print("Creating client 6")
client6 = client.create_client(
    #"/home/ubu/Documents/split_data/homogenous/11.csv",
    "/home/ubu/git/Federated-Learning/data/dataset_6.csv",
    #"http://192.168.2.71:5000",
    #"http://127.0.0.1:5000",
    "http://34.130.51.66:8080",    
    "/home/ubu/git/Federated-Learning/src/models/client_test/client6",
    "y"
)

print("Creating client 7")
client7 = client.create_client(
    #"/home/ubu/Documents/split_data/homogenous/14.csv",
    "/home/ubu/git/Federated-Learning/data/dataset_7.csv",
    #"http://192.168.2.71:5000",
    "http://34.130.51.66:8080",    
    #"http://127.0.0.1:5000",
    "/home/ubu/git/Federated-Learning/src/models/client_test/client7",
    "y"
    )
print("Creating client 8")
client8 = client.create_client(
    #"/home/ubu/Documents/split_data/homogenous/15.csv",
    "/home/ubu/git/Federated-Learning/data/dataset_8.csv",
    #"http://192.168.2.71:5000",
    "http://34.130.51.66:8080",    
    #"http://127.0.0.1:5000",
    "/home/ubu/git/Federated-Learning/src/models/client_test/client8",
    "y"
)

print("Creating client 9")
client9 = client.create_client(
    #"/home/ubu/Documents/split_data/homogenous/16.csv",
    "/home/ubu/git/Federated-Learning/data/dataset_9.csv",
    #"http://192.168.2.71:5000",
#    "http://127.0.0.1:5000",
    "http://34.130.51.66:8080",    
    "/home/ubu/git/Federated-Learning/src/models/client_test/client9",
    "y"
)
print("Creating client 10")
client10 = client.create_client(
    #"/home/ubu/Documents/split_data/homogenous/17.csv",
    "/home/ubu/git/Federated-Learning/data/dataset_10.csv",
    #"http://192.168.2.71:5000",
    "http://34.130.51.66:8080",    
    #"http://127.0.0.1:5000",
    "/home/ubu/git/Federated-Learning/src/models/client_test/client10",
    "y"
)

#print("Delay 5")
#time.sleep(5)
#print("calling start training")
#client.start_training("http://192.168.2.71:5000")
#client.start_training("http://127.0.0.1:5000")