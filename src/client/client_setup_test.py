import client
client1 = client.create_client(
    "C:/Users/spark/Documents/git/Federated-Learning/data/train.csv",
    "http://127.0.0.1:5000",
    "C:/Users/spark/Documents/git/Federated-Learning/models/client_test/client1",
    "n"
    )
client2 = client.create_client(
    "C:/Users/spark/Documents/git/Federated-Learning/data/train.csv",
    "http://127.0.0.1:5000",
    "C:/Users/spark/Documents/git/Federated-Learning/models/client_test/client2",
    "n"
)
client2 = client.create_client(
    "C:/Users/spark/Documents/git/Federated-Learning/data/train.csv",
    "http://127.0.0.1:5000",
    "C:/Users/spark/Documents/git/Federated-Learning/models/client_test/client3",
    "n"
)
