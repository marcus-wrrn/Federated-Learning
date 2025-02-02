from server.database_orm import CoordinationDB

with CoordinationDB("/home/marcus/Projects/Federated-Learning/src/coordination/instance/db.sqlite") as db:
    db.add_client("test", None, "INITIALIZATION")