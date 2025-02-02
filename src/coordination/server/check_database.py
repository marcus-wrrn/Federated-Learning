from flask import current_app
from database_orm import CoordinationDB


def check_database():
    while True:
        with CoordinationDB(current_app.config["DATAPATH"]) as db:
            # Check database
            
            ...