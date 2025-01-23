import sqlite3
from flask import current_app
import os

class CoordinationDB:
    """
    Allows application code to interact with an SQLite Database,

    WebServers don't really allow for mutable data between multiple processes/sessions
    a database is needed to allow for scalability. An ORM (Object Relational Mapper) allows application code to more seamlessly interact with an SQL like database

    """
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_tables()

    def _init_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS clients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT UNIQUE,
                model_path TEXT UNIQUE,
                round INTEGER DEFAULT 0,
                ip_address
            );
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS train_round (
                id INTEGER PRIMARY KEY,
                max_rounds INTEGER,
                current_round INTEGER DEFAULT 0,
                client_threshold INTEGER,
                is_aggregating BOOLEAN DEFAULT 0
            );
        ''')

        self.conn.commit()

    def add_client(self, client_id: str, ip_address: str, current_round=0, commit=True) -> None:
        model_path = os.path.join(current_app.instance_path, f"client{client_id}.pth")
        self.cursor.execute("INSERT INTO clients (client_id, model_path, round,ip_address) VALUES (?, ?, ?,?)", (client_id, model_path, current_round,ip_address))
        if commit: self.conn.commit()

    def start_training_round(self, max_rounds: int, client_threshold: int):
        assert max_rounds > 0
        assert client_threshold > 0

        self.cursor.execute("""
            INSERT INTO train_round (max_rounds, current_round, client_threshold, is_aggregating) VALUES (?, ?, ?, ?)
        """, (max_rounds, 0, client_threshold, False))
        self.conn.commit()

    def close(self):
        self.conn.close()

    # These methods are for context management
    # So the database can be called with 'with' similar to how other files are
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()