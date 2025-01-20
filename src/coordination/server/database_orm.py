import sqlite3
from flask import current_app
import os
from coordination.server.data_classes import TrainRound, Client

class CoordinationDB:
    """
    Allows application code to interact with an SQLite Database,

    WebServers don't really allow for mutable data between multiple processes/sessions
    a database is needed to allow for scalability. An ORM (Object Relational Mapper) allows application code to more seamlessly interact with an SQL like database

    """
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.execute('PRAGMA foreign_keys = ON;')
        self.cursor = self.conn.cursor()
        self._init_tables()

    def _init_tables(self):
        # We have 4 tables, 
        # 'model' which contains the model id and a reference to the training round its assosciated with,
        # The training round which contains the training information for each round,
        # clients, which contains client ids which reference the current model the client is assosciated with,
        # training_config, which is a table with only one row, containing a reference to the current training round
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS model (
                model_id TEXT PRIMARY KEY, 
                round_id INTEGER NOT NULL,
                FOREIGN KEY (round_id) REFERENCES train_round (round_id)
                    ON DELETE CASCADE ON UPDATE CASCADE
            );
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS clients (
                client_id TEXT PRIMARY KEY UNIQUE,
                model_id TEXT,
                current_state TEXT DEFAULT INITIALIZATION,
                next_state TEXT DEFAULT IDLE,
                FOREIGN KEY (model_id) REFERENCES model (model_id)
                    ON DELETE CASCADE ON UPDATE CASCADE
            );
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS train_round (
                round_id INTEGER PRIMARY KEY AUTOINCREMENT,
                current_round INTEGER DEFAULT 0,
                max_rounds INTEGER NOT NULL,
                client_threshold INTEGER NOT NULL,
                learning_rate REAL DEFAULT 0.01,
                is_aggregating INTEGER DEFAULT 0
            );
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_config (
                id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1), -- ensures only one table can exist
                round_id INTEGER,
                FOREIGN KEY (round_id) REFERENCES train_round (round_id)
                    ON DELETE CASCADE ON UPDATE CASCADE
            );
        ''')

        self.conn.commit()

    def add_client(self, client_id: str, model_id: str | None, current_state: str, next_state="IDLE", commit=True) -> None:
        if self.client_exists(client_id):
            raise Exception("Client already exists")

        if model_id is None:
            self.cursor.execute("INSERT INTO clients (client_id, current_state, next_state) VALUES (?, ?, ?)", (client_id, current_state, next_state,))
        else:
            # Check if model_id exists in table
            assert self.model_exists(model_id)
            self.cursor.execute("INSERT INTO clients (client_id, model_id, current_state, next_state) VALUES (?, ?, ?, ?)", (client_id, model_id, current_state, next_state))

        if commit: self.conn.commit()

    def get_client(self, client_id: str) -> Client | None:
        self.cursor.execute("SELECT * FROM clients WHERE client_id = ?", (client_id,))
        result = self.cursor.fetchone()

        if not result:
            return None
        
        return Client(
            client_id=result[0],
            model_id=result[1],
            current_state=result[2],
            next_state=result[3]
        )
    
    def update_client_model(self, client_id: str, model_id: str, commit=True):
        if not self.model_exists(model_id) or not self.client_exists(client_id):
            raise Exception("Model or Client does not exist")
        
        self.cursor.execute("UPDATE clients SET model_id = ? WHERE client_id = ?", (model_id, client_id,))
        if commit: self.conn.commit()

    
    def initialize_training(self,
                            max_rounds: int,
                            client_threshold: int,
                            learning_rate: float):
        self.cursor.execute("INSERT INTO train_round (max_rounds, client_threshold, learning_rate) VALUES (?, ?, ?)", (max_rounds, client_threshold, learning_rate,))
        current_round_id = self.cursor.lastrowid

        if self.current_round_id() is None:
            self.cursor.execute("INSERT INTO training_config (round_id) VALUES (?)", (current_round_id,))
        else:
            self.cursor.execute("UPDATE training_config SET round_id = ? WHERE id = 1", (current_round_id,))
        
        self.conn.commit()


    def start_training_round(self, max_rounds: int, client_threshold: int):
        assert max_rounds > 0
        assert client_threshold > 0

        self.cursor.execute("""
            INSERT INTO train_round (max_rounds, current_round, client_threshold, is_aggregating) VALUES (?, ?, ?, ?)
        """, (max_rounds, 0, client_threshold, False))
        self.conn.commit()

    def client_exists(self, client_id: str) -> bool:
        self.cursor.execute("SELECT 1 FROM clients WHERE client_id = ?", (client_id,))
        result = self.cursor.fetchone()
        return result is not None

    def model_exists(self, model_id: str) -> bool:
        self.cursor.execute("SELECT 1 FROM model WHERE model_id = ?", (model_id,))
        result = self.cursor.fetchone()
        return result is not None
    
    def current_round_id(self) -> int | None:
        self.cursor.execute("SELECT round_id FROM training_config")
        result = self.cursor.fetchone()
        if result is not None:
            return result[0]
        return None
    
    def get_current_round(self) -> TrainRound | None:
        self.cursor.execute("""
            SELECT tr.*
            FROM train_round tr
            JOIN training_config tc on tr.round_id = tc.round_id
            WHERE tc.id = 1
        """)
        result = self.cursor.fetchone()

        if not result:
            return None
        
        return TrainRound(
            round_id=result[0],
            current_round=result[1],
            max_rounds=result[2],
            client_threshold=result[3],
            learning_rate=result[4],
            is_aggregating=result[5]
        )

    def close(self):
        self.conn.close()

    # These methods are for context management
    # So the database can be called with 'with' similar to how python deals with files
    # It means we do not have to worry about closing the database connection in case an error occurs
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()   