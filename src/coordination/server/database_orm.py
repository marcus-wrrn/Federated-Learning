import sqlite3
import os
from server.data_classes import TrainRound, Client, ClientState
import datetime
import random
import hashlib
import string

def generate_random_key() -> str:
    now = datetime.datetime.now()
    strdate = now.isoformat()
    rand_key = ''
    characters = string.ascii_letters + string.digits
    rand_key = rand_key.join(random.choices(characters, k=random.randint(0, len(characters) - 1)))
    key = rand_key + strdate
    key = key.encode()
    hash =  hashlib.md5(key).hexdigest()

    # with open("client_key.txt", "w") as key_file:
    #     key_file.write(hash)
    
    return hash

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
                model_id TEXT DEFAULT NULL,
                state TEXT DEFAULT 'INITIALIZATION',
                has_trained INTEGER DEFAULT 0,
                FOREIGN KEY (model_id) REFERENCES model (model_id)
                    ON DELETE CASCADE ON UPDATE CASCADE
            );
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS client_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cId TEXT,
                mId TEXT,
                FOREIGN KEY (mId) REFERENCES model (model_id)
                    ON DELETE CASCADE ON UPDATE CASCADE,
                FOREIGN KEY (cId) REFERENCES clients (client_id)
                    ON DELETE CASCADE ON UPDATE CASCADE
            );
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS train_round (
                round_id INTEGER PRIMARY KEY AUTOINCREMENT,
                current_round INTEGER DEFAULT 0,
                learning_rate REAL DEFAULT 0.01,
                is_aggregating INTEGER DEFAULT 0
            );
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS super_round (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                current_round_id INTEGER,
                max_rounds INTEGER NOT NULL,
                client_threshold INTEGER NOT NULL,
                FOREIGN KEY (current_round_id) REFERENCES train_round (round_id)
                    ON DELETE CASCADE ON UPDATE CASCADE
            );
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_config (
                id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1), -- ensures only one table can exist
                super_id INTEGER,
                round_id INTEGER,
                FOREIGN KEY (super_id) REFERENCES super_round (id)
                    ON DELETE CASCADE ON UPDATE CASCADE
                FOREIGN KEY (round_id) REFERENCES train_round (round_id)
                    ON DELETE CASCADE ON UPDATE CASCADE
            );
        ''')

        self.conn.commit()

    def add_client(self, client_id: str, model_id: str | None, current_state: str,  commit=True) -> None:
        if self.client_exists(client_id):
            raise Exception("Client already exists")
        print("client does not exist")
        if model_id is None:
            model_id = self.get_current_model_id()
            self.cursor.execute("INSERT INTO clients (client_id, model_id, state) VALUES (?, ?, ?)", (client_id, model_id, current_state,))
            print("model id added")
        else:
            # Check if model_id exists in table
            if not self.model_exists(model_id):
                raise Exception("Model does not exist in database")
            self.cursor.execute("INSERT INTO clients (client_id, model_id, state) VALUES (?, ?, ?)", (client_id, model_id, current_state,))

        if commit: self.conn.commit()

    def get_client(self, client_id: str) -> Client | None:
        self.cursor.execute("SELECT * FROM clients WHERE client_id = ?", (client_id,))
        result = self.cursor.fetchone()

        if not result:
            return None
        
        return Client(
            client_id=result[0],
            model_id=result[1],
            state=result[2],
            has_trained=bool(result[3])
        )
    
    def flag_client_training(self, client_id: str, model_id: str, commit=True):
        self.cursor.execute("UPDATE clients SET has_trained = ? WHERE client_id = ? AND model_id = ?", (1, client_id, model_id,))
        if commit: self.conn.commit()
    
    def add_client_model(self, client_id: str, model_id: str, commit=True):
        self.cursor.execute("INSERT INTO client_models (cId, mId) VALUES (?, ?)", (client_id, model_id,))
        if commit: self.conn.commit()

    def initialize_training(self,
                            instance_path: str,
                            max_rounds: int,
                            client_threshold: int,
                            learning_rate: float):
        if not (max_rounds > 0 and client_threshold > 0):
            raise Exception(f"Max Rounds and Client Threshold must be above 0 got: max rounds: {max_rounds}, client threshold: {client_threshold}")
    
        self.cursor.execute("""
            INSERT INTO train_round (current_round, learning_rate) VALUES (?, ?)""", (1, learning_rate))
        current_round_id = self.cursor.lastrowid

        self.cursor.execute("""
            INSERT INTO super_round (current_round_id, max_rounds, client_threshold)
            VALUES (?, ?, ?)
        """, (current_round_id, max_rounds, client_threshold))
        super_round_id = self.cursor.lastrowid

        if self.current_round_id() is None:
            self.cursor.execute("INSERT INTO training_config (super_id, round_id) VALUES (?, ?)", (super_round_id, current_round_id))
        else:
            self.cursor.execute("UPDATE training_config SET super_id = ?, round_id = ? WHERE id = 1", (super_round_id, current_round_id))
        
        self.conn.commit()

        # Create round directory
        path = os.path.join(instance_path, f"super_round_{super_round_id}/")
        os.makedirs(path)
        path = os.path.join(path, f"training_round_{current_round_id}/")
        os.makedirs(path)

    def create_model(self, round_id: int, commit=True) -> str:
        """Returns model ID"""
        model_id = generate_random_key() 
        # Keep generating a new model ID until a new key is generated
        while self.model_exists(model_id):
            model_id = generate_random_key()
        self.cursor.execute("INSERT INTO model (model_id, round_id) VALUES (?, ?)", (model_id, round_id))
        if commit: self.conn.commit()

        return model_id
    
    def model_exists(self, model_id: str) -> bool:
        self.cursor.execute("SELECT 1 FROM model WHERE model_id = ?", (model_id,))
        result = self.cursor.fetchone()
        return result is not None

    def get_model_id(self, round_id: int) -> str | None:
        self.cursor.execute("SELECT model_id FROM model WHERE round_id = ?", (round_id,))
        model_id = self.cursor.fetchone()
        if model_id:
            return model_id[0]
        return None
    
    def get_model_path(self, instance_path: str, model_id: str) -> str | None:
        round_data = self.get_current_round()
        if round_data:
            return os.path.join(instance_path, f"super_round_{round_data.super_round_id}/training_round_{round_data.round_id}/{model_id}.pth")
        return None

    def client_exists(self, client_id: str) -> bool:
        self.cursor.execute("SELECT 1 FROM clients WHERE client_id = ?", (client_id,))
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
            SELECT sr.id, tr.round_id, tr.current_round, sr.max_rounds, sr.client_threshold, 
                tr.learning_rate, tr.is_aggregating
            FROM train_round tr
            JOIN training_config tc ON tr.round_id = tc.round_id
            JOIN super_round sr ON tc.super_id = sr.id
            WHERE tc.id = 1
        """)
        result = self.cursor.fetchone()

        if not result:
            return None

        return TrainRound(
            super_round_id=result[0],
            round_id=result[1],
            current_round=result[2],
            max_rounds=result[3],  
            client_threshold=result[4],  
            learning_rate=result[5],
            is_aggregating=result[6]
        )
    
    def save_client_model(self, instance_path: str, client_id: str, model_id: str) -> str:
        round_data = self.get_current_round()
        
        if not round_data: return None
        
        path = os.path.join(instance_path, f"super_round_{round_data.super_round_id}/training_round_{round_data.round_id}/client_models/") 
        os.makedirs(path)
        path = os.path.join(path, f"{client_id}.pth")

        return path
    
    def get_current_model_id(self) -> str | None:
        self.cursor.execute("""
            SELECT model.model_id
            FROM model
            JOIN training_config ON model.round_id = training_config.round_id;
        """)

        result = self.cursor.fetchone()
        if result:
            return result[0]
        return None
    
    def update_round(self, model_id: str):
        current_round = self.get_current_round()
        # increment current round
        self.cursor.execute("UPDATE train_round SET current_round = ? WHERE round_id = ?", (current_round.current_round + 1, current_round.round_id))
        # create new model

    def close(self):
        self.conn.close()

    # These methods are for context management
    # So the database can be called with 'with' similar to how python deals with files
    # It means we do not have to worry about closing the database connection in case an error occurs
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()   