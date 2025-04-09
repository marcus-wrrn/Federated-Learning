from flask import request, jsonify, Blueprint, current_app, send_file
import torch
import threading
from flcore.models.basic import HARSModel
import os
import sqlite3
from server.database_orm import CoordinationDB
from server.data_classes import ClientRequest, CoordinationResponse, ClientState, Hyperparameters
from dataclasses import asdict
import hashlib

bp = Blueprint("training", __name__, url_prefix="/training")

@bp.route('/get_model/<model_id>', methods=['GET'])
def get_model(model_id):
    with CoordinationDB(current_app.config["DATAPATH"]) as db:
        path = db.get_model_path(current_app.instance_path, model_id)
        #print(path)
        current_app.logger.info("Getting model to {}".format(path))
        if not path:
            return "Model does not exist", 404
        if not os.path.exists(path):
            return "Model has been deleted", 500
    if(current_app.config["TEST_MODE"]==2):        
        #path = current_app.instance_path +  
        ## Need to find the path to the data
        #<directory of the file where __name__ is defined>/instance is where current_app.instance_app. So it is in coordination/server
        current_path = current_app.instance_path
        print(current_path)

        current_path = os.path.dirname(current_path)
        print(current_path)
        current_path = os.path.dirname(current_path)
        print(current_path)
        current_path = os.path.dirname(current_path)
        print(current_path)        
        path = os.path.join(current_path,"data/test/agg_test/6793b484275ade3f8bb8f07e434d8842.pth")              
        
    return send_file(path)

@bp.route('/upload-model', methods=['POST'])
def upload_model():
    #print("HEre")
    if "model" not in request.files:
        return "No model", 400    
    model_data = request.files["model"]
    client_id = request.form.get("client_id")
    model_id = request.form.get("model_id")
    #print("Recieved client model from : ",client_id)
    current_app.logger.info("Received client model from : {}".format(client_id))
    try:
        # validate model
        with CoordinationDB(current_app.config["DATAPATH"]) as db:
            db.flag_client_training(client_id, model_id,1)
            db.add_client_model(client_id, model_id)
            filepath = db.save_client_model(current_app.instance_path, client_id, model_id)

            if not filepath:
                return f"Pathing error", 500
    
        model_data.save(filepath)

        if(current_app.config["TEST_MODE"]):            
            uploaded_byte_size = model_data.read()
            validation_path = r"../../data/test/client_test/"
            if(client_id == "6241e9b7ba3b4fae90405cd726f30b28"):
                validation_model_path = validation_path + r"client1/model.pth"
            elif(client_id == "8cb593a3d8d1af7e87126012f6c8ba86"):
                validation_model_path = validation_path + r"client2/model.pth"
            elif(client_id == "e6d43a022b8cdc5bca1ac9dd8371afb5"):
                validation_model_path = validation_path + r"client3/model.pth"
            # assuming running from src/coordinate/
            with open(validation_model_path, "rb") as f:
                val_byte_size = f.read()
            upload_hash = hashlib.sha256(uploaded_byte_size).hexdigest()
            val_hash = hashlib.sha256(val_byte_size).hexdigest()
            if(upload_hash == val_hash):
                print("Upload successful")
            else:
                print("Upload failed. Model not identical")
        return "Model saved", 200
    except Exception as e:
        return f"Error uploading model: {e}", 500

@bp.route('/display_models', methods=['GET'])
def display():
    with CoordinationDB(current_app.config["DATAPATH"]) as db:
        db.cursor.execute("SELECT * FROM model")
        results = db.cursor.fetchall()
        return results, 200

@bp.route('/ping', methods=['POST'])
def ping_server():
    if(current_app.config["TEST_MODE"]==1):        
        print("Test Server")

    data = request.get_json()
    try:
        client_resp = ClientRequest(data)
        hyperparameters = None
        #print("Establishing DB connection")
        with CoordinationDB(current_app.config["DATAPATH"]) as db:
            if not db.client_exists(client_resp.client_id):
                db.add_client(client_resp.client_id, client_resp.model_id, client_resp.state.value)
            # Get current round
            current_round = db.get_current_round()
            # If current round is none or the model is currently aggregating do not update the client script
            if current_round is None:
                client = db.get_client(client_resp.client_id)
                response = CoordinationResponse(client_id=client.client_id, model_id=client.model_id, state=client.state,hyperparameters=None)
                return jsonify(asdict(response)), 200
            current_model_id = db.get_model_id(current_round.super_round, current_round.curr_round)
            if client_resp.model_id != current_model_id:
                db.cursor.execute("UPDATE clients SET model_id = ?, has_trained = ? WHERE client_id = ?", (current_model_id, 0, client_resp.client_id))
                db.conn.commit()
            # If the system is aggregating and the client state is not idle, or if the client is initializing set the client to idle
            if (client_resp.state != ClientState.IDLE and current_round.is_aggregating) or client_resp.state == ClientState.INITIALIZATION:
                db.cursor.execute("UPDATE clients SET state = ? WHERE client_id = ?", (ClientState.IDLE.value, client_resp.client_id))
                db.conn.commit()
            
            # Check if the model should be training
            client = db.get_client(client_resp.client_id)

            if not client.has_trained and not current_round.is_aggregating:
                client.state = 'TRAIN'
                hyperparameters = Hyperparameters(learning_rate=current_round.learning_rate)
            response = CoordinationResponse(
                client_id=client.client_id,
                model_id=client.model_id,
                state=client.state,
                hyperparameters=hyperparameters
            ) 

        return jsonify(asdict(response)), 200
    
    except Exception as e:
        return f"Error processing request: {e}", 500


@bp.route('/initialize', methods=['POST'])
def init_training():
    #print("Start training")
    current_app.logger.info("New Super round")
    current_app.logger.info("Start training")
    if(current_app.config["TEST_MODE"]==1):        
        print("Starting Test Training Round")

    """
    Route for initializing a training session.
    """
    data = request.get_json()
    print(f"Data: {data}")
    try:
        if "max_rounds" not in data or "client_threshold" not in data or "learning_rate" not in data or "step_size" not in data or "gamma" not in data:
            raise Exception("Request missing required parameters")
        
        with CoordinationDB(current_app.config["DATAPATH"]) as db:
            db.initialize_training(
                instance_path=current_app.instance_path,
                max_rounds=data["max_rounds"], 
                client_threshold=data["client_threshold"], 
                learning_rate= data["learning_rate"],
                step_size=data["step_size"],
                gamma=data["gamma"]
            )
            current_app.logger.info("Round initializer")
            current_app.logger.info("Round initializer")
            rounds = data["max_rounds"]
            epoch = f"Max rounds : {rounds}"
            c_thresh = client_threshold
            threshold = f"Client threshold : {c_thresh}"
            learning = data["learning_rate"]
            lr = f"Learning rate : {learning}"
            ss = data["step_size"]
            step = f"Step size : {ss}"
            g = data["gamma"]
            gam = f"Gamma : {g}"

            current_app.logger.info(epoch)
            current_app.logger.info(threshold)
            current_app.logger.info(lr)
            current_app.logger.info(step)
            current_app.logger.info(gam)
            if(current_app.config["TEST_MODE"]==5):
                init_error = False
                if(data["max_rounds"]!=1):
                    print("Error max rounds different then expected")
                    init_error = True
                if(data["client_threshold"]!=5):
                    print("Error client threshold different then expected")
                    init_error = True
                if(data["learning_rate"]!=0.001):
                    print("Error learning rate different then expected")
                    init_error = True
                if(data["step_size"]!=3):
                    print("Error step size different then expected")
                    init_error = True
                if(data["gamma"]!=0.2):
                    print("Error gamma different then expected")
                    init_error = True
                if(init_error):
                    return
            #print("Round initialized")

            round = db.get_current_round()
            print(f"Learning rate: {round.learning_rate}")
            if round is None:
                raise Exception("Round is none")
            
            model = HARSModel("cpu")
            model_id = db.create_model(super_id=round.super_round, round_id=round.curr_round)

            # create round directory and current model
        path = os.path.join(current_app.instance_path, f"super_round_{round.super_round}/training_round_{round.curr_round}/{model_id}.pth")
        torch.save(model.state_dict(), path)

        return jsonify(asdict(round)), 200
    except Exception as e:
        return f"Error processing request: {e}", 500

@bp.route('/shutdown', methods=['POST'])
def shutdown():
    with CoordinationDB(current_app.config["DATAPATH"]) as db:
        db.stop_training()
    return jsonify({"message": "Stopped Training", "success": True}), 200
    
@bp.route('/connection_test',methods=['POST','GET'])
def connected():
    print("The following device has connected to the network : "+request.remote_addr)
    print("End message")
    
    return"<p> YOU ARE CONNECTED ! <p>"

