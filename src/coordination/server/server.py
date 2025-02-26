from flask import request, jsonify, Blueprint, current_app, send_file
import torch
import threading
from flcore.models.basic import HARSModel
import os
import sqlite3
from server.database_orm import CoordinationDB
from server.data_classes import ClientRequest, CoordinationResponse, ClientState, Hyperparameters
from dataclasses import asdict

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
            current_model_id = db.get_model_id(current_round.super_round_id, current_round.round_id)
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
            #print("Round initialized")

            round = db.get_current_round()
            print(f"Learning rate: {round.learning_rate}")
            if round is None:
                raise Exception("Round is none")
            
            model = HARSModel("cpu")
            model_id = db.create_model(super_id=round.super_round_id, round_id=round.round_id)

            # create round directory and current model
        path = os.path.join(current_app.instance_path, f"super_round_{round.super_round_id}/training_round_{round.round_id}/{model_id}.pth")
        torch.save(model.state_dict(), path)

        return jsonify(asdict(round)), 200
    except Exception as e:
        return f"Error processing request: {e}", 500

@bp.route('/shutdown', methods=['POST'])
def shutdown():
    with CoordinationDB(current_app.config["DATAPATH"]) as db:
        db.stop_training()
    return jsonify({"message": "Stopped Training", "success": True}), 200
    
@bp.route('/connect_test',methods=['POST','GET'])
def connected():
    print("The following device has connected to the network : "+request.remote_addr)
    print("End message")
    
    return"<p> YOU ARE CONNECTED ! <p>"

