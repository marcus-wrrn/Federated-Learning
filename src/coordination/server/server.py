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
        print(path)
        if not path:
            return "Model does not exist", 404
        if not os.path.exists(path):
            return "Model has been deleted", 500

    return send_file(path)

@bp.route('/upload-model', methods=['POST'])
def upload_model():
    #print("HEre")
    print("Recieved client model")
    if "model" not in request.files:
        return "No model", 400    
    model_data = request.files["model"]
    client_id = request.form.get("client_id")
    model_id = request.form.get("model_id")
    print("Recieved from : ",client_id)

    try:
        # validate model

        print("DB")
        with CoordinationDB(current_app.config["DATAPATH"]) as db:
            print("flag client training")
            db.flag_client_training(client_id, model_id,1)
            print("Update client_model")
            db.add_client_model(client_id, model_id)
            print("Save client model")
            print(current_app.instance_path)
            print(client_id)
            print(model_id)
            filepath = db.save_client_model(current_app.instance_path, client_id, model_id)
            print(filepath)

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
    print("In Ping")
    data = request.get_json()
    try:
        print("In try")
        client_resp = ClientRequest(data)
        hyperparameters = None
        print("Establishing DB connection")
        with CoordinationDB(current_app.config["DATAPATH"]) as db:
            print("In DB")
            if not db.client_exists(client_resp.client_id):
                db.add_client(client_resp.client_id, client_resp.model_id, client_resp.state.value)
            # Get current round
            current_round = db.get_current_round()
            #print("Here1")
            # If current round is none or the model is currently aggregating do not update the client script
            if current_round is None:
                #print("cur 1")
                client = db.get_client(client_resp.client_id)
                #print("cur 2")
                #print(client)
                #print(type(client))
                #print("Corodination Response")
                response = CoordinationResponse(client_id=client.client_id, model_id=client.model_id, state=client.state,hyperparameters=None)
                #print("Finish coordination class")
                #print("here")
                #print(response)
                #print("cur 3")
                return jsonify(asdict(response)), 200
            #print("Here2")
            current_model_id = db.get_current_model_id()
            if client_resp.model_id != current_model_id:
                db.cursor.execute("UPDATE clients SET model_id = ?, has_trained = ? WHERE client_id = ?", (current_model_id, 0, client_resp.client_id))
                db.conn.commit()
            #print("Here3")                
            # If the system is aggregating and the client state is not idle, or if the client is initializing set the client to idle
            if (client_resp.state != ClientState.IDLE and current_round.is_aggregating) or client_resp.state == ClientState.INITIALIZATION:
                db.cursor.execute("UPDATE clients SET state = ? WHERE client_id = ?", (ClientState.IDLE.value, client_resp.client_id))
                db.conn.commit()
            
            # Check if the model should be training
            #print("Here4")
            client = db.get_client(client_resp.client_id)
            #print("trained: ",client.has_trained)
            #print("is agg: ",current_round.is_aggregating)

            if not client.has_trained and not current_round.is_aggregating:
                print("Updating client state")
                client.state = 'TRAIN'
                hyperparameters = Hyperparameters(learning_rate=current_round.learning_rate)
            #print("Here5")
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
    print("Start training")
    """
    Route for initializing a training session.
    """
    data = request.get_json()
    try:
        if "max_rounds" not in data or "client_threshold" not in data or "learning_rate" not in data:
            raise Exception("Request missing required parameters")
        
        with CoordinationDB(current_app.config["DATAPATH"]) as db:
            db.initialize_training(
                instance_path=current_app.instance_path,
                max_rounds=data["max_rounds"], 
                client_threshold=data["client_threshold"], 
                learning_rate= data["learning_rate"]
            )
            print("Round initialized")

            round = db.get_current_round()
            if round is None:
                raise Exception("Round is none")
            
            model = HARSModel("cpu")
            model_id = db.create_model(round.round_id)

            # create round directory and current model
        path = os.path.join(current_app.instance_path, f"super_round_{round.super_round_id}/training_round_{round.round_id}/{model_id}.pth")
        torch.save(model.state_dict(), path)

        return jsonify(asdict(round)), 200
    except Exception as e:
        return f"Error processing request: {e}", 500
    
@bp.route('/connect_test',methods=['POST','GET'])
def connected():
    print("The following device has connected to the network : "+request.remote_addr)
    print("End message")
    return"<p> YOU ARE CONNECTED ! <p>"


# TODO: Improve aggregation logic and move to seperate file
def aggregate_models(client_states: dict) -> dict:
    # Simple average of model parameters
    global_model_state: dict = current_app.config["GLOBAL_MODEL"].state_dict()

    new_state = {}
    for key in global_model_state.keys():
        new_state[key] = sum([client_state[key] for client_state in client_states]) / len(client_states)
    return new_state

