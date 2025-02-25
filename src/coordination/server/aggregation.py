from flask import current_app
from server.database_orm import CoordinationDB
import time
import server
import os
import torch
from flcore.models.basic import HARSModel
from server.validation import validation

def check_database():
    current_app.logger.info("Starting database check")
    #print("Starting database check")
    while True:       
        with CoordinationDB(current_app.config["DATAPATH"]) as db:
            # Check database    
            cur_round = db.get_current_round()
            if (cur_round is None): continue

            current_app.logger.info("Training round : {}".format(cur_round.round_id))
            client_count = db.get_trained_clients(cur_round.super_round_id, cur_round.round_id)
            print(f"Client Count: {client_count}")


            if client_count < cur_round.client_threshold:
                time.sleep(30)
                continue

            
            #print("Aggregating")
            current_app.logger.info("Aggregating")
            db.update_aggregate(1)
            cur_model_id = db.get_model_id(cur_round.super_round_id, cur_round.round_id)

            round_path = db.get_model_path(current_app.instance_path, cur_model_id)

            cur_model = HARSModel("cpu")
            cur_model.load_state_dict(torch.load(round_path))

            client_list_states = []
            client_ids = db.get_round_client_list(cur_model_id)
            for c_idx in range (0,len(client_ids)):
                
                client_path = db.get_client_model(current_app.instance_path, client_ids[c_idx][0], cur_model_id)
                current_app.logger.info("Loading Client ID : {} . Loading file {}".format(client_ids[c_idx][0],client_path))

                client_model = HARSModel("cpu")
                client_model.load_state_dict(torch.load(client_path))

                client_state = client_model.state_dict()
                client_list_states.append(client_state)

                db.flag_client_training(client_ids[c_idx][0], cur_model_id, 0)

            aggregate_states = agg_model(client_list_states,cur_model.state_dict())

            # save the aggregate model         
            cur_model.load_state_dict(aggregate_states)
            torch.save(cur_model.state_dict(), round_path)

            ## Do validation
            # Get test data set path
            datapath = current_app.instance_path
            while True:
                datapath = os.path.dirname(datapath)
                if(os.path.isdir(os.path.join(datapath,"data"))):
                    datapath = os.path.join(datapath,"data","test.csv")
                    break
                if datapath == os.sep:
                    datapath = None
                    break 
                
            # call validation function
            if datapath is not None :
                results = validation("cpu",datapath,cur_model)
                db.update_model_acc(cur_model_id, results.accuracy)

            max_round = cur_round.max_rounds
            db.update_aggregate(0)
            current_app.logger.info("Checking if done training (max round hit)")
            current_app.logger.debug("current round: {}, current + 1 : {}. max : {}".format(cur_round.round_id,cur_round.round_id+1, cur_round.max_rounds))
            
            if(cur_round.round_id + 1 <= cur_round.max_rounds):
                # implement new round logic
                db.update_round()

                new_round =  db.get_current_round()
                new_model_id = db.create_model(new_round.super_round_id, new_round.round_id)
                path = db.get_model_path(current_app.instance_path,new_model_id)
                torch.save(aggregate_states, path)
                current_app.logger.info("Starting training round {}".format(new_round.round_id))
            else:
                current_app.logger.info("Max rounds has been hit. Training done")                    
                db.cursor.execute("DELETE FROM training_config WHERE id = 1")
                db.conn.commit()
            
        time.sleep(30)
                

def agg_model(client_states , round_state:dict) -> dict:
    # Simple average of model parameters
    new_state = {}
    #print(type(round_state))
    #for client_state in client_states:
        #print(type(client_state))
        #print(client_state)
    if len(client_states) == 0:
        return round_state
    for key in round_state.keys():
        new_state[key] = sum([client_state[key] for client_state in client_states]) / len(client_states)
    return new_state        