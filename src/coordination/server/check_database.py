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
    finished_agg = 0
    while True:       
        with CoordinationDB(current_app.config["DATAPATH"]) as db:
            # Check database    
            cur_round = db.get_current_round()
            if (cur_round is None or finished_agg): continue

            #print("Training round: ",cur_round.current_round)
            current_app.logger.info("Training round : {}".format(cur_round.current_round))
            client_threshold = db.get_round_threshold()
            client_count = db.get_client_round_num()
            if(client_count >= client_threshold):
                #print("Aggregating")
                current_app.logger.info("Aggregating")
                db.update_aggregate(1)
                cur_model_id = db.get_model_id(cur_round.current_round)

                round_path = db.get_model_path(current_app.instance_path,cur_model_id)
                cur_model = HARSModel("cpu")
                cur_model.load_state_dict(torch.load(round_path))
                client_list_states = []
                client_ids = db.get_round_client_list2(cur_model_id)
                for c_idx in range (0,len(client_ids)):
                    
                    client_path = db.get_client_model(current_app.instance_path,client_ids[c_idx][0],cur_model_id)
                    current_app.logger.info("Loading Client ID : {} . Loading file {}".format(client_ids[c_idx][0],client_path))
                    #print("Client id: ",client_ids[c_idx][0])
                    #print("Loading : ",client_path)
                    client_model = HARSModel("cpu")
                    client_model.load_state_dict(torch.load(client_path))
                    client_state = client_model.state_dict()
                    client_list_states.append(client_state)
                    db.flag_client_training(client_ids[c_idx][0],cur_model_id,0)

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

                max_round = db.get_max_rounds()
                db.update_aggregate(0)
                if(cur_round.current_round+1 <= max_round[0]):
                    # implement new round logic
                    db.update_round()
                    new_round =  db.get_current_round()
                    new_model_id = db.create_model(new_round.round_id)
                    model = HARSModel("cpu")
                    #new_model_id = db.get_model_id(new_round.round_id)
                    #path = os.path.join(current_app.instance_path, f"super_round_{new_round.super_round_id}/training_round_{new_round.round_id}/{new_model_id}.pth")
                    path = db.get_model_path(current_app.instance_path,new_model_id)
                    torch.save(model.state_dict(), path)
                else:
                    current_app.logger.info("Max rounds has been hit. Training done")                    
                    db.cursor.execute("DELETE FROM training_config WHERE id = 1")
                    finished_agg = 1


                #print("Finished aggregating ")
                
                
                

        time.sleep(30)
                

def agg_model(client_states , round_state:dict) -> dict:
    # Simple average of model parameters
    new_state = {}
    #print(type(round_state))
    #for client_state in client_states:
        #print(type(client_state))
        #print(client_state)

    for key in round_state.keys():
        new_state[key] = sum([client_state[key] for client_state in client_states]) / len(client_states)
    return new_state        