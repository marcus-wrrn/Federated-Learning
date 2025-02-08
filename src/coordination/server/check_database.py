from flask import current_app
from database_orm import CoordinationDB
import time
import server
import os
import torch
from flcore.models.basic import HARSModel


def check_database():
    while True:
        with CoordinationDB(current_app.config["DATAPATH"]) as db:
            # Check database
            
            ...

             # need to load the datapath to the db, and request all the client information
             # check to see if it meets the threshold, if it does the return true
             # if not the return false 
             
             # logic 
             # get the client_threshold from the super round
             # need to get the current training round 
             # get the number of client models for each training round
                # using the number of models related to the training round 
             # if they are the same or greater than set true, else send false 
            # if true set the training round is aggregating to true (1)
            # if true get all the models and aggregate them (getting the weights and adding them together)
            # probably call a funciton called aggregate model just so it can be changed easier later
            # save training round model somewhere.
                # there is an aggregate model method in server 
            # with new model it sends it to the client with the next training information 
            # exit db

            cur_round = db.current_round_id()
            client_threshold = db.get_round_threshold()
            client_count = db.get_client_round_num()
            if(client_count > client_threshold):
                print("Can aggregate")
                db.update_aggregate(1)
                ## Do aggregation
                
                # CODE GOES HERE
                # call aggregation function
                # get current_model_id
                # save aggregate model to current_model_id 
                # need to have a check to see if it has reached the final training round
                # get current round [done]
                # get max round [done]
                # if max > current +1 [done]
                    # increment training round with db.update_round()                
                    
                    # create new round model with db.create_model()
                # update aggregat value to 0 [done]
                
                # need to create a function that creates a dict of the keys and their weights.

                # set up the client_dict

                # need to load the model the current model
                cur_model_id = db.get_model_id(cur_round)
                #path = os.path.join(current_app.instance_path, f"super_round_{cur_round.super_round_id}/training_round_{cur_round.round_id}/{new_model_id}.pth")
                path = db.get_model_path(current_app.instance_path,cur_model_id)
                cur_model = HARSModel("cpu")
                cur_model.load_state_dict(torch.load(path))

                # need to get all the clients who are going to be aggregated
                # need to combine them or something/format them so they can be called in the server.aggregate_models
                # neet to pass in the values to the agggregate_models function
                # need to get results and save them 

                # format clients
                client_list_states = []
                client_ids = db.get_round_client_list2(cur_model_id)
                for c_idx in range (1,len(client_ids)):
                    client_path = db.save_client_model(current_app.instance_path,client_ids[c_idx][0],cur_model_id)
                    client_model = HARSModel("cpu")
                    client_model.load(state_dict(torch.load(client_path)))
                    client_state = client_model.state_dict()
                    client_list_states = [client_list_states, client_state]

                aggregate_states = agg_model(client_list_states,cur_model)

                # save the aggregate model         
                cur_model.load_state_dict(aggregate_states)
                cur_model.save(path)

                if(client_count+1 < client_threshold):
                    # implement new round logic
                    new_round = db.update_round()
                    new_model = db.create_model(new_round)
                    model = HARSModel("cpu")
                    new_model_id = db.create_model(round.round_id)
                    #path = os.path.join(current_app.instance_path, f"super_round_{new_round.super_round_id}/training_round_{new_round.round_id}/{new_model_id}.pth")
                    path = db.get_model_path(current_app.instance_path,new_model_id)
                    torch.save(model.state_dict(), path)
                else:
                    print("Aggregation done")
                db.update_aggregate(0)


                print("Finished aggregating ")
                
                
                ## Do validation
                print("Validating results")
                # CODE GOES HERE
                # get current path
                # pass path to the validaiton function
                # # do something with results                
                print("Finish validating results")
        
        time.sleep(30)
                

def agg_model(client_states , round_state:dict) -> dict:
    # Simple average of model parameters
    new_state = {}
    for key in round_state.keys():
        new_state[key] = sum([client_state[key] for client_state in client_states]) / len(client_states)
    return new_state        