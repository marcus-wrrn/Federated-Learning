from flask import current_app
from server.database_orm import CoordinationDB
import time
import server
import os
import torch
from flcore.models.basic import HARSModel
import server.validation

def check_database():
    print("Starting database check")
    count = 0
    while True:  
        count = count+1
        print ("Check round : ",count)     
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
            
            cur_round = db.get_current_round()

            if(cur_round):
                print("Here")
                print("cur_round : ",cur_round.current_round)
                print("There")
                client_threshold = db.get_round_threshold()
                client_count = db.get_client_round_num()
                print("Client count : ",client_count)
                print("Client threshold : ",client_threshold)
                print(client_count >= client_threshold)
                if(client_count >= client_threshold):
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
                    cur_model_id = db.get_model_id(cur_round.current_round)
                    #path = os.path.join(current_app.instance_path, f"super_round_{cur_round.super_round_id}/training_round_{cur_round.round_id}/{new_model_id}.pth")
                    round_path = db.get_model_path(current_app.instance_path,cur_model_id)
                    cur_model = HARSModel("cpu")
                    cur_model.load_state_dict(torch.load(round_path))

                    # need to get all the clients who are going to be aggregated
                    # need to combine them or something/format them so they can be called in the server.aggregate_models
                    # neet to pass in the values to the agggregate_models function
                    # need to get results and save them 

                    # format clients
                    client_list_states = []
                    client_ids = db.get_round_client_list2(cur_model_id)
                    for c_idx in range (0,len(client_ids)):
                        
                        client_path = db.get_client_model(current_app.instance_path,client_ids[c_idx][0],cur_model_id)
                        print("Client id: ",client_ids[c_idx][0])
                        print("Loading : ",client_path)
                        client_model = HARSModel("cpu")
                        client_model.load_state_dict(torch.load(client_path))
                        client_state = client_model.state_dict()
                        client_list_states.append(client_state)
                        db.flag_client_training(client_ids[c_idx][0],cur_model_id,0)

                    aggregate_states = agg_model(client_list_states,cur_model.state_dict())

                    # save the aggregate model         
                    cur_model.load_state_dict(aggregate_states)
                    torch.save(cur_model.state_dict(), round_path)

                    max_round = db.get_max_rounds()
                    db.update_aggregate(0)
                    if(cur_round.current_round+1 < max_round[0]):
                        print("Incrementing round")
                        # implement new round logic
                        db.update_round()
                        new_round =  db.get_current_round()
                        print(new_round.current_round)
                        new_model_id = db.create_model(new_round.round_id)
                        model = HARSModel("cpu")
                        #new_model_id = db.get_model_id(new_round.round_id)
                        #path = os.path.join(current_app.instance_path, f"super_round_{new_round.super_round_id}/training_round_{new_round.round_id}/{new_model_id}.pth")
                        path = db.get_model_path(current_app.instance_path,new_model_id)
                        torch.save(model.state_dict(), path)
                    else:
                        print("Max rounds has been hit")
                        print("Training done")
                    


                    print("Finished aggregating ")
                    
                    
                    ## Do validation
                    print("Validating results")
                    # CODE GOES HERE
                    # get current path
                    # pass path to the validaiton function
                    # CHANGE DATA PATH
                    #test_datapath = "C:/Users/spark/Documents/git/Federated-Learning/data/train.csv"
                    #validation.validation("cpu",test_datapath,round_path)

                    # # do something with results                
                    print("Finish validating results")
                else:
                    print("Can't aggregate")
            else:
                print("Not training")
        
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