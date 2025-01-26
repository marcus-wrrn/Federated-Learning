import threading
import requests
#from config import TrainingConfig, Hyperparameters, CoordinationServerResponse, ClientState
import torch
from torch.utils.data import DataLoader
from flcore.models.basic import HARSModel
from flcore.data_handling.datasets import HARSDataset
import os
import numpy as np
import torch.utils.data.dataloader
#from sklearn.metrics import confusion_matrix

def validation(device,data_path='',model='',batchsize=5):
    print("Running validation ...")
    device = device
    if torch.cuda.is_available():
        device = "cuda"
    if(data_path == ''):
        data_path = '../data/test.csv'
    if(type(model)== str):
        print("Input is a string")
        if(model == ''):
            print("Error a model needs to be inputted")
        elif(os.path.isfile(model)):
            print("Load model from file")
            model = load_HARSModel(device,model)
    #else(model.instance())
    # check to see if inputted model is actally the correct model 

    model.to(device)
    model.eval()
    val_set = HARSDataset(data_path)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size = 1)

    total_correct = 0
    total_test =0
    for data in val_loader:
        temp = data[0]
        label = data[1]
        #label = data[1].argmax().item()
        temp.to(device)
        #label.to(device)
        output = model(temp)
        pred_label = output.argmax(1).tolist()
        temp2 = output
        pred_np = np.array(pred_label)
        label_np = np.array(label.argmax(1).tolist())
        total_correct = total_correct+ (pred_np == label_np).sum().item()
        total_test =total_test+  len(label)

    model_accuracy = total_correct/total_test 
    #print(total_test)
    #print(len(val_set))
    print(model_accuracy)
        

    
def get_model_true_false(pred,truth,class_list):
    tp =[]
    fp = []
    fn =[]
    tn = []

    for c_idx in range(0,len(class_list)):
        for idx in range(0,len(pred)):
            if(pred[idx]==class_list and truth[idx]==class_list):
                tp[c_idx] = tp[c_idx]+1
            elif(pred[idx]!=class_list and truth[idx]==class_list):
                fp[c_idx] = fp[c_idx]+1
            elif(pred[idx]==class_list and truth[idx]!=class_list):
                fn[c_idx] = fn[c_idx]+1
            elif(pred[idx]==class_list and truth[idx]==class_list):
                tn[c_idx] = tn[c_idx]+1
    return tp,fp,fn,tn
            
def load_HARSModel(device,model_path):
    #print("Loading here")
    temp_model = HARSModel(device)
 #   if (device == "cpu"):
#        temp_model.load_state_dict(
#            torch.load(temp_model, map_location=torch.device('cpu'), weights_only=True))
#    else:
#        temp_model.load_state_dict(torch.load(model_path, weights_only=True))
    temp_model.load_state_dict(torch.load(model_path, weights_only=True))

    temp_model.eval()
    return temp_model

def main():
    print("Running main")
    device = "cpu"
    temp = HARSModel(device)
    print(type(temp))
    #print(temp.isinstance())
    temp2 = "C:/Users/spark/Documents/git/Federated-Learning/src/core_model/flcore/HARSModel2492512025/model.pth"
    print(type(temp2))
    print(type(temp2)== str)
if __name__ == '__main__':
    #main()
    validation("cpu","C:/Users/spark/Documents/git/Federated-Learning/data/test.csv","C:/Users/spark/Documents/git/Federated-Learning/src/core_model/flcore/HARSModel2492512025/model.pth")