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
import torch.nn as nn
import datetime
import csv 
from flcore import logger

class model_results:
    def __init__(self,accuracy_in,tp_in,tn_in,fp_in,fn_in,recall,precision,f1):
        self.accuracy = accuracy_in
        self.tp = tp_in
        self.tn = tn_in
        self.fp = fp_in
        self.fn = fn_in    
        self.all_recall = recall
        self.all_precision = precision
        self.all_f1 = f1
    def load_results(file_path):
        print("To be implemented")
    def get_all_accuracy():
        print("Hi")
    def get_all_recall():        
        print("Hi")
    def get_all_precision():
        print("Hi")
    def get_all_f1():
        print("Hi")

def validation(device,data_path='',model='',batchsize=5,threshold=0.5,save_val=0):
    ## LOAD MODEL ## 
    logger.idle("Running validation ...")
    #print("Running validation ...")
    device = device
    if torch.cuda.is_available():
        device = "cuda"
    if(data_path == ''):
        data_path = '../data/test.csv'
    if(type(model)== str):
        #print("Input is a string")
        current_app.logger.info("Input is a string")
        if(model == ''):
            print("Error a model needs to be inputted")
            current_app.logger.info("Error : a model needs to be inputted")
        elif(os.path.isfile(model)):
            print("Load model from file")
            # Need to see what are the names of the models as they might just all be model.py 
            # need to see where the results should be saved 
            filename = os.path.splitext(os.path.basename(model))
            savefile = filename[0] + ".csv"           
            
            model = load_HARSModel(device,model)
    elif(not isinstance(model,nn.Module)):
        #print("Error input is not a model/n")
        current_app.logger.info("Error input is no a model")
        return -1
    
    ## SET UP MODEL ##
    model.to(device)
    model.eval()
    val_set = HARSDataset(data_path)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size = 1)

    total_correct = 0
    total_test =0
    tp= [0]*len(val_set.class_list)
    fp= [0]*len(val_set.class_list)
    fn= [0]*len(val_set.class_list)
    tn = [0]*len(val_set.class_list)


    for data in val_loader:
        temp = data[0]
        label = data[1]
        #label = data[1].argmax().item()
        temp.to(device)
        #label.to(device)
        output = model(temp)
        pred_label = output.argmax(1).tolist()
        pred_np = np.array(pred_label)
        label_np = np.array(label.argmax(1).tolist())
        total_correct = total_correct+ (pred_np == label_np).sum().item()
        total_test =total_test+  len(label)
        tp,fp,fn,tn = get_model_true_false(pred_label,label_np,val_set.class_list,tp,fp,fn,tn)
        
    model_accuracy = total_correct/total_test 
    #print("Accuracy: {},".format(model_accuracy))
    logger.info("Accuracy: {},".format(model_accuracy))

    agg_tp, agg_fp,agg_fn,agg_tn = aggregated_confusion_values(tp,fp,fn,tn)
    class_weights = weights(tp,tn)

    ## RECALL
    recall_micro = micro_recall(agg_tp,agg_fn)
    class_recall,recall_macro = get_recall(tp,fn)
    recall_weight = weighted_recall(class_recall,class_weights)

    #print("Micro Recall {}, Macro Recall {}, Weighted Recall {}\n".format(recall_micro,recall_macro,recall_weight))
    logger.info("Micro Recall {}, Macro Recall {}, Weighted Recall {}\n".format(recall_micro,recall_macro,recall_weight))

    ## Precision
    precision_micro = micro_precision(agg_tp,agg_fp)
    class_precision,precision_macro = get_macro_precision(tp,fp)
    precision_weight = weighted_precision(class_precision,class_weights)

    print("Micro Precision {}, Macro Precision {}, Weighted Precision {}\n".format(precision_micro,precision_macro,precision_weight))
    logger.info("Micro Precision {}, Macro Precision {}, Weighted Precision {}\n".format(precision_micro,precision_macro,precision_weight))
    ## F1
    f1_micro = micro_f1(precision_micro,recall_micro)
    f1_macro = macro_f1(class_precision,class_recall)
    f1_weight = weighted_f1_score(class_precision,class_recall)

    print("F1 Precision {}, F1 Precision {}, F1 Precision {}\n".format(f1_micro,f1_macro,f1_weight))
    logger.info("F1 Precision {}, F1 Precision {}, F1 Precision {}\n".format(f1_micro,f1_macro,f1_weight))
    ## AUC
    tpr_macro, fpr_macro = macro_tpr_fpr(tp,tn,fp,fn,class_recall)
    tpr_micro, fpr_micro = micro_tpr_fpr(agg_tp,agg_tn,agg_fp,agg_fn,recall_micro)    
    class_auc,auc_macro = macro_auc(tpr_macro,fpr_macro)
    auc_micro = micro_auc(tpr_micro,fpr_micro)
    auc_weight = weighted_auc(class_auc,class_weights)
    print("AUC Precision {}, AUC Precision {}, AUC Precision {}\n".format(auc_micro,auc_macro,auc_weight))
    logger.info("AUC Precision {}, AUC Precision {}, AUC Precision {}\n".format(auc_micro,auc_macro,auc_weight))
    results = model_results(model_accuracy,tp,fp,fn,tn,recall_macro,precision_macro,f1_macro)
    if(save_val):
        with open(savefile,"w",newline="") as file:
            writer = csv.writer(file)
            export_data = [
                ["accuracy",results.accuracy],
                ["tp",results.tp],
                ["tn",results.tn],
                ["fp",results.fp],
                ["fn",results.fn]
            ]
            writer.writerows(export_data) 
        file.close()   
    return results
    
def get_tpr_fpr(recall=0,tp=0,tn=0,fp=0,fn=0,ground_truth=[],pred_label=[],class_list=[]):    
    if(not ground_truth and not pred_label and not class_list and recall  ==0 and tp ==0 and tn ==0 and fp == 0 and fn ==0):
        tp =[]
        fp =[]
        tn=[]
        fn = []
        tp,fp,fn,tn = get_model_true_false(pred_label,label_np,val_set.class_list,tp,fp,fn,tn)
    if(recall == 0 and tp != 0 and tn != 0 and fp != 0 and fn != 0):
        tpr = tp/tp+fn 
    else:
        trp = recall

    fpr = fp/(fp+tn)
    return tpr,fpr

def get_recall(tp,fn):
    recall = [0] * len(tp)
    for i in range(0,len(tp)):
        denom = tp[i]+fn[i]
        if(denom == 0):
            recall[i] =0
        else:
            recall[i] = tp[i]/denom
    if(len(recall) == 0):
        macro_recall = 0
    else:
        macro_recall = sum(recall)/len(recall)
    return recall,macro_recall

def micro_recall(agg_tp,agg_fn):
    denom = (agg_tp+agg_fn)
    if(denom==0):
        recall = 0
    else:
        recall = agg_tp/denom
    return recall

def weighted_recall(class_recall,weight):
    weighted_val = [0] * len(class_recall)
    for i in range(0,len(class_recall)):
        weighted_val[i] = class_recall[i] * weight[i]
    if(sum(weight)==0):
        recall =0
    else:
        recall = sum(weighted_val)/sum(weight)
    return recall

def get_macro_precision(tp,fp):
    precision = [0] * len(tp)
    for i in range(0,len(tp)):
        denom = (tp[i]+fp[i])
        if(denom ==0):
            precision[i] =0
        else:
            precision[i] = tp[i]/denom
    if(len(precision)==0):
        macro_precision =0
    else: 
        macro_precision = sum(precision)/len(precision)
    return precision,macro_precision

def micro_precision(agg_tp,agg_fp):
    denom = (agg_tp+agg_fp)
    if(denom == 0):
        return 0
    else:
        return agg_tp/denom

def weighted_precision(class_precision,weight):
    weighted_val = [0] * len(class_precision)
    for i in range(0,len(class_precision)):
        weighted_val[i] = class_precision[i] * weight[i]
    if(sum(weight)==0):
        precision = 0
    else:
        precision = sum(weighted_val)/sum(weight)
    return precision

def macro_f1(precision,recall):
    f1_score = [0]*len(precision)
    for i in range(0,len(precision)):
        denom = (precision[i]+recall[i])
        if(denom == 0):
            f1_score[i] = 0
        else:
            f1_score[i] = 2*((precision[i]*recall[i])/denom)
    if(len(f1_score)==0):
        macro_f1_score = 0
    else:
        macro_f1_score = sum(f1_score)/len(f1_score)
    return macro_f1_score

def micro_f1(agg_precision,agg_recall):
    denom = (agg_precision+agg_recall)
    if(denom == 0):
        f1_score =0
    else:
        f1_score = 2*((agg_precision*agg_recall)/denom)
    return f1_score

def weighted_f1_score(class_precision,weight):
    weighted_val = [0] * len(class_precision)
    for i in range(0,len(class_precision)):
        weighted_val[i] = class_precision[i] * weight[i]
    if(sum(weight)==0):
        f1_score = 0
    else:
        f1_score = sum(weighted_val)/sum(weight)
    return f1_score

def macro_tpr_fpr(tp,tn,fp,fn,recall=0):
    if(recall == 0):
        tpr,_ = get_recall(tp,tn,fp,fn)
    else:
        tpr = recall
    fpr = [0]*len(tp)
    for i in range(0,len(tp)):
        if(fp[i]==0):
            fpr[i] =0
        else:
            denom = (fp[i]+tn[i])
            if(denom ==0):
                fpr[i] = 0
            else:
                fpr[i] = fp[i]/denom
    return tpr,fpr

def micro_tpr_fpr(agg_tp,agg_tn,agg_fp,agg_fn,recall=0):
    if(recall == 0):
        tpr,_ = micro_recall(agg_tp,agg_tn,agg_fp,agg_fn)
    else:
        tpr = recall
    if(agg_fp==0):
        fpr =0
    else:
        denom = (agg_fp+agg_tn)
        if(denom == 0):
            fpr = 0
        else:
            fpr = agg_fp/denom
    return tpr,fpr

def micro_auc(tpr,fpr):
    auc = np.trapezoid([tpr],[fpr])
    return auc
def macro_auc(tpr,fpr):
    class_auc = [0] * len(tpr)
    for i in range(0,len(tpr)):
        class_auc[i] = np.trapezoid([tpr[i]],[fpr[i]])
    if(len(class_auc)==0):
        macro_auc =0
    else:
        macro_auc = sum(class_auc)/len(class_auc)
    return class_auc,macro_auc

def weighted_auc(class_precision,weight):
    weighted_val = [0] * len(class_precision)
    for i in range(0,len(class_precision)):
        weighted_val[i] = class_precision[i] * weight[i]
    if(sum(weight)==0):
        auc =0
    else:
        auc = sum(weighted_val)/sum(weight)
    return auc

def aggregated_confusion_values(tp,tn,fp,fn):
    agg_tp = sum(tp)
    agg_tn = sum(tn)
    agg_fp = sum(fp)
    agg_fn = sum(fn)
    return agg_tp,agg_tn,agg_fp,agg_fn

def weights(tp,tn):
    tp_weight = [0]*len(tp)
    for i in range(0,len(tp)):
        tp_weight[i] = tp[i] + tn[i]
    return tp_weight 



    
def get_model_true_false(pred,truth,class_list,tp,fp,fn,tn):

    for c_idx in range(0,len(class_list)):
        for idx in range(0,len(pred)):
            if(pred[idx]==c_idx and truth[idx]==c_idx):
                tp[c_idx] = tp[c_idx]+1
            elif(pred[idx]!=c_idx and truth[idx]==c_idx):
                fp[c_idx] = fp[c_idx]+1
            elif(pred[idx]==c_idx and truth[idx]!=c_idx):
                fn[c_idx] = fn[c_idx]+1
            elif(pred[idx]==c_idx and truth[idx]==c_idx):
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