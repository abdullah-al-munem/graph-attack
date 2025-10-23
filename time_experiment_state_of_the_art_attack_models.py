import os
import collections
import pandas as pd
import matplotlib.pyplot as plt
import logging
import warnings
import torch
import tqdm

from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.defense import GCN
from deeprobust.graph.defense import GAT
from deeprobust.graph.defense import SGC
from GIN import GIN
from GSAGE import GraphSAGE

from deeprobust.graph.targeted_attack import RND
from deeprobust.graph.targeted_attack import FGA
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.targeted_attack import SGAttack
from deeprobust.graph.targeted_attack import IGAttack



from utils import get_dataset_from_deeprobust, destructuring_dataset, get_predict_function, get_target_node_list
import time

warnings.simplefilter('ignore')
def convert_time(seconds):
    minutes = seconds // 60
    seconds %= 60
    return minutes, seconds

def get_device():

    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return device

device = get_device()

def start_attack_RND(dataset, defense_model, budget_range, node_list, times=1):
    data = get_dataset_from_deeprobust(dataset)
    adj, features, labels, idx_train, idx_val, idx_test = destructuring_dataset(data)


    predict = get_predict_function(defense_model)
    start_time = time.time()
    print(f"For budget number: {budget_range}")
    
    cnt = 0
    curr_acc = {1:[], 0:[]}
    for target_node in tqdm.tqdm(node_list):
        print(f'Target node: {target_node}')
        model_attack = RND()
        model_attack.attack(adj, labels, idx_train, target_node, n_perturbations=budget_range)

        modified_adj = model_attack.modified_adj

    end_time = time.time()
              
    running_time_seconds = end_time - start_time
    running_time_minutes, running_time_seconds = convert_time(running_time_seconds)
    print(f"running_time: {int(running_time_minutes)} minutes, {running_time_seconds} seconds")
    print(dataset, defense_model, budget_range, node_list)
    print("================================================")
    return running_time_minutes, running_time_seconds

def start_attack_FGA(dataset, defense_model, budget_range, node_list, times=1):
    data = get_dataset_from_deeprobust(dataset)
    adj, features, labels, idx_train, idx_val, idx_test = destructuring_dataset(data)
    

    predict = get_predict_function(defense_model)
    start_time = time.time()
    print(f"For budget number: {budget_range}")
    
    cnt = 0
    curr_acc = {1:[], 0:[]}
    for target_node in tqdm.tqdm(node_list):
        print(f'Target node: {target_node}')
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30, train_iters=100)
        model_attack = FGA(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device).to(device)
        model_attack.attack(features, adj, labels, idx_train, target_node, n_perturbations=budget_range)
    end_time = time.time()
              
    running_time_seconds = end_time - start_time
    running_time_minutes, running_time_seconds = convert_time(running_time_seconds)
    print(f"running_time: {int(running_time_minutes)} minutes, {running_time_seconds} seconds")
    print(dataset, defense_model, budget_range, node_list)
    print("================================================")
    return running_time_minutes, running_time_seconds
            

def start_attack_Nettack(dataset, defense_model, budget_range, node_list, times=1):
    data = get_dataset_from_deeprobust(dataset)
    adj, features, labels, idx_train, idx_val, idx_test = destructuring_dataset(data)

    predict = get_predict_function(defense_model)
    start_time = time.time()
    print(f"For budget number: {budget_range}")
    
    cnt = 0
    curr_acc = {1:[], 0:[]}
    for target_node in tqdm.tqdm(node_list):
        print(f'Target node: {target_node}')
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30, train_iters=100)
        model_attack = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device).to(device)
        model_attack.attack(features, adj, labels, target_node, n_perturbations=budget_range)
    end_time = time.time()
              
    running_time_seconds = end_time - start_time
    running_time_minutes, running_time_seconds = convert_time(running_time_seconds)
    print(f"running_time: {int(running_time_minutes)} minutes, {running_time_seconds} seconds")
    print(dataset, defense_model, budget_range, node_list)
    print("================================================")
    return running_time_minutes, running_time_seconds
            
def start_attack_SGAttack(dataset, defense_model, budget_range, node_list, times=1):
    if dataset == "blogcatalog":
        device = "cpu"
    else:
        device = get_device()
    data = get_dataset_from_deeprobust(dataset)
    adj, features, labels, idx_train, idx_val, idx_test = destructuring_dataset(data)
    pyg_data = Dpr2Pyg(data)
    start_time = time.time()
    surrogate = SGC(nfeat=features.shape[1],
                nclass=labels.max().item() + 1, K=2,
                lr=0.01, device=device).to(device)
            
    surrogate.fit(pyg_data, verbose=False, patience=30, train_iters=100) 

    del pyg_data

    acc_list = []
    acc_node = {}
    
    print(f"For budget number: {budget_range}")
    
    cnt = 0
    curr_acc = {1:[], 0:[]}
    for target_node in tqdm.tqdm(node_list):
        
        print(f'Target node: {target_node}')
            
        model_attack = SGAttack(surrogate, attack_structure=True, attack_features=False, device=device)
        model_attack = model_attack.to(device)
        model_attack.attack(features, adj, labels, target_node, budget_range, direct=True)
    end_time = time.time()
              
    running_time_seconds = end_time - start_time
    running_time_minutes, running_time_seconds = convert_time(running_time_seconds)
    print(f"running_time: {int(running_time_minutes)} minutes, {running_time_seconds} seconds")
    print(dataset, defense_model, budget_range, node_list)
    print("================================================")
    return running_time_minutes, running_time_seconds
            


if __name__ == "__main__":
    dataset = 'polblogs'
    defense_model = 'gcn'
    data = get_dataset_from_deeprobust(dataset=dataset)
    budget_range = 7
    node_list = get_target_node_list(data)
    start_attack_FGA(dataset, defense_model, budget_range, node_list)