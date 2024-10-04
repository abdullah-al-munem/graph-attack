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

warnings.simplefilter('ignore')

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
    # pyg_data = Dpr2Pyg(data)
    acc_list = []
    acc_node = {}

    predict = get_predict_function(defense_model)

    for budget in tqdm.tqdm(range(1, budget_range+1)):
        print(f"For budget number: {budget}")
        
        cnt = 0
        curr_acc = {1:[], 0:[]}
        for target_node in tqdm.tqdm(node_list):
            print(f'Target node: {target_node}')
            model_attack = RND()
            model_attack.attack(adj, labels, idx_train, target_node, n_perturbations=budget)

            modified_adj = model_attack.modified_adj

            accuracy = predict(modified_adj, features, data, target_node)

            print("accuracy = ", accuracy)
            if accuracy == 0:
                curr_acc[0].append(target_node)
                cnt += 1
            else:
                curr_acc[1].append(target_node)
            
        acc_node[budget] = curr_acc
        acc_list.append([budget, cnt / len(node_list), node_list])
        
        print(f"Total Target nodes: {len(node_list)}")
        print('Miss-classification rate Modified : %s' % (cnt / len(node_list)))
    
    df = pd.DataFrame(acc_list, columns =['budget_number', 'miss-classification_modified', 'node_list']) 
    df.to_csv(f'random_attack_{dataset}_{defense_model}_{times}.csv') ## please change the number accordingly 


    # Create two line charts for col1 and col2
    plt.figure(figsize=(8, 6))

    # Line chart for col1
    plt.plot(df['budget_number'], df['miss-classification_modified'], label='Modified Adj', marker='o', markersize=5, linestyle='-')

    # plt.ylim(bottom=0.10, top=0.50)

    # Add labels and a legend
    plt.xlabel('target_number') 
    plt.ylabel('miss-classification')
    plt.title(f'random_attack_{dataset}_{defense_model}_{times}')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig(f'random_attack_{dataset}_{defense_model}_{times}.png')
    # plt.show()

def start_attack_FGA(dataset, defense_model, budget_range, node_list, times=1):
    data = get_dataset_from_deeprobust(dataset)
    adj, features, labels, idx_train, idx_val, idx_test = destructuring_dataset(data)
    acc_list = []
    acc_node = {}

    predict = get_predict_function(defense_model)

    for budget in tqdm.tqdm(range(1, budget_range+1)):
        print(f"For budget number: {budget}")
        
        cnt = 0
        curr_acc = {1:[], 0:[]}
        for target_node in tqdm.tqdm(node_list):
            print(f'Target node: {target_node}')
            surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
            surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30, train_iters=100)
            model_attack = FGA(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device).to(device)
            model_attack.attack(features, adj, labels, idx_train, target_node, n_perturbations=budget)

            modified_adj = model_attack.modified_adj

            accuracy = predict(modified_adj, features, data, target_node)
            print("accuracy = ", accuracy)
            if accuracy == 0:
                curr_acc[0].append(target_node)
                cnt += 1
            else:
                curr_acc[1].append(target_node)
            
        acc_node[budget] = curr_acc
        acc_list.append([budget, cnt / len(node_list), node_list])
        
        print(f"Total Target: {len(node_list)}")
        print('Miss-classification rate Modified : %s' % (cnt / len(node_list)))
    
    df = pd.DataFrame(acc_list, columns =['budget_number', 'miss-classification_modified', 'node_list']) 
    df.to_csv(f'FGA_{dataset}_{defense_model}_{times}.csv') ## please change the number accordingly 


    # Create two line charts for col1 and col2
    plt.figure(figsize=(8, 6))

    # Line chart for col1
    plt.plot(df['budget_number'], df['miss-classification_modified'], label='Modified Adj', marker='o', markersize=5, linestyle='-')

    # plt.ylim(bottom=0.10, top=0.50)

    # Add labels and a legend
    plt.xlabel('target_number') 
    plt.ylabel('miss-classification')
    plt.title(f'FGA_{dataset}_{defense_model}_{times}')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig(f'FGA_{dataset}_{defense_model}_{times}.png')
    # plt.show()

def start_attack_Nettack(dataset, defense_model, budget_range, node_list, times=1):
    data = get_dataset_from_deeprobust(dataset)
    adj, features, labels, idx_train, idx_val, idx_test = destructuring_dataset(data)
    acc_list = []
    acc_node = {}

    predict = get_predict_function(defense_model)

    for budget in tqdm.tqdm(range(1, budget_range+1)):
        print(f"For budget number: {budget}")
        
        cnt = 0
        curr_acc = {1:[], 0:[]}
        for target_node in tqdm.tqdm(node_list):
            print(f'Target node: {target_node}')
            surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
            surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30, train_iters=100)
            model_attack = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device).to(device)
            model_attack.attack(features, adj, labels, target_node, n_perturbations=budget)

            modified_adj = model_attack.modified_adj

            accuracy = predict(modified_adj, features, data, target_node)
            print("accuracy = ", accuracy)
            if accuracy == 0:
                curr_acc[0].append(target_node)
                cnt += 1
            else:
                curr_acc[1].append(target_node)
            
        acc_node[budget] = curr_acc
        acc_list.append([budget, cnt / len(node_list), node_list])
        
        print(f"Total Target: {len(node_list)}")
        print('Miss-classification rate Modified : %s' % (cnt / len(node_list)))

    df = pd.DataFrame(acc_list, columns =['budget_number', 'miss-classification_modified', 'node_list']) 
    df.to_csv(f'Nettack_{dataset}_{defense_model}_{times}.csv') ## please change the number accordingly 

    # Create two line charts for col1 and col2
    plt.figure(figsize=(8, 6))

    # Line chart for col1
    plt.plot(df['budget_number'], df['miss-classification_modified'], label='Modified Adj', marker='o', markersize=5, linestyle='-')

    # plt.ylim(bottom=0.10, top=0.50)

    # Add labels and a legend
    plt.xlabel('target_number') 
    plt.ylabel('miss-classification')
    plt.title(f'Nettack_{dataset}_{defense_model}_{times}')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig(f'Nettack_{dataset}_{defense_model}_{times}.png')
    # plt.show()

def start_attack_SGAttack(dataset, defense_model, budget_range, node_list, times=1):
    data = get_dataset_from_deeprobust(dataset)
    adj, features, labels, idx_train, idx_val, idx_test = destructuring_dataset(data)
    pyg_data = Dpr2Pyg(data)

    acc_list = []
    acc_node = {}

    predict = get_predict_function(defense_model)

    for budget in tqdm.tqdm(range(1, budget_range+1)):
        print(f"For budget number: {budget}")
        
        cnt = 0
        curr_acc = {1:[], 0:[]}
        for target_node in tqdm.tqdm(node_list):
            print(f'Target node: {target_node}')
            surrogate = SGC(nfeat=features.shape[1],
                nclass=labels.max().item() + 1, K=2,
                lr=0.01, device=device).to(device)
            
            surrogate.fit(pyg_data, verbose=False, patience=30, train_iters=100)  
            model_attack = SGAttack(surrogate, attack_structure=True, attack_features=False, device=device)
            model_attack = model_attack.to(device)
            model_attack.attack(features, adj, labels, target_node, budget, direct=True)

            modified_adj = model_attack.modified_adj
            # pyg_data = Dpr2Pyg(data)
            # print(pyg_data)

            accuracy = predict(modified_adj, features, data, target_node)
            print("accuracy = ", accuracy)
            if accuracy == 0:
                curr_acc[0].append(target_node)
                cnt += 1
            else:
                curr_acc[1].append(target_node)
            
        acc_node[budget] = curr_acc
        acc_list.append([budget, cnt / len(node_list), node_list])
        
        print(f"Total Target: {len(node_list)}")
        print('Miss-classification rate Modified : %s' % (cnt / len(node_list)))

    df = pd.DataFrame(acc_list, columns =['budget_number', 'miss-classification_modified', 'node_list']) 
    df.to_csv(f'SGAttack_{dataset}_{defense_model}_{times}.csv') ## please change the number accordingly 

    # Create two line charts for col1 and col2
    plt.figure(figsize=(8, 6))

    # Line chart for col1
    plt.plot(df['budget_number'], df['miss-classification_modified'], label='Modified Adj', marker='o', markersize=5, linestyle='-')

    # plt.ylim(bottom=0.10, top=0.50)

    # Add labels and a legend
    plt.xlabel('target_number') 
    plt.ylabel('miss-classification')
    plt.title(f'SGAttack_{dataset}_{defense_model}_{times}')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig(f'SGAttack_{dataset}_{defense_model}_{times}.png')
    # plt.show()


def start_attack_IGAttack(dataset, defense_model, budget_range, node_list, times=1):
    data = get_dataset_from_deeprobust(dataset)
    adj, features, labels, idx_train, idx_val, idx_test = destructuring_dataset(data)
    acc_list = []
    acc_node = {}

    predict = get_predict_function(defense_model)

    for budget in tqdm.tqdm(range(1, budget_range+1)):
        print(f"For budget number: {budget}")
        
        cnt = 0
        curr_acc = {1:[], 0:[]}
        for target_node in tqdm.tqdm(node_list):
            print(f'Target node: {target_node}')
            surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, device=device)

            surrogate = surrogate.to(device)
            surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30, train_iters=100)
            
            model_attack = IGAttack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
            model_attack = model_attack.to(device)

            model_attack.attack(features, adj, labels, idx_train, target_node, budget, steps=20)

            modified_adj = model_attack.modified_adj

            accuracy = predict(modified_adj, features, data, target_node)
            print("accuracy = ", accuracy)
            if accuracy == 0:
                curr_acc[0].append(target_node)
                cnt += 1
            else:
                curr_acc[1].append(target_node)
            
        acc_node[budget] = curr_acc
        acc_list.append([budget, cnt / len(node_list), node_list])
        
        print(f"Total Target: {len(node_list)}")
        print('Miss-classification rate Modified : %s' % (cnt / len(node_list)))

    df = pd.DataFrame(acc_list, columns =['budget_number', 'miss-classification_modified', 'node_list']) 
    df.to_csv(f'IGAttack_{dataset}_{defense_model}_{times}.csv') ## please change the number accordingly 

    # Create two line charts for col1 and col2
    plt.figure(figsize=(8, 6))

    # Line chart for col1
    plt.plot(df['budget_number'], df['miss-classification_modified'], label='Modified Adj', marker='o', markersize=5, linestyle='-')

    # plt.ylim(bottom=0.10, top=0.50)

    # Add labels and a legend
    plt.xlabel('target_number') 
    plt.ylabel('miss-classification')
    plt.title(f'IGAttack_{dataset}_{defense_model}_{times}')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig(f'IGAttack_{dataset}_{defense_model}_{times}.png')
    # plt.show()


if __name__ == "__main__":
    dataset = 'polblogs'
    defense_model = 'gcn'
    data = get_dataset_from_deeprobust(dataset=dataset)
    budget_range = 7
    node_list = get_target_node_list(data)
    start_attack_FGA(dataset, defense_model, budget_range, node_list)