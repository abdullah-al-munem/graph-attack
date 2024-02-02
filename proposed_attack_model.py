import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tqdm
import json
import collections

import networkx as nx

import logging
import warnings

import torch
import torch.nn as nn
import torch_geometric
from torch.nn import init
import torch_geometric.transforms as T
# from pygod.utils import load_data
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures

from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.defense import GCN
from deeprobust.graph.defense import GAT
from GIN import GIN
from GSAGE import GraphSAGE


from autoencoders import GAEModel, VGAEModel

from predict import test_GCN, test_GAT, test_GIN, test_GraphSAGE

from utils import get_dataset_from_deeprobust, destructuring_dataset, get_predict_function, get_target_node_list

warnings.simplefilter('ignore')


log_file_name = "./attack.log"

# Check if the file exists
if os.path.exists(log_file_name):
    if os.path.exists(f"{log_file_name}.bak"):
        os.remove(f"{log_file_name}.bak")
        
    os.rename(log_file_name, f"{log_file_name}.bak")
    # Remove the original file
    # os.remove(f"{log_file_name}")



# Configure logging to both console and file
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Log to console
                        logging.FileHandler(log_file_name)  # Log to a file
                    ])

# Create a logger
logger = logging.getLogger(__name__)
logger.info('The attack has been started...')

# Example usage
# logger.debug('This is a debug message')
# logger.info('This is an info message')
# logger.warning('This is a warning message')
# logger.error('This is an error message')
# logger.critical('This is a critical message')
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

# add_remove_stat = collections.defaultdict(lambda : 0)

class ProposedAttack:
    def __init__(self, surrogate_model, dataset, defense_model, important_edge_list=None, attack_structure=True, attack_features=False, device=device):
        data = Dataset(root=r'./', name=dataset) # load clean graph
        self.dataset = dataset
        self.data2 = data
        self.pyg_data = Dpr2Pyg(data) # convert dpr to pyg
        self.data = self.pyg_data[0]
        self.adj = self.pyg_data[0].edge_index.t()
        self.features = self.pyg_data[0].x
        self.labels = self.pyg_data[0].y
        self.surrogate = surrogate_model.lower()
        self.device = device
        self.defense_model = defense_model.lower()
        self.predict = self.get_predict_function()
        self.surrogate_model_output = self.get_surrogate_model_output()
        # print(len(self.surrogate_model_output))
        # print(len(self.labels), type(self.labels))
        self.important_edge_list = important_edge_list
        # if important_edge_list != None:
        #     exit()
    
    def get_predict_function(self):
        predict_classes = {
            'gcn': test_GCN,
            'gin': test_GIN,
            'gat': test_GAT,
            'graphsage': test_GraphSAGE
        }
        return predict_classes[self.defense_model]
        

    def get_surrogate_model_output(self):
        adj2, features2, labels2 = self.data2.adj, self.data2.features, self.data2.labels
        idx_train2, idx_val2, idx_test2 = self.data2.idx_train, self.data2.idx_val, self.data2.idx_test
        output = None
        if self.surrogate == 'gcn':
            gcn = GCN(nfeat=features2.shape[1], nhid=16, nclass=labels2.max().item() + 1, dropout=0.5, device=device)
            gcn = gcn.to(device)
            gcn.fit(features2, adj2, labels2, idx_train2, idx_val2, patience=30, train_iters=100)
            gcn.eval()
            output = gcn.predict()
            # print(output)
            # exit()
        elif self.surrogate == 'gin':
            gin = GIN(nfeat=self.features.shape[1], nhid=8, heads=8, nclass=self.labels.max().item() + 1, dropout=0.5, device=device)
            gin = gin.to(device)
            gin.fit(self.pyg_data, verbose=False) 
            gin.eval()
            output = gin.predict()
        elif self.surrogate == 'gat':
            gat = GAT(nfeat=self.features.shape[1], nhid=8, heads=8, nclass=self.labels.max().item() + 1, dropout=0.5, device=device)
            gat = gat.to(device)
            gat.fit(self.pyg_data, verbose=False)  
            gat.eval()
            output = gat.predict()
        elif self.surrogate == 'graphsage':
            graphsage = GraphSAGE(nfeat=self.features.shape[1], nhid=8, heads=8, nclass=self.labels.max().item() + 1, dropout=0.5, device=device)
            graphsage = graphsage.to(device)
            graphsage.fit(self.pyg_data, verbose=False) 
            graphsage.eval()
            output = graphsage.predict()
                    
        assert output is not None, "Surrogate model should be any one of the listed model: [gcn, gin, gat, graphsage]"
        return output
    
    def get_degree(self):
        degree_dict = {}
        for (u, v) in self.data.edge_index.t():
            if int(u) in degree_dict:
                degree_dict[int(u)] += 1
            else:
                degree_dict[int(u)] = 1
        return degree_dict
    
    def get_nodes_to_connect_by_distance(self, adj, target_node):
        G = nx.Graph()
        G.add_edges_from(adj)

        # Dictionary to store shortest distances for each node
        shortest_distances = {}
        label = self.surrogate_model_output.argmax(1)[target_node]

        # Calculate shortest distances for each node in the node list
        shortest_paths = nx.single_source_dijkstra_path_length(G, target_node)
        shortest_distances[target_node] = shortest_paths
        distances_from_specific_node = shortest_distances[target_node]
        top_nodes_from_specific_node = sorted(distances_from_specific_node.items(), key=lambda x: x[1], reverse=True)

        for node in top_nodes_from_specific_node:
            return node[0]
        #     if self.labels[node[0]] != label:
        #         # print(f'Distance between {target_node} and {node[0]} is {node[1]}')
        #         return node[0]

        # print(shortest_distances, len(adj))
        # print(shortest_paths)
        
        # print("Nothing found!!")

    def get_decoded_edge_index_from_GAE(self):
        # Normalize node features
        normalizer = T.NormalizeFeatures()
        normalize_data = normalizer(self.data)

        # Apply RandomLinkSplit transformation
        splitter = T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, split_labels=True, add_negative_train_samples=False)
        split_data = splitter(normalize_data)

        # Split data into train, validation, and test sets
        train_data, val_data, test_data = split_data

        in_channels, out_channels = self.data.num_features, 16

        model = GAEModel(in_channels, out_channels)
        model.train_and_test(train_data, test_data)
        encoded = model.get_encoding(train_data)
        decoded = model.get_decoding(train_data, encoded)
        decoded_edge_index = model.get_decoded_edge_index(train_data, decoded)

        return decoded_edge_index

    def get_decoded_edge_index_from_VGAE(self):
        # Normalize node features
        normalizer = T.NormalizeFeatures()
        normalize_data = normalizer(self.data)

        # Apply RandomLinkSplit transformation
        splitter = T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, split_labels=True, add_negative_train_samples=False)
        split_data = splitter(normalize_data)

        # Split data into train, validation, and test sets
        train_data, val_data, test_data = split_data

        in_channels, out_channels = self.data.num_features, 16

        model = VGAEModel(in_channels, out_channels)
        model.train_and_test(train_data, test_data)
        encoded = model.get_encoding(train_data)
        decoded = model.get_decoding(train_data, encoded)
        decoded_edge_index = model.get_decoded_edge_index(train_data, decoded)

        return decoded_edge_index

    def get_important_edge_list(self):
        decoded_edge_index_GAE = self.get_decoded_edge_index_from_GAE()
        decoded_edge_index_VGAE = self.get_decoded_edge_index_from_VGAE()

        adj_list_GAE = decoded_edge_index_GAE.t().tolist()
        adj_list_VGAE = decoded_edge_index_VGAE.t().tolist()

        adj_list_GAE_set = set(tuple(item) for item in adj_list_GAE)
        adj_list_VGAE_set = set(tuple(item) for item in adj_list_VGAE)

        # Find the common elements using the intersection operation
        common_elements = adj_list_GAE_set.intersection(adj_list_VGAE_set)

        # Convert the result back to a list of lists if needed
        common_elements_list = [tuple(item) for item in common_elements]

        # Print the common elements
        return common_elements_list
    
    def attack(self, target_node, n_perturbations, isBoth=1, isAdd=0, isRemove=0):
        global add_remove_stat
        assert self.important_edge_list is not None, "Create the important_edge_list first by calling get_important_edge_list function."

        G = nx.Graph()
        G.add_edges_from(self.adj.tolist())
        degree = dict(nx.degree(G))
        
        prune_list = []
        for (u, v) in self.important_edge_list:
            if (u == target_node or v == target_node):
                if (u, v) not in prune_list:
                    prune_list.append((u, v))
                if (v, u) not in prune_list:
                    prune_list.append((v, u))
        
        final_prune_list = prune_list.copy()
        
        i = 0
        while i < len(prune_list):  
            u, v = map(int, prune_list[i])  

            if u in degree and degree[u] > 1:
                degree[u] -= 1
            else:
                final_prune_list = [lst for lst in final_prune_list if lst != prune_list[i]]
                if prune_list[i] in final_prune_list:
                    print("why here bro!")

            if v in degree and degree[v] > 1:
                degree[v] -= 1
            else:
                final_prune_list = [lst for lst in final_prune_list if lst != prune_list[i]]
                if prune_list[i] in final_prune_list:
                    print("why here bro!")
            i += 1

        updated_edges = self.adj.tolist()        
        final_prune_list_tmp = final_prune_list.copy()
        
        cnt1 = 0
        cnt2 = 0
        isLimitOver = 0
        degree = dict(nx.degree(G))
        removed_edges = []
        added_edges = []
        while n_perturbations:
            flg_d, flg_a = 0, 0
            removed_edge, added_edge = None, None
            if (isBoth or isRemove) and isLimitOver==0:
                updated_edges_tmp_2 = updated_edges.copy()
                try:
                    edge_to_remove_1 = list(final_prune_list[0])
                except:
                    isAddNewEdge=0
                    tmp = updated_edges.copy()
                    for (u, v) in tmp:
                        if (u == target_node or v == target_node) and ((u, v) not in final_prune_list_tmp):
                            if degree[u] <= 1 or degree[v] <= 1:
                                continue
                            # print(f"({u}, {v}) is not important but I am removing this anyway. ")
                            edge_to_remove_1 = [u, v]
                            isAddNewEdge=1
                            final_prune_list_tmp.append((u, v))
                            break
                    if isAddNewEdge == 0:
                        # print("Limit for removing edge is over..")
                        isLimitOver = 1
                if isLimitOver == 0:
                    edge_to_remove_2 = [edge_to_remove_1[1], edge_to_remove_1[0]]
                    flg = 0
                    flg_remove_edge_1, flg_remove_edge_2 = 0, 0
                    try:
                        updated_edges_tmp_2.remove(edge_to_remove_1)
                        flg = 1
                        flg_remove_edge_1 = 1
                        degree[edge_to_remove_1[0]]-=1
                        # print('okay1')
                    except:
                        pass
                    
                    try:
                        updated_edges_tmp_2.remove(edge_to_remove_2)
                        flg = 1
                        flg_remove_edge_2 = 1
                        degree[edge_to_remove_2[0]]-=1
                        # print('okay2')
                    except:
                        pass
                    
                    try:
                        final_prune_list.remove(tuple(edge_to_remove_1))
                    except:
                        pass
                    
                    try:
                        final_prune_list.remove(tuple(edge_to_remove_2))
                    except:
                        pass
                    
                    if flg==0:
                        print("Not removed any egdes.")
                    else:
                        pass
                        # print(f"removed edge {edge_to_remove_1}")
                    G = nx.Graph()
                    G.add_edges_from(updated_edges_tmp_2)

                    # Check if the graph is connected
                    is_connected = nx.is_connected(G)
                    
                    # assert is_connected, "Graph is disconnected."
                    if not is_connected:
                        isLimitOver = 1
                        updated_edges_tmp_2 = updated_edges.copy()
                    modified_adj = torch.tensor(updated_edges_tmp_2, dtype=torch.long)

                    # accuracy = evaluate(model, features, modified_adj, labels, target_node)
                    accuracy = self.predict(modified_adj, target_node, self.dataset)

                    removed_edge = [list(edge_to_remove_1), list(edge_to_remove_2), flg_remove_edge_1, flg_remove_edge_2]
                    flg_d = 1 if accuracy == 0 else 0
                    # print("============,", len(final_prune_list))
                    if accuracy == 0:
                        # print(f"added val: {flg_a}, removed val: {flg_d}")
                        cnt2 += 1
                        add_remove_stat['add'] += cnt1
                        add_remove_stat['remove'] += cnt2
                        print(f"added: {cnt1}, removed: {cnt2}")
                        logger.info(f"added: {cnt1}, removed: {cnt2}")
                        return modified_adj, accuracy

            if isBoth or isAdd:
                updated_edges_tmp = updated_edges.copy()
                edge_to_add = self.get_nodes_to_connect_by_distance(updated_edges_tmp, target_node)
                updated_edges_tmp.append([target_node, edge_to_add])
                # nodes_to_connect.remove(edge_to_add)
                # print(f"added edge {edge_to_add}")

                modified_adj = torch.tensor(updated_edges_tmp, dtype=torch.long)
                # try:
                # accuracy = evaluate(model, features, modified_adj, labels, target_node)
                accuracy = self.predict(modified_adj, target_node, self.dataset)

                added_edge = [target_node, edge_to_add]
                flg_a = 1 if accuracy == 0 else 0
                if accuracy == 0:
                    # print(f"added val: {flg_a}, removed val: {flg_d}")
                    cnt1 += 1
                    add_remove_stat['add'] += cnt1
                    add_remove_stat['remove'] += cnt2
                    print(f"added: {cnt1}, removed: {cnt2}")
                    logger.info(f"added: {cnt1}, removed: {cnt2}")
                    return modified_adj, accuracy
                # except:
                #     pass
                
                # print("end of loop")
            if isBoth:
                if flg_a or isLimitOver:
                    cnt1 += 1
                    # print("Len before add: ", len(updated_edges))
                    updated_edges = updated_edges_tmp.copy()
                    # print("Len after add: ", len(updated_edges))
                    added_edges.append(added_edge)
                else:
                    cnt2 += 1
                    # print("Len before delete: ", len(updated_edges))
                    updated_edges = updated_edges_tmp_2.copy()
                    # print("Len after delete: ", len(updated_edges))
                    removed_edges.append(removed_edge)
                    
                # print(f"added val: {flg_a}, removed val: {flg_d}")
                # logger.info(f"added: {cnt1}, removed: {cnt2+1}")
            elif isAdd:
                cnt1+=1
                updated_edges = updated_edges_tmp.copy()
            else:
                if isLimitOver:
                    break
                cnt2+=1
                updated_edges = updated_edges_tmp_2.copy()

            n_perturbations -= 1

        
        # add_remove_stat['add'] += cnt1
        # add_remove_stat['remove'] += cnt2
        
        # print(f"added: {cnt1}, removed: {cnt2}")
        # logger.info(f"added: {cnt1}, removed: {cnt2}")

        modified_adj = torch.tensor(updated_edges, dtype=torch.long)
        # accuracy = evaluate(model, features, modified_adj, labels, target_node)
        accuracy = self.predict(modified_adj, target_node, self.dataset)
        updated_edges_tmp = updated_edges.copy()
        # updated_edges_tmp_2 = updated_edges.copy()
        if accuracy == 0:
            add_remove_stat['add'] += cnt1
            add_remove_stat['remove'] += cnt2
            print(f"added: {cnt1}, removed: {cnt2}")
            logger.info(f"added: {cnt1}, removed: {cnt2}")
            return modified_adj, accuracy
        
        if isAdd or isRemove:
            return modified_adj, accuracy
        
        cnt_extra_check = 0
        for i, removed_edge in enumerate(removed_edges):
            # print(removed_edge)
            removed_edge_1, removed_edge_2, flg_remove_edge_1, flg_remove_edge_2 = removed_edge
            if flg_remove_edge_1:
                updated_edges_tmp.append(removed_edge_1)
            if flg_remove_edge_2:
                updated_edges_tmp.append(removed_edge_2)
            
            edge_to_add = self.get_nodes_to_connect_by_distance(updated_edges_tmp, target_node)
            updated_edges_tmp.append([target_node, edge_to_add])

            modified_adj = torch.tensor(updated_edges_tmp, dtype=torch.long)
            accuracy = self.predict(modified_adj, target_node, self.dataset)
            cnt1 += 1
            cnt2 -= 1
            cnt_extra_check += 1
            
            if accuracy == 0:
                # print(f"added val: {flg_a}, removed val: {flg_d}")
                add_remove_stat['add'] += cnt1
                add_remove_stat['remove'] += cnt2
                print(f"added: {cnt1}, removed: {cnt2}")
                logger.info(f"added: {cnt1}, removed: {cnt2}")
                return modified_adj, accuracy
        
        remove_cnt = cnt_extra_check//2
        add_cnt = cnt_extra_check - remove_cnt 

        for i in range(remove_cnt):
            if i < len(removed_edges):
                removed_edge_1, removed_edge_2, flg_remove_edge_1, flg_remove_edge_2 = removed_edges[i]
                if flg_remove_edge_1:
                    updated_edges_tmp.remove(removed_edge_1)

                if flg_remove_edge_2:
                    updated_edges_tmp.remove(removed_edge_2)
            else:
                print(f'something might be wrong: remove count: {remove_cnt}, add count {add_cnt}, len removed_edges: {len(removed_edges)}, cnt1: {cnt1}, cnt2: {cnt2} cnt_extra_check: {cnt_extra_check}')

                logger.info(f'something might be wrong: remove count: {remove_cnt}, add count {add_cnt}, len removed_edges: {len(removed_edges)}, cnt1: {cnt1}, cnt2: {cnt2} cnt_extra_check: {cnt_extra_check}')

        modified_adj = torch.tensor(updated_edges_tmp, dtype=torch.long)
        accuracy = self.predict(modified_adj, target_node, self.dataset)

        add_remove_stat['add'] += add_cnt
        add_remove_stat['remove'] += remove_cnt
        print(f"added: {add_cnt}, removed: {remove_cnt}")
        logger.info(f"added: {add_cnt}, removed: {remove_cnt}")
        return modified_adj, accuracy

            
            


        


def start_attack_proposed_model(surrogate_model, dataset, defense_model, budget_range, node_list, times=1):
    acc_list = []
    acc_node = {}

    global add_remove_stat
    add_remove_stat = collections.defaultdict(lambda : 0)
    start_budget = 1 ## this will be the function parameter (code cleaning is not done yet.)
    proposed_model = ProposedAttack(surrogate_model, dataset, defense_model)
    important_edge_list = proposed_model.get_important_edge_list()
    for budget in tqdm.tqdm(range(start_budget, budget_range+1)):
        print(f"For budget number: {budget}")
        logger.info(f"For budget number: {budget}")
        
        cnt = 0
        curr_acc = {1:[], 0:[]}
        for target_node in tqdm.tqdm(node_list):
            print(f'Target node: {target_node}')
            logger.info(f"Target node: {target_node}")
            proposed_model = ProposedAttack(surrogate_model, dataset, defense_model, important_edge_list)
            modified_adj, accuracy = proposed_model.attack(target_node=target_node, n_perturbations=budget)
            print("accuracy = ", accuracy)
            logger.info(f"accuracy: {accuracy}")
            if accuracy == 0:
                curr_acc[0].append(target_node)
                cnt += 1
            else:
                curr_acc[1].append(target_node)
            
        acc_node[budget] = curr_acc
        acc_list.append([budget, cnt / len(node_list), node_list])
        
        print(f"Total Target: {len(node_list)}")
        print('Miss-classification rate Modified : %s' % (cnt / len(node_list)))
        logger.info('Miss-classification rate Modified : %s' % (cnt / len(node_list)))
    
    df = pd.DataFrame(acc_list, columns =['budget_number', 'miss-classification_modified', 'node_list']) 
    df.to_csv(f'proposed_model_{dataset}_{defense_model}_{times}.csv') ## please change the number accordingly 

    with open(f"./add_remove_stat_proposed_model_{dataset}_{defense_model}_{times}.json", 'w', encoding='utf-8') as json_file:
        json.dump(add_remove_stat, json_file, indent=4, ensure_ascii=False)


    # Create two line charts for col1 and col2
    plt.figure(figsize=(8, 6))

    # Line chart for col1
    plt.plot(df['budget_number'], df['miss-classification_modified'], label='Modified Adj', marker='o', markersize=5, linestyle='-')

    # plt.ylim(bottom=0.10, top=0.50)

    # Add labels and a legend
    plt.xlabel('target_number') 
    plt.ylabel('miss-classification')
    plt.title(f'proposed_model_{dataset}_{defense_model}_{times}')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig(f'proposed_model_{dataset}_{defense_model}_{times}.png')
    # plt.show()


if __name__ == "__main__": 

    surrogate_model = 'gcn'
    dataset = 'cora'
    defense_model = 'gcn'

    data = get_dataset_from_deeprobust(dataset=dataset)
    print("Dataset loaded...")
    budget_range = 7

    # node_list = [929, 1049, 2082, 1541, 2406, 1669, 1347, 1163, 2077, 1504, 238, 1478, 2209, 1474]
    node_list = get_target_node_list(data)
    print("Targegt nodes are being selected...")

    start_attack_proposed_model(surrogate_model, dataset, defense_model, budget_range, node_list)


