import time

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

import scipy.sparse as sp

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

def convert_time(seconds):
    minutes = seconds // 60
    seconds %= 60
    return minutes, seconds

# add_remove_stat = collections.defaultdict(lambda : 0)
def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """

    # TODO: maybe using coo format would be better?
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

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
        self.surrogate_model, self.surrogate_model_output = self.get_surrogate_model_output()
        # print(len(self.surrogate_model_output))
        # print(len(self.labels), type(self.labels))
        self.important_edge_list = important_edge_list
        self.W = self.get_linearized_weight()
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
        
    def get_tained_surrogate_model_accuracy(self, modified_adj, target_node):
        features = self.data2.features
        n = self.pyg_data[0].num_nodes
        modified_adj = modified_adj.t() 
        modified_adj = sp.csr_matrix((np.ones(modified_adj.shape[1]),
                                  (modified_adj[0], modified_adj[1])), shape=(n, n))
        
        self.surrogate_model.eval()
        output_modified = self.surrogate_model.predict(features=features, adj=modified_adj)
        label = output_modified.argmax(1)[target_node]
        acc_test = self.labels[target_node] == label
        return acc_test


    def get_surrogate_model_output(self):
        adj2, features2, labels2 = self.data2.adj, self.data2.features, self.data2.labels
        idx_train2, idx_val2, idx_test2 = self.data2.idx_train, self.data2.idx_val, self.data2.idx_test
        output = None
        model = None
        if self.surrogate == 'gcn':
            gcn = GCN(nfeat=features2.shape[1], nhid=16, nclass=labels2.max().item() + 1, dropout=0.5, device=device)
            gcn = gcn.to(device)
            gcn.fit(features2, adj2, labels2, idx_train2, idx_val2, patience=30, train_iters=100)
            gcn.eval()
            output = gcn.predict()
            model = gcn
            # print(output)
            # exit()
        elif self.surrogate == 'gin':
            gin = GIN(nfeat=self.features.shape[1], nhid=8, heads=8, nclass=self.labels.max().item() + 1, dropout=0.5, device=device)
            gin = gin.to(device)
            gin.fit(self.pyg_data, verbose=False) 
            gin.eval()
            output = gin.predict()
            model = gin
        elif self.surrogate == 'gat':
            gat = GAT(nfeat=self.features.shape[1], nhid=8, heads=8, nclass=self.labels.max().item() + 1, dropout=0.5, device=device)
            gat = gat.to(device)
            gat.fit(self.pyg_data, verbose=False)  
            gat.eval()
            output = gat.predict()
            model = gat
        elif self.surrogate == 'graphsage':
            graphsage = GraphSAGE(nfeat=self.features.shape[1], nhid=8, heads=8, nclass=self.labels.max().item() + 1, dropout=0.5, device=device)
            graphsage = graphsage.to(device)
            graphsage.fit(self.pyg_data, verbose=False) 
            graphsage.eval()
            output = graphsage.predict()
            model = graphsage
                    
        assert output is not None, "Surrogate model should be any one of the listed model: [gcn, gin, gat, graphsage]"
        return model, output
    
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
    
    def get_linearized_weight(self):
        surrogate = self.surrogate_model
        W = surrogate.gc1.weight @ surrogate.gc2.weight
        return W.detach().cpu().numpy()
    
    def get_highest_loss_edge_after_add(self, updated_edges, target_node):
        updated_edges_tmp = updated_edges.copy()
        edges_list_before_perturbation = updated_edges_tmp.copy()

        edge_to_add = self.get_nodes_to_connect_by_distance(updated_edges_tmp, target_node)
        updated_edges_tmp.append([target_node, edge_to_add])
        # updated_edges_tmp.append([edge_to_add, target_node])

        nclass= self.data2.labels.max().item()+1
        n = self.pyg_data[0].num_nodes
        features2 = self.data2.features
        labels = self.data2.labels

        def get_surrogate_losses(updated_edges_tmp):
            modified_adj_tmp = torch.tensor(updated_edges_tmp).T
            modified_adj_2 = sp.csr_matrix((np.ones(modified_adj_tmp.shape[1]),
                                        (modified_adj_tmp[0], modified_adj_tmp[1])), shape=(n, n))
            
            adj_norm = normalize_adj(modified_adj_2)

            modified_features = features2.copy()
            logits = (adj_norm @ adj_norm @ modified_features @ self.W )[target_node]
            label_u = labels[target_node]
            
            label_target_onehot = np.eye(int(nclass))[labels[target_node]]
            best_wrong_class = (logits - 1000*label_target_onehot).argmax()
            surrogate_losses = [logits[labels[target_node]] - logits[best_wrong_class]]

            return surrogate_losses[0]

        def get_score(edges_list_before_perturbation, updated_edges_tmp_2):
            return get_surrogate_losses(updated_edges_tmp_2) - get_surrogate_losses(edges_list_before_perturbation)
        

        score = get_score(edges_list_before_perturbation, updated_edges_tmp)

        # modified_adj_tmp = torch.tensor(updated_edges_tmp).T
        # modified_adj_2 = sp.csr_matrix((np.ones(modified_adj_tmp.shape[1]),
        #                             (modified_adj_tmp[0], modified_adj_tmp[1])), shape=(n, n))
        
        # adj_norm = normalize_adj(modified_adj_2)

        # modified_features = features2.copy()
        # logits = (adj_norm @ adj_norm @ modified_features @ self.W )[target_node]
        # label_u = labels[target_node]
        
        # label_target_onehot = np.eye(int(nclass))[labels[target_node]]
        # best_wrong_class = (logits - 1000*label_target_onehot).argmax()
        # surrogate_losses = [logits[labels[target_node]] - logits[best_wrong_class]]
        
        # return updated_edges_tmp, surrogate_losses
        return updated_edges_tmp, score
    
    def get_highest_loss_edge_after_remove(self, final_prune_list, updated_edges, target_node):
        updated_edges_tmp_2 = updated_edges.copy()
        # G = nx.Graph()
        # G.add_edges_from(updated_edges_tmp_2)
        # degree = dict(nx.degree(G))
        # if degree[target_node] < 2:
        #     return []
        
        final_prune_list_with_loss = [list(val) for val in final_prune_list]
        for i in range(len(final_prune_list_with_loss)):
            final_prune_list_with_loss[i].append(float('-inf'))

        nclass= self.data2.labels.max().item()+1
        n = self.pyg_data[0].num_nodes
        features2 = self.data2.features
        labels = self.data2.labels

        def get_surrogate_losses(updated_edges_tmp_2):
            modified_adj_tmp = torch.tensor(updated_edges_tmp_2).T
            modified_adj_2 = sp.csr_matrix((np.ones(modified_adj_tmp.shape[1]),
                                        (modified_adj_tmp[0], modified_adj_tmp[1])), shape=(n, n))
            
            adj_norm = normalize_adj(modified_adj_2)

            modified_features = features2.copy()
            logits = (adj_norm @ adj_norm @ modified_features @ self.W )[target_node]
            label_u = labels[target_node]
            
            label_target_onehot = np.eye(int(nclass))[labels[target_node]]
            best_wrong_class = (logits - 1000*label_target_onehot).argmax()
            surrogate_losses = [logits[labels[target_node]] - logits[best_wrong_class]]

            return surrogate_losses[0]

        def get_score(edges_list_before_perturbation, updated_edges_tmp_2):
            return get_surrogate_losses(updated_edges_tmp_2) - get_surrogate_losses(edges_list_before_perturbation)

        for idx, edge in enumerate(list(final_prune_list)):
            edges_list_before_perturbation = updated_edges_tmp_2.copy()
            edge_to_remove_1 = edge
            edge_to_remove_2 = [edge_to_remove_1[1], edge_to_remove_1[0]]
            try:
                updated_edges_tmp_2.remove(edge_to_remove_1)
            except:
                pass

            try:
                updated_edges_tmp_2.remove(edge_to_remove_2)
            except:
                pass

            # G = nx.Graph()
            # G.add_edges_from(updated_edges_tmp_2)
            # degree = dict(nx.degree(G))
            # if degree[target_node] < 1:
            #     return []

            # Check if the graph is connected
            # is_connected = nx.is_connected(G)
            
            # assert is_connected, "Graph is disconnected."
            # if not is_connected:
            #     updated_edges_tmp_2 = updated_edges.copy()
            #     continue


            # After perturbation 

            # modified_adj_tmp = torch.tensor(updated_edges_tmp_2).T
            # modified_adj_2 = sp.csr_matrix((np.ones(modified_adj_tmp.shape[1]),
            #                             (modified_adj_tmp[0], modified_adj_tmp[1])), shape=(n, n))
            
            # adj_norm = normalize_adj(modified_adj_2)

            # modified_features = features2.copy()
            # logits = (adj_norm @ adj_norm @ modified_features @ self.W )[target_node]
            # label_u = labels[target_node]
            
            # label_target_onehot = np.eye(int(nclass))[labels[target_node]]
            # best_wrong_class = (logits - 1000*label_target_onehot).argmax()
            # surrogate_losses = [logits[labels[target_node]] - logits[best_wrong_class]]

            # final_prune_list_with_loss[idx][2] = surrogate_losses[0]

            score = get_score(edges_list_before_perturbation, updated_edges_tmp_2)

            final_prune_list_with_loss[idx][2] = score
            updated_edges_tmp_2 = updated_edges.copy()
        
        sorted_list = sorted(final_prune_list_with_loss, key=lambda x: x[2], reverse=True)
        return sorted_list
    
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

        if len(final_prune_list) < n_perturbations:
            for (u, v) in updated_edges.copy():
                if (u == target_node or v == target_node) and ((u, v) not in final_prune_list) and ((v, u) not in final_prune_list):
                    if degree[u] <= 1 or degree[v] <= 1:
                        continue
                    final_prune_list.append((u, v))
                    if len(final_prune_list) > n_perturbations:
                        break
        
        final_prune_list_tmp = final_prune_list.copy()
        
        cnt1 = 0
        cnt2 = 0
        isLimitOver = 0
        degree = dict(nx.degree(G))
        while n_perturbations:
            final_prune_list_empty = 0
            if (isBoth or isRemove) and isLimitOver==0:
                edges_to_remove = self.get_highest_loss_edge_after_remove(final_prune_list, updated_edges, target_node)
                try:
                    surrogate_losses_remove = edges_to_remove[0][2]
                except:
                    final_prune_list_empty = 1
            
            if isBoth or isAdd:
                # updated_edges_tmp = updated_edges.copy()
                updated_edges_tmp, surrogate_losses_add = self.get_highest_loss_edge_after_add(updated_edges, target_node)
            
            should_remove = 0

            # print("surrogate_losses: ", surrogate_losses_remove, surrogate_losses_add)
            # exit()

            if isBoth and final_prune_list_empty==0 and isLimitOver==0:
                if surrogate_losses_remove > surrogate_losses_add:
                    should_remove = 1
            elif isRemove:
                should_remove = 1

            if (isBoth or isRemove) and isLimitOver==0 and should_remove==1:
                updated_edges_tmp_2 = updated_edges.copy()
                # final_prune_list = self.get_highest_loss_edge_after_remove(final_prune_list, updated_edges, target_node)
                try:
                    edge_to_remove_1 = [edges_to_remove[0][0], edges_to_remove[0][1]]
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
                    try:
                        degree_target_node = dict(nx.degree(G))[target_node]
                        if not is_connected or degree_target_node < 1:
                            isLimitOver = 1
                            updated_edges_tmp_2 = updated_edges.copy()
                    except:
                        isLimitOver = 1
                        updated_edges_tmp_2 = updated_edges.copy()
   
            if should_remove:
                cnt2+=1
                updated_edges = updated_edges_tmp_2.copy()
            else:
                cnt1+=1
                updated_edges = updated_edges_tmp.copy()

            n_perturbations -= 1
        
        print(f"added: {cnt1}, removed: {cnt2}")
        logger.info(f"added: {cnt1}, removed: {cnt2}")

        modified_adj = torch.tensor(updated_edges, dtype=torch.long)
        
        return modified_adj

def start_attack_proposed_model(surrogate_model, dataset, defense_model, budget_range, node_list, times=1):
    acc_list = []
    acc_node = {}

    global add_remove_stat
    add_remove_stat = collections.defaultdict(lambda : 0)
    start_budget = 1 ## this will be the function parameter (code cleaning is not done yet.)
    proposed_model = ProposedAttack(surrogate_model, dataset, defense_model)
    important_edge_list = proposed_model.get_important_edge_list()

    predict_classes = {
            'gcn': test_GCN,
            'gin': test_GIN,
            'gat': test_GAT,
            'graphsage': test_GraphSAGE
        }
    predict = predict_classes[defense_model]

    already_misclassified = set()

    for budget in tqdm.tqdm(range(start_budget, budget_range+1)):
        print(f"For budget number: {budget}")
        logger.info(f"For budget number: {budget}")
        
        cnt = 0
        curr_acc = {1:[], 0:[]}

        start_time = time.time()
        
        for target_node in tqdm.tqdm(node_list):
            print(f'Target node: {target_node}')
            logger.info(f"Target node: {target_node}")
            if target_node in already_misclassified:
                cnt += 1
                print("already misclassified...")
                logger.info(f"already misclassified...")
                continue

            proposed_model = ProposedAttack(surrogate_model, dataset, defense_model, important_edge_list)
            modified_adj = proposed_model.attack(target_node=target_node, n_perturbations=budget)
            # print("haha!..")
            accuracy = predict(modified_adj, target_node, dataset)
            print("accuracy = ", accuracy)
            logger.info(f"accuracy: {accuracy}")
            if accuracy == 0:
                curr_acc[0].append(target_node)
                already_misclassified.add(target_node)
                cnt += 1
            else:
                curr_acc[1].append(target_node)
        
        end_time = time.time()
        running_time_seconds = end_time - start_time
        running_time_minutes, running_time_seconds = convert_time(running_time_seconds)
        print(f"running_time: {int(running_time_minutes)} minutes, {running_time_seconds} seconds")


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

    '''
    Issues:
    1. model can assume the class is false but after the modification, for new graph representation it can be true, although all the budgets are not used yet. It can be possible if we use all the budget then it may really return false. since the model is trained on original graph, it is very likely to happend. 

    '''

    surrogate_model = 'gcn'
    dataset = 'cora'
    defense_model = 'gcn'

    data = get_dataset_from_deeprobust(dataset=dataset)
    print("Dataset loaded...")
    budget_range = 7

    node_list = [929, 1342, 1554, 1255, 2406, 1163, 1340, 2077, 1347, 1820, 429, 1267, 1068, 1223, 1330, 1959, 2469, 1343, 1070, 2355, 1829, 482, 2035, 615, 1441, 23, 582, 875, 1309, 2256, 2396, 2228, 336, 463, 2142, 603, 2423, 2109, 846, 117]
    # node_list = get_target_node_list(data)
    print("Targegt nodes are being selected...")

    start_attack_proposed_model(surrogate_model, dataset, defense_model, budget_range, node_list)


