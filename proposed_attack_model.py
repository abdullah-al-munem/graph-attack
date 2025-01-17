import time
import json

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

from predict import test_GCN, test_GAT, test_GIN, test_GraphSAGE, test_RGCN, test_acc_GCN, test_acc_GIN, test_acc_GSAGE, test_acc_RGCN, test_acc_MDGCN, test_acc_JacGCN, test_acc_SVDGCN

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

# import scipy.sparse as sp
def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        given tensor

    Returns
    -------
    bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False
    
def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    
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

def compute_new_a_hat_uv(potential_edges, target_node, modified_adj, adj_norm, nnodes):
        """
        Compute the updated A_hat_square_uv entries that would result from inserting/deleting the input edges,
        for every edge.

        Parameters
        ----------
        potential_edges: np.array, shape [P,2], dtype int
            The edges to check.

        Returns
        -------
        sp.sparse_matrix: updated A_hat_square_u entries, a sparse PxN matrix, where P is len(possible_edges).
        """

        edges = np.array(modified_adj.nonzero()).T
        edges_set = {tuple(x) for x in edges}
        A_hat_sq = adj_norm @ adj_norm
        values_before = A_hat_sq[target_node].toarray()[0]
        node_ixs = np.unique(edges[:, 0], return_index=True)[1]
        twohop_ixs = np.array(A_hat_sq.nonzero()).T
        degrees = modified_adj.sum(0).A1 + 1

        ixs, vals = compute_new_a_hat_uv_2(edges, node_ixs, edges_set, twohop_ixs, values_before, degrees,
                                         potential_edges.astype(np.int32), target_node)
        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, 0], ixs_arr[:, 1])), shape=[len(potential_edges), nnodes])

        return a_hat_uv

# @jit(nopython=True)
def connected_after(u, v, connected_before, delta):
    if u == v:
        if delta == -1:
            return False
        else:
            return True
    else:
        return connected_before


# @jit(nopython=True)
def compute_new_a_hat_uv_2(edge_ixs, node_nb_ixs, edges_set, twohop_ixs, values_before, degs, potential_edges, u):
    """
    Compute the new values [A_hat_square]_u for every potential edge, where u is the target node. C.f. Theorem 5.1
    equation 17.

    """
    N = degs.shape[0]

    twohop_u = twohop_ixs[twohop_ixs[:, 0] == u, 1]
    nbs_u = edge_ixs[edge_ixs[:, 0] == u, 1]
    nbs_u_set = set(nbs_u)

    return_ixs = []
    return_values = []

    for ix in range(len(potential_edges)):
        edge = potential_edges[ix]
        edge_set = set(edge)
        degs_new = degs.copy()
        delta = -2 * ((edge[0], edge[1]) in edges_set) + 1
        degs_new[edge] += delta

        nbs_edge0 = edge_ixs[edge_ixs[:, 0] == edge[0], 1]
        nbs_edge1 = edge_ixs[edge_ixs[:, 0] == edge[1], 1]

        affected_nodes = set(np.concatenate((twohop_u, nbs_edge0, nbs_edge1)))
        affected_nodes = affected_nodes.union(edge_set)
        a_um = edge[0] in nbs_u_set
        a_un = edge[1] in nbs_u_set

        a_un_after = connected_after(u, edge[0], a_un, delta)
        a_um_after = connected_after(u, edge[1], a_um, delta)

        for v in affected_nodes:
            a_uv_before = v in nbs_u_set
            a_uv_before_sl = a_uv_before or v == u

            if v in edge_set and u in edge_set and u != v:
                if delta == -1:
                    a_uv_after = False
                else:
                    a_uv_after = True
            else:
                a_uv_after = a_uv_before
            a_uv_after_sl = a_uv_after or v == u

            from_ix = node_nb_ixs[v]
            to_ix = node_nb_ixs[v + 1] if v < N - 1 else len(edge_ixs)
            node_nbs = edge_ixs[from_ix:to_ix, 1]
            node_nbs_set = set(node_nbs)
            a_vm_before = edge[0] in node_nbs_set

            a_vn_before = edge[1] in node_nbs_set
            a_vn_after = connected_after(v, edge[0], a_vn_before, delta)
            a_vm_after = connected_after(v, edge[1], a_vm_before, delta)

            mult_term = 1 / np.sqrt(degs_new[u] * degs_new[v])

            sum_term1 = np.sqrt(degs[u] * degs[v]) * values_before[v] - a_uv_before_sl / degs[u] - a_uv_before / \
                        degs[v]
            sum_term2 = a_uv_after / degs_new[v] + a_uv_after_sl / degs_new[u]
            sum_term3 = -((a_um and a_vm_before) / degs[edge[0]]) + (a_um_after and a_vm_after) / degs_new[edge[0]]
            sum_term4 = -((a_un and a_vn_before) / degs[edge[1]]) + (a_un_after and a_vn_after) / degs_new[edge[1]]
            new_val = mult_term * (sum_term1 + sum_term2 + sum_term3 + sum_term4)

            return_ixs.append((ix, v))
            return_values.append(new_val)

    return return_ixs, return_values

def struct_score(a_hat_uv, XW, label_u):
    """
    Compute structure scores, cf. Eq. 15 in the paper

    Parameters
    ----------
    a_hat_uv: sp.sparse_matrix, shape [P,2]
        Entries of matrix A_hat^2_u for each potential edge (see paper for explanation)

    XW: sp.sparse_matrix, shape [N, K], dtype float
        The class logits for each node.

    Returns
    -------
    np.array [P,]
        The struct score for every row in a_hat_uv
    """

    logits = a_hat_uv.dot(XW)
    label_onehot = np.eye(XW.shape[1])[label_u]
    best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
    logits_for_correct_class = logits[:,label_u]
    struct_scores = logits_for_correct_class - best_wrong_class_logits

    return struct_scores

def get_scores_and_egdes(filtered_edges, modified_features, target_node, W, label_u, modified_adj, adj_norm, nnodes):

    # potential_edges = potential_edges.astype("int32")
    # singleton_filter = filter_singletons(potential_edges, modified_adj)
    # filtered_edges = potential_edges[singleton_filter]

    # Compute new entries in A_hat_square_uv
    a_hat_uv_new = compute_new_a_hat_uv(filtered_edges, target_node, modified_adj, adj_norm, nnodes)
    # Compute the struct scores for each potential edge
    struct_scores = struct_score(a_hat_uv_new, modified_features @ W, label_u)
    best_edge_ix = struct_scores.argmin()
    best_edge_score = struct_scores.min()
    best_edge = filtered_edges[best_edge_ix]

    return [best_edge_score, best_edge]

def filter_singletons(edges, adj):
    """
    Filter edges that, if removed, would turn one or more nodes into singleton nodes.
    """

    if type(adj) is torch.Tensor:
        adj = to_scipy(adj).tolil()
    else:
        adj = adj.tolil()

    # print(len(edges))

    degs = np.squeeze(np.array(np.sum(adj,0)))
    existing_edges = np.squeeze(np.array(adj.tocsr()[tuple(edges.T)]))
    if existing_edges.size > 0:
        edge_degrees = degs[np.array(edges)] + 2*(1-existing_edges[:,None]) - 1
    else:
        edge_degrees = degs[np.array(edges)] + 1

    zeros = edge_degrees == 0
    zeros_sum = zeros.sum(1)
    return zeros_sum == 0

predict_classes = {
    'gcn': test_acc_GCN,
    'gin': test_acc_GIN,
    'graphsage': test_acc_GSAGE,
    'rgcn': test_acc_RGCN,
    'mdgcn': test_acc_MDGCN,
    'jacgcn': test_acc_JacGCN,
    'svdgcn': test_acc_SVDGCN
}

class ProposedAttack:
    def __init__(self, surrogate_model, dataset, defense_model, important_edge_list=None, attack_structure=True, attack_features=False, device=device):
        # data = Dataset(root=r'./', name=dataset) # load clean graph
        data = get_dataset_from_deeprobust(dataset=dataset)
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
        # self.predict = self.get_predict_function()
        self.surrogate_model, self.surrogate_model_output = self.get_surrogate_model_output()
        # print(len(self.surrogate_model_output))
        # print(len(self.labels), type(self.labels))
        self.important_edge_list = important_edge_list
        self.W = self.get_linearized_weight()
        # if important_edge_list != None:
        #     exit()
    
    def get_predict_function(self):
        
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

    def get_nodes_to_connect_by_distance_top_n(self, adj, target_node, N):
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

        return top_nodes_from_specific_node[:N]

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

        return updated_edges_tmp, score
    
    def get_highest_loss_edge_after_remove(self, final_prune_list, updated_edges, target_node):
        updated_edges_tmp_2 = updated_edges.copy()
        
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
                    prune_list.append([u, v])
                if (v, u) not in prune_list:
                    prune_list.append([v, u])
        
        final_prune_list = prune_list.copy()

        updated_edges = self.adj.tolist()  
         
        adj2, features2, labels2 = self.data2.adj, self.data2.features, self.data2.labels
        modified_adj = adj2.copy().tolil()
        # print(features2)
        # print(type(features2))
        modified_features = features2.copy().tolil()
        adj_norm = normalize_adj(modified_adj)
        nnodes = adj2.shape[0]

        if len(final_prune_list) == 0:
            final_prune_list = np.column_stack((np.tile(target_node, nnodes-1), np.setdiff1d(np.arange(nnodes), target_node)))

        final_prune_list = np.array(final_prune_list).astype("int32")
        
        singleton_filter = filter_singletons(final_prune_list, modified_adj)
        final_prune_list = final_prune_list[singleton_filter]

        if len(final_prune_list) < n_perturbations:
            for (u, v) in updated_edges.copy():
                if (u == target_node or v == target_node) and ((u, v) not in final_prune_list) and ((v, u) not in final_prune_list):
                    if degree[u] <= 1 or degree[v] <= 1:
                        continue
                    final_prune_list.append([u, v])
                    if len(final_prune_list) > n_perturbations:
                        break
        
        # final_prune_list_tmp = np.array(final_prune_list.copy()).astype("int32")

        n_add = max(len(final_prune_list), n_perturbations) 

        potential_edges_add = self.get_nodes_to_connect_by_distance_top_n(updated_edges, target_node, n_add)
        potential_edges_add = [[target_node, node[0]] for node in potential_edges_add]
        # print(potential_edges_add)

        potential_edges_remove = [node for edge in final_prune_list for node in edge if node != target_node]
        potential_edges_remove = [[target_node, node] for node in potential_edges_remove]
        # print(potential_edges_remove)
        potential_edges_final = potential_edges_remove + potential_edges_add
        potential_edges_final = np.array(potential_edges_final).astype("int32")

        if len(potential_edges_final) <= n_perturbations:
            potential_edges = np.column_stack((np.tile(target_node, nnodes-1), np.setdiff1d(np.arange(nnodes), target_node)))
            potential_edges_final = np.concatenate((potential_edges_final, potential_edges))

        # print(potential_edges_final)
        singleton_filter = filter_singletons(potential_edges_final, modified_adj)
        # print(singleton_filter)
        potential_edges_final = potential_edges_final[singleton_filter]
        # print(potential_edges_final)
        
        label_u = labels2[target_node]
        cnt_add = 0
        cnt_remove = 0

        print(f"Total potential edges: {len(potential_edges_final)}")

        # if len(potential_edges_final) < n_perturbations:

        while n_perturbations:
            best_edge_score, best_edge = get_scores_and_egdes(potential_edges_final, modified_features, target_node, self.W, label_u, modified_adj, adj_norm, nnodes)

            modified_adj[tuple(best_edge)] = modified_adj[tuple(best_edge[::-1])] = 1 - modified_adj[tuple(best_edge)]
            adj_norm = normalize_adj(modified_adj)
            nnodes = modified_adj.shape[0]

            if adj2[best_edge[0], best_edge[1]]:
                msg = "Removed"  
                cnt_remove += 1
            else: 
                msg = "Added"
                cnt_add += 1

            print(f"{msg} best_edge_score: {best_edge_score}, best_edge: {best_edge}")
            # potential_edges_final.remove(best_edge)
            potential_edges_final = potential_edges_final[~np.all(potential_edges_final == best_edge, axis=1)]
            potential_edges_final = potential_edges_final[~np.all(potential_edges_final == best_edge[::-1], axis=1)]
            n_perturbations -= 1

        print(f"added: {cnt_add}, removed: {cnt_remove}")
        logger.info(f"added: {cnt_add}, removed: {cnt_remove}")

        return modified_adj
    
    def predict(self, modified_adj, target_node):
        
        predict_inner = predict_classes[self.defense_model]

        return predict_inner(modified_adj, self.data2.features, self.data2, target_node)

def get_important_edge_list_for_precompute(surrogate_model, dataset, defense_model):
    proposed_model = ProposedAttack(surrogate_model, dataset, defense_model)
    important_edge_list = proposed_model.get_important_edge_list()
    return important_edge_list

def start_attack_proposed_model(surrogate_model, dataset, defense_model, budget_range, node_list, times=1):
    acc_list = []
    acc_node = {}

    global add_remove_stat
    add_remove_stat = collections.defaultdict(lambda : 0)
    start_budget = 1 ## this will be the function parameter (code cleaning is not done yet.)

    # Load the dictionary from the JSON file
    with open('./important_edge_list.json', 'r') as json_file:
        important_edge_list_dict = json.load(json_file)

    # Convert lists back to tuples
    important_edge_list = [tuple(item) for item in important_edge_list_dict[dataset]]

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
            accuracy = proposed_model.predict(modified_adj, target_node)
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
    # dataset = 'cora'
    defense_model = 'gcn'

    # data = get_dataset_from_deeprobust(dataset=dataset)
    # print("Dataset loaded...")
    # budget_range = 7

    # node_list = [929, 1342, 1554, 1255, 2406, 1163, 1340, 2077, 1347, 1820, 429, 1267, 1068, 1223, 1330, 1959, 2469, 1343, 1070, 2355, 1829, 482, 2035, 615, 1441, 23, 582, 875, 1309, 2256, 2396, 2228, 336, 463, 2142, 603, 2423, 2109, 846, 117]
    # node_list = get_target_node_list(data)
    # node_list = [1079,]
    # print("Targegt nodes are being selected...")

    # start_attack_proposed_model(surrogate_model, dataset, defense_model, budget_range, node_list)

    # dataset_list = ['cora', 'citeseer', 'polblogs']
    # important_edge_list_dict = {}
    # for dataset in dataset_list:
    #     important_edge_list = get_important_edge_list_for_precompute(surrogate_model, dataset, defense_model)
    #     important_edge_list_dict[dataset] = important_edge_list
    #     important_edge_list_dict[dataset] = [list(item) for item in important_edge_list_dict[dataset]]

    # # Save the dictionary to a JSON file
    # with open('./important_edge_list.json', 'w') as json_file:
    #     json.dump(important_edge_list_dict, json_file, indent=4)

    # dataset = 'ogbn-arxiv'
    # important_edge_list_dict = json.load(open('./important_edge_list.json', 'r', encoding='utf-8'))
    # important_edge_list = get_important_edge_list_for_precompute(surrogate_model, dataset, defense_model)
    # important_edge_list_dict[dataset] = important_edge_list
    # important_edge_list_dict[dataset] = [list(item) for item in important_edge_list_dict[dataset]]

    # with open('./important_edge_list.json', 'w') as json_file:
    #     json.dump(important_edge_list_dict, json_file, indent=4)



    





