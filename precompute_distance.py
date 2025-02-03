# First, precompute the distances and save them
from deeprobust.graph.data import Dataset
from utils import precompute_all_pairs_shortest_distances, save_distances_to_json, load_distances_from_json, get_nodes_to_connect_by_distance_top_n_from_precompute, get_nodes_to_connect_by_distance_top_n

# Load your dataset
dataset = "cora"
data = Dataset(root=r'./', name=dataset)

# Get the edge list from your adjacency matrix
edges = []
adj = data.adj
rows, cols = adj.nonzero()
for i, j in zip(rows, cols):
    edges.append((i, j))

# Precompute distances and save them
distances = precompute_all_pairs_shortest_distances(edges)
save_distances_to_json(distances, dataset)

# Later, when you need to use the distances:
precomputed_distances = load_distances_from_json(dataset)
result = get_nodes_to_connect_by_distance_top_n_from_precompute(precomputed_distances, target_node=1687, N=5)

print(result)

result = get_nodes_to_connect_by_distance_top_n(data, target_node=1687, N=5)
print(result)




# utils

import networkx as nx
import json
import numpy as np
from pathlib import Path

def precompute_all_pairs_shortest_distances(adj):
    """
    Precompute sorted shortest distances between all pairs of nodes in the graph.
    
    Parameters:
    -----------
    adj : list of tuples or numpy array
        The adjacency matrix or edge list of the graph
        
    Returns:
    --------
    dict
        Dictionary where keys are source nodes and values are lists of 
        (target_node, distance) sorted by distance in descending order
    """
    G = nx.Graph()
    G.add_edges_from(adj)
    
    # Dictionary to store all pairwise distances
    all_pairs_distances = {}
    
    # Compute shortest paths for all pairs of nodes and sort by distance
    for node in G.nodes():
        shortest_paths = nx.single_source_dijkstra_path_length(G, node)
        
        # Sort distances in descending order and convert to list of tuples
        sorted_distances = sorted(
            [(str(k), float(v)) for k, v in shortest_paths.items()], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        all_pairs_distances[str(node)] = sorted_distances
    
    return all_pairs_distances

def save_distances_to_json(distances, dataset_name, save_dir="./precomputed_distances"):
    """
    Save the precomputed sorted distances to a JSON file.
    
    Parameters:
    -----------
    distances : dict
        Dictionary of precomputed sorted distances
    dataset_name : str
        Name of the dataset (e.g., "cora")
    save_dir : str
        Directory to save the JSON file
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    file_path = Path(save_dir) / f"{dataset_name}_distances.json"
    
    with open(file_path, 'w') as f:
        json.dump(distances, f)
    
    print(f"Distances saved to {file_path}")

def load_distances_from_json(dataset_name, save_dir="./precomputed_distances"):
    """
    Load precomputed distances from JSON file.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., "cora")
    save_dir : str
        Directory where the JSON file is stored
        
    Returns:
    --------
    dict
        Dictionary of precomputed sorted distances
    """
    file_path = Path(save_dir) / f"{dataset_name}_distances.json"
    
    with open(file_path, 'r') as f:
        distances = json.load(f)
    
    return distances

def get_nodes_to_connect_by_distance_top_n_from_precompute(precomputed_distances, target_node, N):
    """
    Get the N nodes that are farthest from the target node using precomputed sorted distances.
    
    Parameters:
    -----------
    precomputed_distances : dict
        Dictionary of precomputed sorted distances
    target_node : int
        The source node
    N : int
        Number of nodes to return
        
    Returns:
    --------
    list
        List of (node, distance) tuples for the N farthest nodes
    """
    # Get presorted distances from target node to all other nodes
    distances = precomputed_distances[str(target_node)]
    
    # Convert string node IDs back to integers and return top N
    return [(int(node), dist) for node, dist in distances[:N]]

def get_nodes_to_connect_by_distance_top_n(data, target_node, N):
    pyg_data = Dpr2Pyg(data)
    adj = pyg_data[0].edge_index.t().tolist()  
    G = nx.Graph()
    G.add_edges_from(adj)

    # Dictionary to store shortest distances for each node
    shortest_distances = {}

    # Calculate shortest distances for each node in the node list
    shortest_paths = nx.single_source_dijkstra_path_length(G, target_node)
    shortest_distances[target_node] = shortest_paths
    distances_from_specific_node = shortest_distances[target_node]
    top_nodes_from_specific_node = sorted(distances_from_specific_node.items(), key=lambda x: x[1], reverse=True)

    return top_nodes_from_specific_node[:N]

