from deeprobust.graph.data import Dataset
from utils import precompute_all_pairs_shortest_distances, save_distances_to_json, load_distances_from_json, get_nodes_to_connect_by_distance_top_n_from_precompute, get_nodes_to_connect_by_distance_top_n

# Load your dataset
dataset = "polblogs"
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
result = get_nodes_to_connect_by_distance_top_n_from_precompute(precomputed_distances, target_node=767, N=5)

print(result)

result = get_nodes_to_connect_by_distance_top_n(data, target_node=767, N=5)
print(result)
