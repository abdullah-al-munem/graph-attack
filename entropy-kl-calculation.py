import numpy as np
import torch
import torch.nn.functional as F
from deeprobust.graph.data import Dataset, Dpr2Pyg
import scipy.sparse as sp
from GIN import GIN
from deeprobust.graph.defense import GCN
import json
from proposed_attack_model import ProposedAttack
import tqdm

def calculate_metrics(model, features, adj, target_node, pyg_data=None, modified_adj=None, labels=None, idx_train=None, idx_val=None):
    """
    Calculate entropy and KL divergence for a target node's classification probabilities
    
    Parameters:
    -----------
    model: GNN model (trained)
    features: node features
    adj: original adjacency matrix
    target_node: index of the target node
    pyg_data: PyG data object
    modified_adj: modified adjacency matrix after perturbation (optional)
    
    Returns:
    --------
    dict: containing original entropy, new entropy (if modified_adj provided),
          entropy difference, and KL divergence (if modified_adj provided)
    """
    model.eval()
    
    # Get original probabilities - convert from log probabilities to probabilities
    log_probs = model.predict()[target_node]
    orig_probs = torch.exp(log_probs).detach().cpu().numpy()
    
    # Ensure probabilities sum to 1 (numerical stability)
    orig_probs = orig_probs / np.sum(orig_probs)
    
    
    # Calculate original entropy
    orig_entropy = -np.sum(orig_probs * np.log2(orig_probs + 1e-10))
    
    results = {
        'original_entropy': orig_entropy,
        'original_probs': orig_probs,
        'predicted_class': np.argmax(orig_probs)
    }
    
    # If modified adjacency matrix is provided, calculate new metrics
    if modified_adj is not None:
        perturbed_adj = modified_adj.tocsr()
        pyg_data.update_edge_index(perturbed_adj)
        # model.fit(pyg_data, verbose=False)
        model.fit(features, modified_adj, labels, idx_train, idx_val, patience=30)
        model.eval()
        
        # Get new probabilities
        log_probs_new = model.predict()[target_node]
        new_probs = torch.exp(log_probs_new).detach().cpu().numpy()
        new_probs = new_probs / np.sum(new_probs)  # Ensure probabilities sum to 1
        
        # Calculate new entropy
        new_entropy = -np.sum(new_probs * np.log2(new_probs + 1e-10))
        
        # Calculate KL divergence
        kl_div = np.sum(orig_probs * np.log2((orig_probs + 1e-10)/(new_probs + 1e-10)))
        
        results.update({
            'new_entropy': new_entropy,
            'entropy_difference': new_entropy - orig_entropy,
            'kl_divergence': kl_div,
            'new_probs': new_probs,
            'new_predicted_class': np.argmax(new_probs)
        })
        
    return results

from utils import get_target_node_list

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    dataset = "cora" # ["cora", "citeseer", "polblogs"]
    defense_model = "gcn" # ["gcn", "gin"]
    # Load Cora dataset
    cora = Dataset(root='./', name=dataset)
    adj, features, labels = cora.adj, cora.features, cora.labels
    idx_train, idx_val, idx_test = cora.idx_train, cora.idx_val, cora.idx_test

    node_list = get_target_node_list(cora)
    print(node_list)
    with open('./important_edge_list.json', 'r') as json_file:
        important_edge_list_dict = json.load(json_file)
    important_edge_list = [tuple(item) for item in important_edge_list_dict[dataset]]
    
    # target_node = 1554
    # best_edge = [1554, 1530]
    # Convert lists back to tuples
    miss_classification_stat = {}
    add_limit = 2
    remove_limit = 2
    for target_node in tqdm.tqdm(node_list):
        
        proposed_model = ProposedAttack("gcn", dataset, defense_model, important_edge_list)
        modified_adj, best_edge = proposed_model.attack(target_node=target_node, n_perturbations=1)
        # modified_adj = adj.copy().tolil()
        # modified_adj[tuple(best_edge)] = modified_adj[tuple(best_edge[::-1])] = 1 - modified_adj[tuple(best_edge)]
        
        if adj[best_edge[0], best_edge[1]]:
            msg = "Removed"  
            # print(msg)
        else: 
            msg = "Added"
            
            # print(msg)
        # Initialize and train GCN
        labels = cora.labels
        cora.features = sp.csr_matrix(cora.features.shape, dtype=int)
        pyg_data = Dpr2Pyg(cora)

        # model = GIN(nfeat=features.shape[1], nhid=8, heads=8, 
        #             nclass=labels.max().item() + 1, dropout=0.5, device="cuda")
        # model = model.to("cuda")
        
        # # Train model on original graph
        # model.fit(pyg_data, verbose=False)

        model = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device="cuda")
        model = model.to("cuda")
        model.fit(features, adj, labels, idx_train, idx_val, patience=30)
        
        # print("Actual class:", labels[target_node].item())
        
        # Calculate metrics for original graph
        # print("\nOriginal graph metrics:")
        metrics_org = calculate_metrics(model, features, adj, target_node)
        # print(f"Original entropy: {metrics_org['original_entropy']:.4f}")
        # print(f"Predicted class: {metrics_org['predicted_class']}")
        # print(f"Class probabilities: {metrics_org['original_probs']}")

        # Calculate metrics after perturbation
        # print("\nMetrics after edge perturbation:")
        metrics = calculate_metrics(model, features, adj, target_node, pyg_data, modified_adj, labels, idx_train, idx_val)
        # print(f"New entropy: {metrics['new_entropy']:.4f}")
        # print(f"Entropy difference: {metrics['entropy_difference']:.4f}")
        # print(f"KL divergence: {metrics['kl_divergence']:.4f}")
        # print(f"New predicted class: {metrics['new_predicted_class']}")
        # print(f"New class probabilities: {metrics['new_probs']}")
        print(labels[target_node].item(), metrics_org['predicted_class'], metrics['new_predicted_class'])
        
        # if msg == "Added":
        # if int(metrics_org['predicted_class']) == int(labels[target_node].item()) and int(metrics['new_predicted_class']) != int(labels[target_node].item()):
        # if int(metrics['new_predicted_class']) == int(labels[target_node].item()):
        # print("target_node: ", target_node)
        # print("best_edge: ", best_edge)
        # # print(msg)
        # print("Actual class:", labels[target_node].item())
        # print("\nOriginal graph metrics:")
        # print(f"Original entropy: {metrics_org['original_entropy']:.4f}")
        # print(f"Predicted class: {metrics_org['predicted_class']}")
        # print(f"Class probabilities: {metrics_org['original_probs']}")

        # Calculate metrics after perturbation
        # print("\nMetrics after edge perturbation:")
        # print(f"New entropy: {metrics['new_entropy']:.4f}")
        # print(f"Entropy difference: {metrics['entropy_difference']:.4f}")
        # print(f"KL divergence: {metrics['kl_divergence']:.4f}")
        # print(f"New predicted class: {metrics['new_predicted_class']}")
        # print(f"New class probabilities: {metrics['new_probs']}")

        miss_classification_stat[str(target_node)] = {
            'target_node': target_node,
            'action': msg,
            'candidate_node': list(best_edge),
            "Actual class": labels[target_node].item(),
            "Predicted class": metrics['new_predicted_class'],
            'Original entropy': metrics_org['original_entropy'],
            "New entropy": metrics['new_entropy'],
            "Entropy difference": metrics['entropy_difference'],
            "KL divergence": metrics['kl_divergence']
        }

        # add_limit -= 1
        # else:
            # if remove_limit and int(metrics_org['predicted_class']) == int(labels[target_node].item()) and int(metrics['new_predicted_class']) != int(labels[target_node].item()):
            #     print("target_node: ", target_node)
            #     print("Actual class:", labels[target_node].item())
            #     print(msg)
            #     print("\nOriginal graph metrics:")
            #     print(f"Original entropy: {metrics_org['original_entropy']:.4f}")
            #     print(f"Predicted class: {metrics_org['predicted_class']}")
            #     print(f"Class probabilities: {metrics_org['original_probs']}")

            #     # Calculate metrics after perturbation
            #     print("\nMetrics after edge perturbation:")
            #     print(f"New entropy: {metrics['new_entropy']:.4f}")
            #     print(f"Entropy difference: {metrics['entropy_difference']:.4f}")
            #     print(f"KL divergence: {metrics['kl_divergence']:.4f}")
            #     print(f"New predicted class: {metrics['new_predicted_class']}")
            #     print(f"New class probabilities: {metrics['new_probs']}")
            #     remove_limit -= 1

    print(miss_classification_stat)
    with open(f'./{dataset}_{defense_model}_miss_classification_stat_normalized.json', 'w') as json_file:
        json.dump(miss_classification_stat, json_file, indent=4, cls=NpEncoder)



if __name__ == "__main__":
    main()