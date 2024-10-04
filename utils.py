import os
import warnings
import torch
import numpy as np
import json

from deeprobust.graph.defense import GCN

from deeprobust.graph.data import Dataset
from predict import test_GCN, test_GAT, test_GIN, test_GraphSAGE, test_RGCN, test_acc_GCN, test_acc_GIN, test_acc_GSAGE, test_acc_RGCN, test_acc_MDGCN, test_acc_JacGCN, test_acc_SVDGCN

from ogb.nodeproppred import PygNodePropPredDataset
from deeprobust.graph.data import Pyg2Dpr, Dpr2Pyg

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

def get_dataset_from_deeprobust(dataset):
    if dataset == 'ogbn-arxiv':
        pyg_data = PygNodePropPredDataset(name=dataset)
        data = Pyg2Dpr(pyg_data)
    else:
        data = Dataset(root=r'./', name=dataset) 
    return data

def destructuring_dataset(data):
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    return adj, features, labels, idx_train, idx_val, idx_test

def get_predict_function(defense_model):
    predict_classes = {
        'gcn': test_acc_GCN,
        'gin': test_acc_GIN,
        'graphsage': test_acc_GSAGE,
        'rgcn': test_acc_RGCN,
        'mdgcn': test_acc_MDGCN,
        'jacgcn': test_acc_JacGCN,
        'svdgcn': test_acc_SVDGCN
    }
    return predict_classes[defense_model]

def classification_margin(output, true_label):
    """Calculate classification margin for outputs.
    `probs_true_label - probs_best_second_class`

    Parameters
    ----------
    output: torch.Tensor
        output vector (1 dimension)
    true_label: int
        true label for this node

    Returns
    -------
    list
        classification margin for this node
    """

    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()


def select_nodes(adj, features, labels, idx_train, idx_val, idx_test, target_gcn=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    if target_gcn is None:
        target_gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)
        target_gcn = target_gcn.to(device)
        target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    target_gcn.eval()
    output = target_gcn.predict()

    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other

def get_target_node_list(data):
    adj, features, labels, idx_train, idx_val, idx_test = destructuring_dataset(data)
    target_node_list = select_nodes(adj, features, labels, idx_train, idx_val, idx_test)
    return target_node_list

def get_miss_classification_original_dataset(defense_model, dataset, node_list, time=1):

    file_path = f"./miss_classification_rate_{dataset}_{defense_model}_{time}"
    if os.path.exists(file_path):
        return

    data = get_dataset_from_deeprobust(dataset)
    adj, features, labels, idx_train, idx_val, idx_test = destructuring_dataset(data)
    # print(data)
    predict = get_predict_function(defense_model)
    pyg_data = Dpr2Pyg(data)
    
    cnt = 0
    curr_acc = {1:[], 0:[]}
    for target_node in node_list:
        accuracy = predict(adj, target_node, pyg_data, is_torch_geometric=False)

        if accuracy == 0:
            curr_acc[0].append(target_node)
            cnt += 1
        else:
            curr_acc[1].append(target_node)

    print('Miss-classification rate : %s' % (cnt / len(node_list)))

    miss_classification_dict = {f'miss_classification_rate_{dataset}_{defense_model}_{time}': (cnt / len(node_list))}
    with open(f"./miss_classification_rate_{dataset}_{defense_model}_{time}.json", 'w', encoding='utf-8') as json_file:
        json.dump(miss_classification_dict, json_file, indent=4, ensure_ascii=False)

