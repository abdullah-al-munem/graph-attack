import torch

from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.defense import GCN
from deeprobust.graph.defense import GAT
from GIN import GIN
from GSAGE import GraphSAGE

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

def test_GCN(adj, target_node, dataset, is_torch_geometric=True):
    ''' test on GCN '''
    data3 = Dpr2Pyg(Dataset(root=r'./', name=dataset))
    if is_torch_geometric:
        data3[0].edge_index = adj.t()

    data2 = Pyg2Dpr(data3)
        
    adj2, features2, labels2 = data2.adj, data2.features, data2.labels
    idx_train2, idx_val2, idx_test2 = data2.idx_train, data2.idx_val, data2.idx_test

    gcn = GCN(nfeat=features2.shape[1], nhid=16, nclass=labels2.max().item() + 1, dropout=0.5, device=device)
    gcn = gcn.to(device)
    if is_torch_geometric:
        gcn.fit(features2, adj2, labels2, idx_train2, idx_val2, patience=30, train_iters=100)
    else:
        gcn.fit(features2, adj, labels2, idx_train2, idx_val2, patience=30, train_iters=100)

    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    acc_test = (output.argmax(1)[target_node] == labels2[target_node])

    return acc_test.item()

def test_GAT(adj, target_node, dataset, is_torch_geometric=True):
    ''' test on GAT '''
    pyg_data = Dpr2Pyg(Dataset(root=r'./', name=dataset))
    # pyg_data = Dpr2Pyg(data)
    if is_torch_geometric:
        perturbed_adj = adj.t()  
        pyg_data[0].edge_index = perturbed_adj
    else:
        perturbed_adj = adj.tocsr()
        pyg_data.update_edge_index(perturbed_adj)  

    features = pyg_data[0].x
    labels = pyg_data[0].y

    gat = GAT(nfeat=features.shape[1], nhid=8, heads=8, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gat = gat.to(device)

    # pyg_data.update_edge_index(perturbed_adj)  ############ inplace operation
    gat.fit(pyg_data, verbose=False)  ############ train with earlystopping
    gat.eval()
    output = gat.predict()
    probs = torch.exp(output[[target_node]])[0]
    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])

    return acc_test.item()

def test_GIN(adj, target_node, dataset, is_torch_geometric=True):
    ''' test on GIN '''
    pyg_data = Dpr2Pyg(Dataset(root=r'./', name=dataset))
    if is_torch_geometric:
        perturbed_adj = adj.t()  
        pyg_data[0].edge_index = perturbed_adj
    else:
        perturbed_adj = adj.tocsr()
        pyg_data.update_edge_index(perturbed_adj)  ############ inplace operation

    features = pyg_data[0].x
    labels = pyg_data[0].y

    gin = GIN(nfeat=features.shape[1], nhid=8, heads=8, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gin = gin.to(device)

    gin.fit(pyg_data, verbose=False)  ############ train with earlystopping
    gin.eval()
    output = gin.predict()
    probs = torch.exp(output[[target_node]])[0]
    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])

    return acc_test.item()

def test_GraphSAGE(adj, target_node, dataset, is_torch_geometric=True):
    ''' test on GraphSAGE '''
    pyg_data = Dpr2Pyg(Dataset(root=r'./', name=dataset))
    if is_torch_geometric:
        perturbed_adj = adj.t()  
        pyg_data[0].edge_index = perturbed_adj
    else:
        perturbed_adj = adj.tocsr()
        pyg_data.update_edge_index(perturbed_adj)

    features = pyg_data[0].x
    labels = pyg_data[0].y
    
    graphsage = GraphSAGE(nfeat=features.shape[1], nhid=8, heads=8, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    graphsage = graphsage.to(device)
    
    # pyg_data.update_edge_index(perturbed_adj)  ############ inplace operation
    graphsage.fit(pyg_data, verbose=False)  ############ train with earlystopping

    #gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    #gcn = gcn.to(device)
    #gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    graphsage.eval()
    output = graphsage.predict()
    probs = torch.exp(output[[target_node]])[0]
    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()


