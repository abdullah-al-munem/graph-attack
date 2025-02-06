import torch

from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.defense import GCN
from deeprobust.graph.defense import GAT
from deeprobust.graph.defense import RGCN
from deeprobust.graph.defense import GCNJaccard
from deeprobust.graph.defense import GCNSVD
from deeprobust.graph.defense import MedianGCN

from GIN import GIN
from GSAGE import GraphSAGE
from scipy.sparse import csr_matrix

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

def test_GCN(adj, target_node, pyg_data, is_torch_geometric=True):
    ''' test on GCN '''
    # data3 = Dpr2Pyg(pyg_data)
    if is_torch_geometric:
        pyg_data[0].edge_index = adj.t()

    # print(data3)
    data2 = Pyg2Dpr(pyg_data)
        
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

def test_GAT(adj, target_node, pyg_data, is_torch_geometric=True):
    ''' test on GAT '''
    # pyg_data = Dpr2Pyg(pyg_data)
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

def test_GIN(adj, target_node, pyg_data, is_torch_geometric=True):
    ''' test on GIN '''
    # pyg_data = Dpr2Pyg(pyg_data)
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

def test_GraphSAGE(adj, target_node, pyg_data, is_torch_geometric=True):
    ''' test on GraphSAGE '''
    # pyg_data = Dpr2Pyg(pyg_data)
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


def test_RGCN(adj, target_node, pyg_data, is_torch_geometric=True):
    ''' test on RGCN '''
    # print(type(adj))
    if is_torch_geometric:
        perturbed_adj = adj.t()  
        pyg_data[0].edge_index = perturbed_adj
    else:
        perturbed_adj = adj.tocsr()
        pyg_data.update_edge_index(perturbed_adj)  
    
    # print(type(pyg_data.x))
    data2 = Pyg2Dpr(pyg_data)
    adj, features, labels = data2.adj, data2.features, data2.labels
    idx_train, idx_val, idx_test = data2.idx_train, data2.idx_val, data2.idx_test
    
    features = csr_matrix(features)
    # print(type(adj), type(features))

    rcnn = RGCN(nnodes=adj.shape[0],nfeat=features.shape[1], nhid=8, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    rcnn = rcnn.to(device)
    
    # pyg_data.update_edge_index(perturbed_adj)  ############ inplace operation
    rcnn.fit(features, adj, labels, idx_train, idx_val, patience=30, train_iters=100, verbose=False)  ############ train with earlystopping

    #gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    #gcn = gcn.to(device)
    #gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    rcnn.eval()
    output = rcnn.predict()
    probs = torch.exp(output[[target_node]])[0]
    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    # print(acc_test.item())
    return acc_test.item()

def test_acc_GCN(adj, features, data,target_node):
    idx_train, idx_val = data.idx_train, data.idx_val
    labels = data.labels
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()

def test_acc_GIN(adj,features, data, target_node):
    labels = data.labels

    ''' test on GIN '''
    # reset feature to 0------------------------Remove this line if you don't want to feed GIN with node features.
    data.features = csr_matrix(data.features.shape, dtype=int)
    pyg_data = Dpr2Pyg(data)


    gin = GIN(nfeat=features.shape[1], nhid=8, heads=8, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gin = gin.to(device)
    perturbed_adj = adj.tocsr()
    pyg_data.update_edge_index(perturbed_adj)
    gin.fit(pyg_data, verbose=False)
    gin.eval()
    output = gin.predict()
    acc_test = (output.argmax(1)[target_node] == labels[target_node])

    return acc_test.item()

def test_acc_GSAGE(adj,features, data, target_node):
    labels= data.labels
    ''' test on GSAGE '''
    pyg_data = Dpr2Pyg(data)
    gsage = GraphSAGE(nfeat=features.shape[1], nhid=8, heads=8, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gsage = gsage.to(device)
    perturbed_adj = adj.tocsr()           ###########   perturbed_data.adj
    pyg_data.update_edge_index(perturbed_adj)  ############ inplace operation
    gsage.fit(pyg_data, verbose=False)
    gsage.eval()
    output = gsage.predict()
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()

def test_acc_RGCN(adj, features, data,target_node):
    idx_train, idx_val = data.idx_train, data.idx_val
    labels = data.labels
    perturbed_adj=adj
    ''' test on GCN '''
    rgcn =RGCN(nnodes=perturbed_adj.shape[0], nfeat=features.shape[1], nclass=labels.max()+1,
                nhid=32, device=device)
    rgcn = rgcn.to(device)

    rgcn.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
    # You can use the inner function of model to test
    #model.test(idx_test)

    #GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    #gcn = gcn.to(device)
    #gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    rgcn.eval()
    output = rgcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()


def test_acc_MDGCN(adj,features, data, target_node):
    labels= data.labels
    ''' test on GSAGE '''
    pyg_data = Dpr2Pyg(data)
    mgcn = MedianGCN(nfeat=features.shape[1],nhid=16,nclass=labels.max().item() + 1, dropout=0.5, device=device)
    #gsage = GraphSAGE(nfeat=features.shape[1], nhid=8, heads=8, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    mgcn = mgcn.to(device)
    perturbed_adj = adj.tocsr()           ###########   perturbed_data.adj
    pyg_data.update_edge_index(perturbed_adj)  ############ inplace operation
    mgcn.fit(pyg_data, verbose=False)
    mgcn.eval()
    output = mgcn.predict()
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()


def test_acc_JacGCN(adj, features, data,target_node):
    idx_train, idx_val = data.idx_train, data.idx_val
    labels = data.labels
    perturbed_adj=adj
    ''' test on GCN '''
    # Setup Defense Model
    jacgcn = GCNJaccard(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)
    jacgcn = jacgcn.to(device)
    jacgcn.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=0.01)
    jacgcn.eval()
    output = jacgcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()


def test_acc_SVDGCN(adj, features, data,target_node):
    idx_train, idx_val = data.idx_train, data.idx_val
    labels = data.labels
    perturbed_adj=adj
    ''' test on GCN '''
    # Setup Defense Model
    # Setup Defense Model
    svdgcn = GCNSVD(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)

    svdgcn = svdgcn.to(device)
    svdgcn.fit(features, perturbed_adj, labels, idx_train, idx_val, k=15, verbose=True)
    svdgcn.eval()
    output = svdgcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()
