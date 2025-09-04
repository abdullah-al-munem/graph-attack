import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from copy import deepcopy
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv, GINConv
import gc

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

class GIN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, lr=0.01,
            weight_decay=5e-4, with_bias=True, device=device):

        super(GIN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        # Reduced hidden dimensions for memory efficiency
        hidden_dim = min(nhid, 64)  # Cap hidden dimension to reduce memory usage
        
        self.gc1 = GINConv(
            Sequential(Linear(nfeat, hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()))
        self.gc2 = GINConv(
            Sequential(Linear(hidden_dim, hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()))
        
        # Use only 2 GIN layers instead of 3 to reduce memory
        self.lin1 = Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = Linear(hidden_dim, nclass)

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Clear cache before forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use gradient checkpointing to save memory
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
            h1 = self.gc1(x, edge_index)
            h2 = self.gc2(h1, edge_index)
            
            # Concatenate only 2 layers instead of 3
            h = torch.cat((h1, h2), dim=1)
            
            # Clear intermediate tensors
            del h1, h2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            h = self.lin1(h)
            h = h.relu()
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.lin2(h)
            
        return F.log_softmax(h, dim=1)

    def initialize(self):
        """Initialize parameters of GIN."""
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, pyg_data, train_iters=1000, initialize=True, verbose=False, patience=100, **kwargs):
        """Train the GIN model with memory optimization."""
        
        if initialize:
            self.initialize()

        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.data = pyg_data[0].to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """Early stopping based on validation loss with memory optimization."""
        if verbose:
            print('=== training GIN model ===')

        # Use mixed precision training to reduce memory usage
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            
            # Clear cache at the beginning of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Use mixed precision if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.forward(self.data)
                    loss_train = F.nll_loss(output[train_mask], labels[train_mask])
                
                scaler.scale(loss_train).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = self.forward(self.data)
                loss_train = F.nll_loss(output[train_mask], labels[train_mask])
                loss_train.backward()
                optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            # Validation with no_grad to save memory
            with torch.no_grad():
                self.eval()
                output = self.forward(self.data)
                loss_val = F.nll_loss(output[val_mask], labels[val_mask])

                if best_loss_val > loss_val:
                    best_loss_val = loss_val
                    self.output = output
                    weights = deepcopy(self.state_dict())
                    patience = early_stopping
                else:
                    patience -= 1
                    
            # Clear variables to free memory
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val))
        self.load_state_dict(weights)

    def test(self):
        """Evaluate GIN performance on test set."""
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        
        with torch.no_grad():
            output = self.forward(self.data)
            loss_test = F.nll_loss(output[test_mask], labels[test_mask])
            # Note: utils.accuracy needs to be imported or defined
            # acc_test = utils.accuracy(output[test_mask], labels[test_mask])
            
            # Alternative accuracy calculation
            pred = output[test_mask].argmax(dim=1)
            acc_test = (pred == labels[test_mask]).float().mean()
            
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def predict(self):
        """Returns output (log probabilities) of GIN."""
        self.eval()
        with torch.no_grad():
            return self.forward(self.data)


