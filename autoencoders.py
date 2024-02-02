import torch
import time
from torch_geometric.nn import GAE, VGAE, GCNConv

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

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    
class GAEModel:
    def __init__(self, in_channels, out_channels, epochs=1000, device=device, lr=0.01, variational=False):
        self.model = GAE(GCNEncoder(in_channels, out_channels)).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.variational = variational
        self.device = device

    def train(self, train_data):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(train_data.x.to(self.device), train_data.edge_index.to(self.device))
        loss = self.model.recon_loss(z, train_data.pos_edge_label_index)
        if self.variational:
            loss = loss + (1 / train_data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self, test_data):
        self.model.eval()
        z = self.model.encode(test_data.x.to(self.device), test_data.edge_index.to(self.device))
        return self.model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)

    def train_and_test(self, train_data, test_data):
        times = []
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            loss = self.train(train_data)
            auc, ap = self.test(test_data)
            if epoch % 100 == 0:
                print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
            times.append(time.time() - start)
        print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

    def get_encoding(self, data):
        encoded = self.model.encode(data.x.to(self.device), data.edge_index.to(self.device))
        return encoded
    
    def get_decoding(self, data, encoded):
        decoded = self.model.decoder(encoded, data.edge_index)
        return decoded

    def get_decoded_edge_index(self, data, decoded, threshold=0.90):
        reshaped_edge_index = torch.transpose(data.edge_index, 0, 1)
        decoded_edge_index_list = []
        for i in range(len(decoded)):
            if decoded[i] > threshold:
                decoded_edge_index_list.append(list(reshaped_edge_index[i]))

        decoded_edge_index = torch.tensor(decoded_edge_index_list, dtype=torch.long).t()
        return decoded_edge_index

class VGAEModel:
    def __init__(self, in_channels, out_channels, epochs=1000, device=device, lr=0.01, variational=True):
        self.model = VGAE(VariationalGCNEncoder(in_channels, out_channels)).to(device) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.variational = variational
        self.device = device

    def train(self, train_data):
        self.model.train()
        self.optimizer.zero_grad() 
        z = self.model.encode(train_data.x.to(self.device), train_data.edge_index.to(self.device))
        loss = self.model.recon_loss(z, train_data.pos_edge_label_index)
        if self.variational:
            loss = loss + (1 / train_data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self, test_data):
        self.model.eval()
        z = self.model.encode(test_data.x.to(self.device), test_data.edge_index.to(self.device))
        return self.model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)

    def train_and_test(self, train_data, test_data):
        times = []
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            loss = self.train(train_data)
            auc, ap = self.test(test_data)
            if epoch % 100 == 0:
                print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
            times.append(time.time() - start)
        print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

    def get_encoding(self, data):
        encoded = self.model.encode(data.x.to(self.device), data.edge_index.to(self.device))
        return encoded
    
    def get_decoding(self, data, encoded):
        decoded = self.model.decoder(encoded, data.edge_index)
        return decoded

    def get_decoded_edge_index(self, data, decoded, threshold=0.90):
        reshaped_edge_index = torch.transpose(data.edge_index, 0, 1)
        decoded_edge_index_list = []
        for i in range(len(decoded)):
            if decoded[i] > threshold:
                decoded_edge_index_list.append(list(reshaped_edge_index[i]))

        decoded_edge_index = torch.tensor(decoded_edge_index_list, dtype=torch.long).t()
        return decoded_edge_index
    