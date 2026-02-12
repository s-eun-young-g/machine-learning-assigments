import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv

torch.manual_seed(0)

### load dataset ###
dataset = Planetoid(root="data/Planetoid", name="Cora", transform=NormalizeFeatures())
data = dataset[0]

print("num nodes:", data.num_nodes)
print("num edges:", data.num_edges)
print("num features:", dataset.num_features)
print("num classes:", dataset.num_classes)

### define a basic 2-layer GCN ###
class StudentGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)  # typical student choice
        h = self.conv2(h, edge_index)
        return h

model = StudentGCN(dataset.num_features, 16, dataset.num_classes)

### training setup ###
device = "cuda" if torch.cuda.is_available() else "cpu"
data = data.to(device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def accuracy(logits, y):
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()

### train loop ###
for epoch in range(201):
    model.train()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)

            train_acc = accuracy(out[data.train_mask], data.y[data.train_mask])
            val_acc   = accuracy(out[data.val_mask], data.y[data.val_mask])
            test_acc  = accuracy(out[data.test_mask], data.y[data.test_mask])

        print(f"epoch {epoch:3d} | loss {loss.item():.4f} | train {train_acc:.3f} | val {val_acc:.3f} | test {test_acc:.3f}")
