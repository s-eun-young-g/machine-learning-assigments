import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

torch.manual_seed(0)

### make toy time series data ###
def make_series(n=20000):
    t = torch.linspace(0, 200 * math.pi, n)
    y = torch.sin(t) + 0.15 * torch.randn(n)  # noisy sine
    return y

y = make_series()

seq_len = 64

### build dataset: X = y[t:t+seq_len], target = y[t+1:t+seq_len+1] ###
Xs, Ys = [], []
for i in range(len(y) - seq_len - 1):
    Xs.append(y[i:i+seq_len])
    Ys.append(y[i+1:i+seq_len+1])

X = torch.stack(Xs)[:, :, None]  # (N, T, 1)
Y = torch.stack(Ys)[:, :, None]  # (N, T, 1)

# split
N = X.shape[0]
train_N = int(0.8 * N)
X_train, Y_train = X[:train_N], Y[:train_N]
X_test,  Y_test  = X[train_N:], Y[train_N:]

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, Y_test), batch_size=128)

### tiny decoder-only transformer ###
class TinyTimeSeriesTransformer(nn.Module):
    def __init__(self, seq_len, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.in_proj = nn.Linear(1, d_model)

        # student-ish: learnable positional embeddings (not sinusoidal)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B,T,1)
        h = self.in_proj(x) + self.pos

        # causal mask so position t can't see future positions > t
        T = h.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()

        h = self.enc(h, mask=causal_mask)
        pred = self.out(h)  # (B,T,1)
        return pred

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyTimeSeriesTransformer(seq_len=seq_len).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

### training loop ###
for epoch in range(5):
    model.train()
    running = 0.0
    for xb, yb in tqdm(train_loader, desc=f"epoch {epoch}"):
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # nice habit
        opt.step()

        running += loss.item()

    # quick test loss
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            test_loss += loss_fn(pred, yb).item()

    print("avg train loss:", running / len(train_loader), "avg test loss:", test_loss / len(test_loader))

### simple next-step prediction demo ###
model.eval()
with torch.no_grad():
    sample = X_test[0:1].to(device)  # (1,T,1)
    pred = model(sample)[0, :, 0].cpu()
print("first 10 preds:", pred[:10])
print("first 10 true: ", Y_test[0, :10, 0])
