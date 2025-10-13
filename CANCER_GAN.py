#!/usr/bin/env python3
"""
Cancer-GAT: Graph Attention Network on Breast Cancer Dataset

Steps:
1. Load breast-cancer.csv (drop id column if present, map diagnosis: B=0, M=1).
2. Scale feature columns with StandardScaler.
3. Build a k-NN graph (nodes=patients, edges=similarity; cosine via L2-normalize + Euclidean KNN, k=10).
4. Split nodes into train/test (80/20, stratified by class).
5. Train a Graph Attention Network (GAT) to classify nodes.
6. Print Accuracy, Precision, Recall, F1, and ROC-AUC on the test set.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, knn_graph
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# 1) load and preprocess dataset
df = pd.read_csv("breast-cancer.csv")

# drop id column if it exists
if "id" in df.columns:
    df = df.drop(columns=["id"])

# encode labels: B=0 (benign), M=1 (malignant)
if "diagnosis" not in df.columns:
    raise ValueError("Expected a 'diagnosis' column with values 'B' or 'M'.")
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})
if df["diagnosis"].isna().any():
    raise ValueError("Found labels other than 'B'/'M' in 'diagnosis' column.")

# separate features and labels
X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# convert to PyTorch tensors
x_tensor = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)

# 2) build k-NN graph (k=10 neighbors)
# CPU-safe cosine KNN: L2-normalize then use Euclidean KNN (equivalent to cosine KNN)
x_tensor = F.normalize(x_tensor, p=2, dim=1)
edge_index = knn_graph(x_tensor, k=10, loop=False)  # <- no cosine=True on CPU
# (optional) make edges undirected for better message passing
edge_index = to_undirected(edge_index, num_nodes=x_tensor.size(0))

print(f"Graph built: {x_tensor.shape[0]} nodes, {edge_index.shape[1]} edges")

# 3) split nodes into train/test
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(
    indices, stratify=y, test_size=0.2, random_state=42
)
train_idx = torch.tensor(train_idx, dtype=torch.long)
test_idx = torch.tensor(test_idx, dtype=torch.long)

print(f"Train nodes: {len(train_idx)}, Test nodes: {len(test_idx)}")

# 4) define the GAT model
class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels,
                            heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

# 5) training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GATNet(in_channels=x_tensor.size(1),
               hidden_channels=8,
               out_channels=2,
               heads=8,
               dropout=0.6).to(device)

x_tensor = x_tensor.to(device)
y_tensor = y_tensor.to(device)
edge_index = edge_index.to(device)
train_idx = train_idx.to(device)
test_idx = test_idx.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# 6) train
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(x_tensor, edge_index)
    loss = criterion(out[train_idx], y_tensor[train_idx])
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 7) evaluate
model.eval()
with torch.no_grad():
    out = model(x_tensor, edge_index)

test_logits = out[test_idx]
test_probs = test_logits.softmax(dim=1)[:, 1].cpu().numpy()
test_pred = test_logits.argmax(dim=1).cpu().numpy()
true_labels = y_tensor[test_idx].cpu().numpy()

acc = accuracy_score(true_labels, test_pred)
prec = precision_score(true_labels, test_pred, pos_label=1)
rec = recall_score(true_labels, test_pred, pos_label=1)
f1 = f1_score(true_labels, test_pred, pos_label=1)
auc = roc_auc_score(true_labels, test_probs)

print("\nTest Set Performance (GAT):")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")

cm = confusion_matrix(true_labels, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap="Blues")
plt.title("Breast Cancer – GAT Confusion Matrix")
plt.tight_layout()
plt.show()