# THIS IS VERY COPY PASTA AN NOT REFINED YET!!!
# ref: https://www.geeksforgeeks.org/deep-learning/graph-neural-networks-with-pytorch/

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import os

# =====================
# Load Breast Cancer Data
# =====================
try:
    df = pd.read_csv("breast-cancer.csv")  # kaggle file
except:
    from sklearn.datasets import load_breast_cancer
    bc = load_breast_cancer()
    df = pd.DataFrame(bc.data, columns=bc.feature_names)
    df["diagnosis"] = bc.target

# Drop ID column if present
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Labels
label_col = "diagnosis"
y_raw = df[label_col].values
if y_raw.dtype == object or y_raw.dtype == str:
    y = LabelEncoder().fit_transform(y_raw)  # B->0, M->1 (typical)
else:
    y = y_raw.astype(int)

# Features
X = df.drop(columns=[label_col]).values
X = StandardScaler().fit_transform(X).astype(np.float32)

# =====================
# Build Graph (k-NN)
# =====================
k = 8  # sarah used k=8
nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
_, indices = nbrs.kneighbors(X)

rows, cols = [], []
for i in range(len(X)):
    for j in indices[i, 1:]:
        rows.append(i)
        cols.append(j)

edge_index = torch.tensor([rows, cols], dtype=torch.long)
edge_index = to_undirected(edge_index)

# =====================
# Train/Test Masks
# =====================
idx = np.arange(len(X))
train_idx, test_idx = train_test_split(idx, stratify=y, test_size=0.2, random_state=42)

train_mask = torch.zeros(len(X), dtype=torch.bool)
test_mask = torch.zeros(len(X), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

# =====================
# PyG Data object
# =====================
graph_data = Data(
    x=torch.tensor(X, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(y, dtype=torch.long),
    train_mask=train_mask,
    test_mask=test_mask,
)

# =====================
# GCN Model
# =====================
class CustomGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = GCNConv(input_dim, hidden_dim)
        self.layer2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layer2(x, edge_index)
        return F.log_softmax(x, dim=1)

input_features = graph_data.num_node_features
num_classes = len(np.unique(y))
model = CustomGNN(input_features, 16, num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# =====================
# Training
# =====================
def train_model():
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = F.nll_loss(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(1, 201):
    loss_value = train_model()
    if epoch % 20 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss_value:.4f}")

# =====================
# Evaluation + Confusion Matrix
# =====================
model.eval()
with torch.no_grad():
    out = model(graph_data.x, graph_data.edge_index)

logits = out[graph_data.test_mask]
probs = logits.softmax(dim=1)[:, 1].cpu().numpy()
preds = logits.argmax(dim=1).cpu().numpy()
labels = graph_data.y[graph_data.test_mask].cpu().numpy()

# Metrics
accuracy = (preds == labels).mean()
print(f"Test Accuracy: {accuracy:.4f}")
print("Precision:", precision_score(labels, preds))
print("Recall:",    recall_score(labels, preds))
print("F1:",        f1_score(labels, preds))
print("ROC-AUC:",   roc_auc_score(labels, probs))

# Confusion matrix (and save figure)
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap="Blues")
plt.title("Breast Cancer – GCN Confusion Matrix")
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.show()
