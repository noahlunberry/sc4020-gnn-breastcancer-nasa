import torch
from torch import Tensor
import torch_geometric as tgeo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


# --- Choose model type (user input) ---
print("Which model would you like to train? Type: 'GCN' or 'GAT'")
model_type = input()


# --- GCN Model ---
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = tgeo.nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = tgeo.nn.GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    

# --- GAT Model ---
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.attent1 = GATConv(in_channels, hidden_channels, heads=8, concat=True, dropout=0.2)
        self.attent2 = GATConv(hidden_channels * 8, out_channels, heads=1, concat=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.attent1(x, edge_index).relu()
        x = self.attent2(x, edge_index)
        return x


# --- Read in data ---
index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i+1) for i in range(0, 21)]
col_names = index_names + setting_names + sensor_names

dftrain = pd.read_csv('NASA_data/train_FD001.txt', sep='\s+', header=None, names=col_names)
dfvalid = pd.read_csv('NASA_data/test_FD001.txt', sep='\s+', header=None, names=col_names)


# --- Compute RULs ---
def compute_rul(df):
    index_last = 0
    current_engine = 1
    for i in range(df.shape[0]):
        if i<df.shape[0]-1:
            if df.iloc[i+1,0]>current_engine:
                current_engine +=1
                max_cycles = df.iloc[i,1]
                for j in range(index_last, i+1):
                    df.loc[j,'RUL'] = max_cycles - df.loc[j, 'time_cycles']
                index_last = i+1
        if i+1==df.shape[0]:
            max_cycles = df.iloc[i, 1]
            for j in range(index_last, i+1):
                df.loc[j, 'RUL'] = max_cycles - df.loc[j, 'time_cycles']
            index_last = i + 1


compute_rul(dftrain)
compute_rul(dfvalid)


# --- Normalise inputs ---
feature_cols = setting_names + sensor_names + ['time_cycles']
scaler = StandardScaler()

X_train_np = dftrain[feature_cols].values
X_valid_np = dfvalid[feature_cols].values

X_train_scaled = scaler.fit_transform(X_train_np).astype(np.float32)
X_valid_scaled = scaler.transform(X_valid_np).astype(np.float32)


# --- Load inputs ---
x_train = torch.tensor(X_train_scaled, dtype=torch.float32)
x_valid = torch.tensor(X_valid_scaled, dtype=torch.float32)


# --- Load outputs ---
y_train = torch.tensor(dftrain['RUL'].values, dtype=torch.float32).unsqueeze(1)
y_valid = torch.tensor(dfvalid['RUL'].values, dtype=torch.float32).unsqueeze(1)


# --- Build edges ---
edge_list = []
for unit in dftrain['unit_number'].unique():
    idx = dftrain[dftrain['unit_number'] == unit].index.tolist()
    for i in range(len(idx) - 1):
        edge_list.append([1, 1])  


edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  


# --- PyG Data object ---
data = Data(x=x_train, edge_index=edge_index, y=y_train)


# --- Define GCN/GAT model ---
if model_type == GCN:
    model = GCN(in_channels=x_train.shape[1], hidden_channels=64, out_channels=1)
else: 
    model = GAT(in_channels=x_train.shape[1], hidden_channels=64, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.MSELoss()


# --- Training loop ---
model.train()
print("Training the " + model_type + " model...")
for epoch in range(300):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# --- Predict on validation set ---
model.eval()
with torch.no_grad():
    edge_list_val = []
    for unit in dfvalid['unit_number'].unique():
        idx = dfvalid[dfvalid['unit_number'] == unit].index.tolist()
        for i in range(len(idx) - 1):
            edge_list_val.append([idx[i], idx[i + 1]])

    edge_index_val = torch.tensor(edge_list_val, dtype=torch.long).t().contiguous()
    data_valid = Data(x=x_valid, edge_index=edge_index_val, y=y_valid)

    y_pred = model(data_valid.x, data_valid.edge_index).squeeze().cpu().numpy()
    y_true = y_valid.squeeze().cpu().numpy()


dfvalid_plot = dfvalid.copy()
dfvalid_plot['predicted_RUL'] = y_pred
dfvalid_plot['true_RUL'] = y_true


# --- Plots ---
unique_engines = dfvalid_plot['unit_number'].unique()

print("\nGenerating plots...\n")
for engine_id in unique_engines:
    engine_data = dfvalid_plot[dfvalid_plot['unit_number'] == engine_id]

    plt.figure(figsize=(10, 5))
    plt.plot(engine_data['time_cycles'], engine_data['true_RUL'], label='True RUL', marker='o', color='xkcd:periwinkle blue')
    plt.plot(engine_data['time_cycles'], engine_data['predicted_RUL'], label='Predicted RUL', marker='x', color='xkcd:tomato')
    plt.xlabel('Cycle Number')
    plt.ylabel('Remaining Useful Life')
    plt.title(f'Engine {engine_id}: GAT RUL Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # break  
