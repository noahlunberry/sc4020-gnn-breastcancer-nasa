import torch
from torch import Tensor
import torch_geometric as tgeo
import pandas as pd


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = tgeo.nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = tgeo.nn.GCNConv(hidden_channels, out_channels)
        self.attent1 = tgeo.nn.GATConv(in_channels, hidden_channels)
        self.attent2 = tgeo.nn.GATConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.attent1(x, edge_index).relu()
        x = self.attent2(x, edge_index)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = tgeo.nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = tgeo.nn.GCNConv(hidden_channels, out_channels)
        self.attent1 = tgeo.nn.GATConv(in_channels, hidden_channels)
        self.attent2 = tgeo.nn.GATConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
col_names = index_names + setting_names + sensor_names

dftrain = pd.read_csv('NASA_data/train_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
dfvalid = pd.read_csv('NASA_data/test_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
y_valid = pd.read_csv('NASA_data/RUL_FD001.txt',sep='\s+',header=None,index_col=False,names=['RUL'])




index_last = 0
current_engine= 1
for i in range(dftrain.shape[0]):
    if i<dftrain.shape[0]-10:
        if dftrain.iloc[i+1,0]>current_engine:
            current_engine +=1
            max_cycles = dftrain.iloc[i,1]
            for j in range(index_last,i+1):
                dftrain.iloc[j,1] = max_cycles - dftrain.iloc[j,1]
            index_last = i+1
    if i+1==dftrain.shape[0]:
        max_cycles = dftrain.iloc[i, 1]
        for j in range(index_last, i + 1):
            dftrain.iloc[j, 1] = max_cycles - dftrain.iloc[j, 1]
        index_last = i + 1
y_train = torch.tensor(dftrain.iloc[:,1],dtype=torch.float32)
x_train = torch.tensor(dftrain.iloc[:,2:].to_numpy(),dtype=torch.float32)
y_train = y_train.unsqueeze(1)

y_valid = torch.tensor(y_valid.iloc[:,0].to_numpy(),dtype=torch.float32)
x_valid = torch.tensor(dfvalid.iloc[:,2:].to_numpy(),dtype=torch.float32)



model_GAT = GAT(in_channels=24, hidden_channels=256, out_channels=1)
model_NN = torch.nn.Sequential(torch.nn.Linear(24,128),
                               torch.nn.ReLU(),torch.nn.Linear(128,1))

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_NN.parameters(), lr=0.001)

print("training started")
for epoch in range(10000):
    pred = model_NN(x_train)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(epoch, loss.item()/x_train.size(0))


pred = model_NN(x_valid)
loss = loss_fn(pred, y_valid)
print(loss.item())
breakpoint()




for epoch in range(200):
    pred = model_GAT(x_train, data.edge_index)
    loss = torch.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()