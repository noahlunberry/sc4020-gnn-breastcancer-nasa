import torch
from torch import Tensor
import torch_geometric as tgeo
import pandas as pd


class GAN(torch.nn.Module):
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
    
def adapt_data_input(dataframe):
    index_last = 0
    current_engine = 1
    for i in range(dataframe.shape[0]):
        if i < dataframe.shape[0] - 10:
            if dataframe.iloc[i + 1, 0] > current_engine:
                current_engine += 1
                max_cycles = dataframe.iloc[i, 1]
                for j in range(index_last, i + 1):
                    dataframe.iloc[j, 1] = max_cycles - dataframe.iloc[j, 1]
                index_last = i + 1
        if i + 1 == dataframe.shape[0]:
            max_cycles = dataframe.iloc[i, 1]
            for j in range(index_last, i + 1):
                dataframe.iloc[j, 1] = max_cycles - dataframe.iloc[j, 1]
            index_last = i + 1
    return dataframe
    


index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
col_names = index_names + setting_names + sensor_names

dftrain = pd.read_csv('archive/CMaps/train_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
dfvalid = pd.read_csv('archive/CMaps/test_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
y_valid = pd.read_csv('archive/CMaps/RUL_FD001.txt',sep='\s+',header=None,index_col=False,names=['RUL'])

dftrain = adapt_data_input(dftrain)
dfvalid = adapt_data_input(dfvalid)



y_train = torch.tensor(dftrain.iloc[:,1],dtype=torch.float32)
y_train = y_train.unsqueeze(1)
x_train = torch.tensor(dftrain.iloc[:,2:].to_numpy(),dtype=torch.float32)

y_valid = torch.tensor(dfvalid.iloc[:,1],dtype=torch.float32)
x_valid = torch.tensor(dfvalid.iloc[:,2:].to_numpy(),dtype=torch.float32)



model_GAN = GAN(in_channels=24, hidden_channels=256, out_channels=1)
model_NN = torch.nn.Sequential(torch.nn.Linear(24,256),
                               torch.nn.LeakyReLU(),torch.nn.Linear(256,1))

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_NN.parameters(), lr=0.001)

print("training started")
for epoch in range(15000):
    pred = model_NN(x_train)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(epoch, loss.item()/x_train.size(0))


pred = model_NN(x_train)
loss = loss_fn(pred, y_train)
print(loss.item())
for i in range(pred.shape[0]):
    print(pred[i]-y_train[i])



breakpoint()




for epoch in range(200):
    pred = model_GAN(x_train, data.edge_index)
    loss = torch.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()