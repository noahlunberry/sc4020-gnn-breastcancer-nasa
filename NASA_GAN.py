import torch
from torch import Tensor
import torch_geometric as tgeo

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
        x = self.attent1(x, edge_index).relu()
        x = self.attent2(x, edge_index)
        return x


model = GCN(in_channels=2, hidden_channels=2, out_channels=1)

data = 0


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    pred = model(data.x, data.edge_index)
    loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()