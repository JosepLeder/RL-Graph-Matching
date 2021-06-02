import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch.nn import Linear, ReLU
from torch_geometric.nn import GraphConv


class GraphConvNet(nn.Module):

    def __init__(self, n_feat, n_hid, n_out):
        super(GraphConvNet).__init__()
        self.conv1 = GraphConv(n_feat, n_hid)
        self.conv2 = GraphConv(n_hid, n_hid * 2)
        self.conv3 = GraphConv(n_hid * 2, n_out)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index))
        data.x = F.elu(self.conv2(data.x, data.edge_index))
        data.x = F.elu(self.conv3(data.x, data.edge_index))
        x_1 = scatter_mean(data.x, data.batch, dim=0)
        x = x_1

        return x


class DoubleGraphConvNet(nn.Module):

    def __init__(self, graph, subgraph, point):
        super(DoubleGraphConvNet).__init__()

        self.graph_conv = GraphConvNet(graph.n_feat, graph.n_feat * 2, graph.n_feat * 3)
        self.subgraph_conv = GraphConvNet(subgraph.n_feat, subgraph.n_feat * 2, subgraph.n_feat * 3)

        self.l1 = Linear(graph.n_feat * 3 + subgraph.n_feat * 3 + point, 600)
        self.l2 = Linear(600, 256)
        self.l3 = Linear(256, graph.n_feat)


    def forward(self, graph, subgraph, point):
        x1 = self.graph_conv(graph)
        x2 = self.subgraph_conv(subgraph)
        x = torch.cat([x1, x2, point])

        x = ReLU(self.l1(x))
        x = ReLU(self.l2(x))
        x = self.l3(x)
        return x




