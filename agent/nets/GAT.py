import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv



class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_head, dropout, alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GATConv(n_feat, n_hid, dropout=dropout, negative_slope=alpha, concat=True)
                           for _ in range(n_head)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATConv(n_hid * n_head, 1, dropout=dropout, negative_slope=alpha, concat=False)

    def forward(self, data):
        x, adj = data.x, data.edge_index

        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

