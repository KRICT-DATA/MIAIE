import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from model.rgat_conv import RGATConv


class GAT(nn.Module):
    def __init__(self, num_node_feats, num_graph_feats, dim_out):
        super(GAT, self).__init__()
        self.gc1 = RGATConv(num_node_feats, 256)
        self.gc2 = RGATConv(256, 256)
        self.gc3 = RGATConv(256, 256)
        self.fc1 = nn.Linear(256 + num_graph_feats, 196)
        self.fc2 = nn.Linear(196, dim_out)
        self.estimation = False

    def forward(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index))
        h = F.relu(self.gc2(h, g.edge_index))
        h = F.relu(self.gc3(h, g.edge_index))

        if self.estimation:
            hg = torch.mean(h, dim=0).view(1, -1)
        else:
            hg = global_mean_pool(h, g.batch)

        h = F.relu(self.fc1(torch.cat([hg, g.mol_feats], dim=1)))
        out = self.fc2(h)

        return out

    def est(self, flag):
        self.estimation = flag
        self.gc1.estimation = flag
        self.gc2.estimation = flag
        self.gc3.estimation = flag

    def compute_rgsa(self, g):
        num_atoms = g.x.shape[0]
        num_edges = g.edge_index.shape[1]
        attn_scores1 = torch.zeros([num_atoms, num_atoms])
        attn_scores2 = torch.zeros([num_atoms, num_atoms])
        attn_scores3 = torch.zeros([num_atoms, num_atoms])
        out = self.forward(g)
        attn_gc1 = self.gc1.attns
        attn_gc2 = self.gc2.attns
        attn_gc3 = self.gc3.attns
        rgsa_gc1 = torch.zeros(num_atoms)
        rgsa_gc2 = torch.zeros(num_atoms)
        rgsa_gc3 = torch.zeros(num_atoms)

        for i in range(0, num_edges):
            attn_scores1[g.edge_index[0, i], attn_gc1[i][0]] = attn_gc1[i][1]
            attn_scores2[g.edge_index[0, i], attn_gc2[i][0]] = attn_gc1[i][1]
            attn_scores3[g.edge_index[0, i], attn_gc3[i][0]] = attn_gc1[i][1]

        for i in range(0, num_atoms):
            attn_scores1[i, i] = attn_gc1[num_edges + i][1]
            attn_scores2[i, i] = attn_gc2[num_edges + i][1]
            attn_scores3[i, i] = attn_gc3[num_edges + i][1]

        for i in range(0, num_atoms):
            for j in range(0, num_atoms):
                rgsa_gc1[i] += attn_scores1[i, j]
                rgsa_gc2[i] += attn_scores2[i, j]
                rgsa_gc3[i] += attn_scores3[i, j]

        rgsa_gc1 = torch.exp(rgsa_gc1)
        rgsa_gc2 = torch.exp(rgsa_gc2)
        rgsa_gc3 = torch.exp(rgsa_gc3)

        rgsa_gc1 /= torch.sum(rgsa_gc1)
        rgsa_gc2 /= torch.sum(rgsa_gc2)
        rgsa_gc3 /= torch.sum(rgsa_gc3)

        return out, rgsa_gc1, rgsa_gc2, rgsa_gc3
