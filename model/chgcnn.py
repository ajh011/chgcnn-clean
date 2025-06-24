import torch
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn.conv import HeteroConv, HypergraphConv
import torch_geometric.nn as nn
import torch

from .convolutions.inter_conv import CHGInterConv
from .convolutions.agg_conv import CHGConv



class CrystalHypergraphConv(torch.nn.Module):
    def __init__(self, classification, h_dim = 64, hout_dim = 128, hidden_hedge_dim = 64, layers = [], n_layers = 3, bonds = True, motifs = True, triplets = False, update_hedges = False):
        super().__init__()

        self.classification = classification
        self.update_hedges = update_hedges

        bond_hedge_dim = 40
        motif_hedge_dim = 94 
        triplet_hedge_dim = 40
        self.embed = nn.Linear(92, h_dim)
        self.bembed = nn.Linear(bond_hedge_dim, hidden_hedge_dim)
        self.membed = nn.Linear(motif_hedge_dim, hidden_hedge_dim)
        self.convs = torch.nn.ModuleList() 

        if layers == []:
            for n in range(n_layers):
                if bonds == True:
                    layers.append('b')
                if triplets == True:
                    layers.append('t')
                if motifs == True:
                    layers.append('m')

        for l in layers:
            if l == 'b':
                conv = HeteroConv({('bond', 'contains', 'atom'): CHGConv(node_fea_dim = h_dim, hedge_fea_dim = hidden_hedge_dim, update_hedges = self.update_hedges)})
                self.convs.append(conv)
            elif l == 't':    
                conv = HeteroConv({('triplet', 'contains', 'atom'): CHGConv(node_fea_dim = h_dim, hedge_fea_dim = triplet_hedge_dim, update_hedges = self.update_hedges)})
                self.convs.append(conv)
            elif l == 'm':    
                conv = HeteroConv({('motif', 'contains', 'atom'): CHGConv(node_fea_dim = h_dim, hedge_fea_dim = hidden_hedge_dim, update_hedges = self.update_hedges)})
                self.convs.append(conv)
        
        self.layers = layers
        print(f'Using {layers} CHGConv Layers!')

        self.l1 = nn.Linear(h_dim, h_dim)
        self.l2 = nn.Linear(h_dim, hout_dim)
        self.l3 = nn.Linear(hout_dim, hout_dim)
        self.act1 = torch.nn.Softplus()
        self.act2 = torch.nn.Softplus()
        self.act3 = torch.nn.Softplus()

        if self.classification:
            self.out = nn.Linear(hout_dim, 2)
            self.sigmoid = torch.nn.Sigmoid()
            self.dropout = torch.nn.Dropout()
        else:
            self.out = nn.Linear(hout_dim,1)
 
    def forward(self, data):
        hyperedge_attrs_dict = data.hyperedge_attrs_dict
        hyperedge_index_dict = data.hyperedge_index_dict
        batch = data['atom'].batch
        hyperedge_attrs_dict['atom'] = self.embed(hyperedge_attrs_dict['atom'].float())
        hyperedge_attrs_dict['bond'] = self.bembed(hyperedge_attrs_dict['bond'].float())
        hyperedge_attrs_dict['motif'] = self.membed(hyperedge_attrs_dict['motif'].float())
        for conv in self.convs:
            hyperedge_attrs_dict_update = conv(hyperedge_attrs_dict, hyperedge_index_dict)
            hyperedge_attrs_dict['atom'] = hyperedge_attrs_dict_update['atom'].relu()
        x = scatter(hyperedge_attrs_dict['atom'], batch, dim=0, reduce='mean')
        x = self.l1(x)
        if self.classification:
            x = self.dropout(x)
        x = self.act1(x)
        x = self.l2(x)
        x = self.act2(x)
        x = self.l3(x)
        x = self.act3(x)
        output = self.out(x)
        if self.classification:
            output = self.sigmoid(output)
        return output


       
