from __future__ import print_function, division

import torch
import torch.nn as nn

from torch_geometric.nn import aggr
from torch.nn import BatchNorm1d, Linear

class CHGConv(nn.Module):
    def __init__(self, node_fea_dim=92, hedge_fea_dim=35, batch_norm = True):
        super().__init__()
        self.batch_norm = batch_norm
        self.node_fea_dim = node_fea_dim
        self.hedge_fea_dim = hedge_fea_dim

        self.lin_f1 = Linear(node_fea_dim+hedge_fea_dim, hedge_fea_dim+node_fea_dim)
        self.lin_c1 = Linear(node_fea_dim+hedge_fea_dim, hedge_fea_dim)
        self.lin_f2 = Linear(2*node_fea_dim+hedge_fea_dim, 2*node_fea_dim)

        self.softplus_hedge = torch.nn.Softplus()
        self.sigmoid_filter = torch.nn.Sigmoid()
        self.softplus_core = torch.nn.Softplus()
        self.softplus_out = torch.nn.Softplus()

        self.hedge_aggr = aggr.MultiAggregation(
                            aggrs=['mean', 'std','max','min'],
                            mode='attn',
                            mode_kwargs=dict(in_channels=64, out_channels=64, num_heads=4),
                            )
        self.node_aggr = aggr.MultiAggregation(
                            aggrs=['mean', 'std','max','min'],
                            mode='attn',
                            mode_kwargs=dict(in_channels=64, out_channels=64, num_heads=4),
                            )
        #self.hedge_aggr = aggr.SoftmaxAggregation(learn = True)
        #self.node_aggr = aggr.SoftmaxAggregation(learn = True)

        if batch_norm == True:
            self.bn_f = BatchNorm1d(node_fea_dim)
            self.bn_c = BatchNorm1d(node_fea_dim)

            self.bn_o = BatchNorm1d(node_fea_dim)

    def forward(self, x, hyperedge_index, hyperedge_attrs):
        '''
        hyperedge_attrs_tuple:    tuple of torch tensor (of type float) of source and destination hyperedge attributes

                        ([hedge1_feat],...],[[node1_feat],[node2_feat],...)
                        (dim(hedge_feat_dim,num_hedges),dim(num_nodes, node_feat_dim))

        hedge_index:    torch tensor (of type long) of
                        hyperedge indices (as in HypergraphConv)

                        [[node_indxs,...],[hyperedge_indxs,...]]
                        dim([2,num nodes in all hedges])


        '''

        '''
        The goal is to generalize the CGConv gated convolution structure to hyperedges. The
        primary problem with such a generalization is the variable number of nodes contained
        in each hyperedge (hedge). I propose we simply aggregate the nodes contained within
        each hedge to complete the message, and then concatenate that with the hyperedge feature
        to form the message.

        Below, the node attributes are first placed in order with their hyperedge_indices
        and then aggregated according to their hyperedges to form a component of the message corresponding to
        each hyperedge
        '''
        hedge_attr = hyperedge_attrs
        num_nodes = x.shape[0]
        num_hedges = hedge_attr.shape[0]
        hedge_index_xs = x[hyperedge_index[1].int()]
        hedge_index_xs = self.hedge_aggr(hedge_index_xs, hyperedge_index[0], dim_size = num_hedges)

        '''
        To finish forming the message, I concatenate these aggregated neighborhoods with their
        corresponding hedge features.
        '''

        message_holder = torch.cat([hedge_index_xs, hedge_attr], dim = 1)
        '''
        We then can aggregate the messages and add to node features after some activation
        functions and linear layers.
        '''
        x_i = x[hyperedge_index[1]]  # Target node features
        x_j = message_holder[hyperedge_index[0]]  # Source node features
        z = torch.cat([x_i,x_j], dim=-1)  # Form reverse messages (for origin node messages)
        z = self.lin_f2(z)
        z_f, z_c = z.chunk(2, dim = -1)
        if self.batch_norm == True:
            z_f = self.bn_f(z_f)
            z_c = self.bn_c(z_c)
        out = self.sigmoid_filter(z_f)*self.softplus_core(z_c) # Apply CGConv like structure
        out = self.node_aggr(out, hyperedge_index[1], dim_size = num_nodes) #aggregate according to node

        if self.batch_norm == True:
            out = self.bn_o(out)

        out = self.softplus_out(out + x)

        return out

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalHypergraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, n_hconv=1, 
                 h_fea_len=128, n_h=1, motifs = True,
                 triplets = True, classification=False, bonds = True):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalHypergraphConvNet, self).__init__()
        self.motifs = motifs
        self.bonds = bonds
        self.triplets = triplets
        motif_fea_dim = 59
        triplet_fea_dim = 41
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        state_string = f'{n_conv} '
        if self.bonds == True:
            state_string += 'bond '
        if self.triplets == True:
            state_string += 'triplet '
        if self.motifs == True:
            state_string += 'motif '
        print(f'Using {state_string} layers:')
        self.convs = []
        self.conv_labels = []
        for conv in range(n_conv):
            if self.bonds == True:
                self.convs.append(ConvLayer(atom_fea_len=atom_fea_len,nbr_fea_len=nbr_fea_len))
                self.conv_labels.append('b')
            if self.triplets == True:
                self.convs.append(CHGConv(atom_fea_len, triplet_fea_dim)) 
                self.conv_labels.append('t')
            if self.motifs == True:
                self.convs.append(CHGConv(atom_fea_len, motif_fea_dim)) 
                self.conv_labels.append('m')
        print(self.conv_labels)
        self.convs = nn.ModuleList(self.convs)
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, motif_idx, motif_fea, triplet_idx, triplet_fea, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        for conv_func, label in zip(self.convs, self.conv_labels):
            if label == 'b':
                atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
            if label == 't':
                atom_fea = conv_func(atom_fea, triplet_fea, triplet_idx)
            if label == 'm':
                atom_fea = conv_func(atom_fea, motif_fea, motif_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
