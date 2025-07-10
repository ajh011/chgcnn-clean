from __future__ import print_function, division

import csv
import functools
import json
import os
import os.path as osp
import random
import warnings
import itertools

from torch_geometric.data import HeteroData
import jsonpickle

import sklearn.model_selection
import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from pymatgen.analysis.local_env import \
    LocalStructOrderParams, \
    VoronoiNN, \
    CrystalNN, \
    JmolNN, \
    MinimumDistanceNN, \
    MinimumOKeeffeNN, \
    EconNN, \
    BrunnerNN_relative, \
    MinimumVIRENN

from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.local_env import LocalStructOrderParams

from matminer.featurizers.site.fingerprint import OPSiteFingerprint, ChemEnvSiteFingerprint


def get_nested_folds(dataset, collate_fn=default_collate, num_folds = 5, num_workers = 1,
                       batch_size=64, val_ratio = 0.2, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    num_workers: int
    pin_memory: bool

    Returns
    -------
    [(train_loader, val_loader), (train_loader, val_loader), ...]
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    """
    total_size = len(dataset)
    train_size = int(total_size * (num_folds-1)/(num_folds))
    val_size = int(val_ratio*train_size)
    indices = list(range(total_size))
    kf = sklearn.model_selection.KFold(n_splits=num_folds)
    folds = []
    for train_indices, test_indices in kf.split(dataset):

        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]
        train_dataset = torch.utils.data.dataset.Subset(dataset,train_indices)
        val_dataset = torch.utils.data.dataset.Subset(dataset,val_indices)
        test_dataset = torch.utils.data.dataset.Subset(dataset,test_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
        folds.append((train_loader, val_loader, test_loader))
    return folds






def get_k_folds(dataset, collate_fn=default_collate, num_folds = 5, num_workers = 1,
                       batch_size=64, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    num_workers: int
    pin_memory: bool

    Returns
    -------
    [(train_loader, val_loader), (train_loader, val_loader), ...]
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    """
    total_size = len(dataset)
    indices = list(range(total_size))
    seg = int(total_size/num_folds)
    folds = []
    for i in range(num_folds):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size

        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))

        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))

        train_dataset = torch.utils.data.dataset.Subset(dataset,train_indices)
        val_dataset = torch.utils.data.dataset.Subset(dataset,val_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
        folds.append((train_loader, val_loader))
    return folds




def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    batch_motif_fea, batch_motif_idx = [], [[],[]]
    batch_triplet_fea, batch_triplet_idx = [], [[],[]]
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    base_motif_idx = 0
    base_triplet_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx, motif_idx, motif_fea, triplet_idx, triplet_fea), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        
        n_m = motif_fea.shape[0]
        batch_motif_fea.append(motif_fea)
        batch_motif_idx[0].append(motif_idx[0]+base_motif_idx)
        batch_motif_idx[1].append(motif_idx[1]+base_idx)

        n_t = triplet_fea.shape[0]
        batch_triplet_fea.append(triplet_fea)
        batch_triplet_idx[0].append(triplet_idx[0]+base_triplet_idx)
        batch_triplet_idx[1].append(triplet_idx[1]+base_idx)
        

        base_idx += n_i
        base_motif_idx += n_m
        base_triplet_idx += n_t

    batch_motif_idx[0] = torch.cat(batch_motif_idx[0], dim=0)
    batch_motif_idx[1] = torch.cat(batch_motif_idx[1], dim=0)
    batch_triplet_idx[0] = torch.cat(batch_triplet_idx[0], dim=0)
    batch_triplet_idx[1] = torch.cat(batch_triplet_idx[1], dim=0)
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(batch_motif_fea, dim=0),
            torch.stack([batch_motif_idx[0],batch_motif_idx[1]]),
            torch.cat(batch_triplet_fea, dim=0),
            torch.stack([batch_triplet_idx[0],batch_triplet_idx[1]]),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids

class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123, processed_ids_csv= 'processed_ids.csv', target_name = 'y'):
        self.target_name = target_name
        self.processed_dir = root_dir
        self.processed_ids_csv = processed_ids_csv
        self.max_num_nbr, self.radius = max_num_nbr, radius
        print(f'Retrieving data from {root_dir}')
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_file = os.path.join(self.processed_dir, processed_ids_csv)
        print(f'Checking for {id_file}')
        with open(id_file) as f:
            reader = csv.reader(f)
            self.ids = [row[0] for row in reader]
        random.seed(random_seed)
        random.shuffle(self.ids)

    def __len__(self):
        return len(self.ids)


    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        if isinstance(idx,str):
            mp_id = idx
        else:
            mp_id = self.ids[idx]
        data = torch.load(osp.join(self.processed_dir, f'data_{mp_id}'), weights_only=False)
        atom_fea = data.atom_fea
        nbr_fea = data.nbr_fea
        triplet_fea = data.triplet_fea
        motif_fea = data.motif_fea
        nbr_fea_idx = data.nbr_fea_idx
        triplet_idx = data.triplet_idx
        motif_idx = data.motif_idx
        target = torch.tensor(float(data[self.target_name]))
        cif_id = data.cif_id

        return (atom_fea, nbr_fea, nbr_fea_idx, motif_idx, motif_fea, triplet_idx, triplet_fea), target, cif_id

### Define general hyperedge type class
class HyperedgeType(object):
    def __init__(self, generate_features = True):
        self.hyperedge_index = [[],[]]
        self.hyperedge_attrs = []
        self.neighborsets = []
        self.generate_features = generate_features

### Define triplets hyperedge type for generation
class Triplets(HyperedgeType):
    def __init__(self, dir_or_nbrset=None, generate_features = True):
        super().__init__(generate_features = generate_features)
        self.name = 'triplet'
        self.order = 3

        if dir_or_nbrset != None:
            self.generate(dir_or_nbrset)

    def generate(self, dir_or_nbrset, nn_strat = 'voro', gauss_dim = 40, radius = 8):
        if type(dir_or_nbrset) == str:
            struc = CifParser(dir_or_nbrset).get_structures()[0]
            nbr_list, nn_strat = get_nbrlist(struc, nn_strategy = nn_strat, max_nn=12)
        else:
            nbr_list = dir_or_nbrset
        if gauss_dim != 1:
            ge = gaussian_expansion(dmin = -1, dmax = 1, steps = gauss_dim)

        triplet_index = 0
        for cnt_idx, neighborset in nbr_list:
                for i in itertools.combinations(neighborset, 2):
                    (pair_1_idx, offset_1, distance_1), (pair_2_idx, offset_2, distance_2) = i

                    if self.generate_features == True:
                        offset_1 = np.stack(offset_1)
                        offset_2 = np.stack(offset_2)
                        cos_angle = (offset_1 * offset_2).sum(-1) / (np.linalg.norm(offset_1, axis=-1) * np.linalg.norm(offset_2, axis=-1))

                        #Stop-gap to fix nans from zero displacement vectors
                        cos_angle = np.nan_to_num(cos_angle, nan=1)

                        if gauss_dim != 1:
                            cos_angle = ge.expand(cos_angle)

                        self.hyperedge_attrs.append(cos_angle)

                    self.hyperedge_index[0].append(pair_1_idx)
                    self.hyperedge_index[1].append(triplet_index)

                    self.hyperedge_index[0].append(pair_2_idx)
                    self.hyperedge_index[1].append(triplet_index)

                    self.hyperedge_index[0].append(cnt_idx)
                    self.hyperedge_index[1].append(triplet_index)

                    self.neighborsets.append([cnt_idx, pair_1_idx, pair_2_idx])

                    triplet_index += 1

### Define motifs hyperedge type for generation
class Motifs(HyperedgeType):
    def __init__(self,  dir_or_nbrset=None, struc=None, generate_features = True):
        super().__init__(generate_features = generate_features)
        self.name = 'motif'
        self.order = 12
        self.struc=struc
        self.all_lsop_types = [ "cn",
                            "sgl_bd",
                            "bent",
                            "tri_plan",
                            "tri_plan_max",
                            "reg_tri",
                            "sq_plan",
                            "sq_plan_max",
                            "pent_plan",
                            "pent_plan_max",
                            "sq",
                            "tet",
                            "tet_max",
                            "tri_pyr",
                            "sq_pyr",
                            "sq_pyr_legacy",
                            "tri_bipyr",
                            "sq_bipyr",
                            "oct",
                            "oct_legacy",
                            "pent_pyr",
                            "hex_pyr",
                            "pent_bipyr",
                            "hex_bipyr",
                            "T",
                            "cuboct",
                            "cuboct_max",
                            "see_saw_rect",
                            "bcc",
                            "q2",
                            "q4",
                            "q6",
                            "oct_max",
                            "hex_plan_max",
                            "sq_face_cap_trig_pris"]
        
        #Removed S:10, S:12, SH:11, CO:11 and H:10 due to errors in package
        self.all_ce_types = ['S:1', 
                             'L:2', 
                             'A:2', 
                             'TL:3', 
                             'TY:3', 
                             'TS:3', 
                             'T:4', 
                             'S:4', 
                             'SY:4', 
                             'SS:4', 
                             'PP:5', 
                             'S:5', 
                             'T:5', 
                             'O:6', 
                             'T:6', 
                             'PP:6', 
                             'PB:7', 
                             'ST:7', 
                             'ET:7', 
                             'FO:7', 
                             'C:8', 
                             'SA:8', 
                             'SBT:8', 
                             'TBT:8', 
                             'DD:8', 
                             'DDPN:8', 
                             'HB:8', 
                             'BO_1:8', 
                             'BO_2:8', 
                             'BO_3:8', 
                             'TC:9', 
                             'TT_1:9', 
                             'TT_2:9', 
                             'TT_3:9', 
                             'HD:9', 
                             'TI:9', 
                             'SMA:9', 
                             'SS:9', 
                             'TO_1:9', 
                             'TO_2:9', 
                             'TO_3:9', 
                             'PP:10', 
                             'PA:10', 
                             'SBSA:10', 
                             'MI:10', 
                             'BS_1:10', 
                             'BS_2:10', 
                             'TBSA:10', 
                             'PCPA:11', 
                             'H:11', 
                             'DI:11', 
                             'I:12', 
                             'PBP:12', 
                             'TT:12', 
                             'C:12', 
                             'AC:12',
                             'SC:12',
                             'HP:12',
                             'HA:12']

        
        if dir_or_nbrset != None:
            self.generate(dir_or_nbrset)

    def generate(self, dir_or_nbrset, nn_strat = 'crys', lsop_types = [], ce_types = []):
        if type(dir_or_nbrset) == str:
            struc = CifParser(dir_or_nbrset).get_structures()[0]
            nbr_list, nn_strat = get_nbrlist(struc, nn_strategy = nn_strat, max_nn=12)
        else: 
            nbr_list = dir_or_nbrset 
            if self.struc == None:
                print('Structure required as input for motif neighbor lists')
            struc = self.struc

        self.nbr_strategy = nn_strat

        neighborhoods = []
        motif_index = 0
        for n, neighborset in nbr_list:
            neigh_idxs = []
            for idx in neighborset:
                neigh_idx = idx[0]
                neigh_idxs.append(neigh_idx)
                self.hyperedge_index[0].append(neigh_idx)
                self.hyperedge_index[1].append(motif_index)
            self.hyperedge_index[0].append(n)
            self.hyperedge_index[1].append(motif_index)
            neighborhoods.append([n, neigh_idxs])
            neigh_idxs.append(n)
            self.neighborsets.append(neigh_idxs)
            motif_index += 1
        if self.generate_features == True and lsop_types == []:
            lsop_types = self.all_lsop_types
        if self.generate_features == True and ce_types == []:
            ce_types = self.all_ce_types

        lgf = LocalGeometryFinder()
        lgf.setup_parameters(
            centering_type="centroid",
            include_central_site_in_centroid=True,
            structure_refinement=lgf.STRUCTURE_REFINEMENT_NONE,
        )

        ##Compute order parameter features
        lsop = LocalStructOrderParams(lsop_types)
        CSM = ChemEnvSiteFingerprint(ce_types, MultiWeightsChemenvStrategy.stats_article_weights_parameters(), lgf)

        lsop_tol = 0.05
        for site, neighs in neighborhoods:
            op_feat = lsop.get_order_parameters(struc, site, indices_neighs = neighs)
            csm_feat = CSM.featurize(struc, site)
            for n,f in enumerate(op_feat):
                if f == None:
                    op_feat[n] = 0
                elif f > 1:
                    op_feat[n] = f
                ##Account for tolerance:
                elif f > lsop_tol:
                    op_feat[n] = f
                else:
                    op_feat[n] = 0
            feat = np.concatenate((op_feat, csm_feat))
            self.hyperedge_attrs.append(feat)


def get_nbrlist(struc, nn_strategy = 'crys', max_nn=12):
    NN = {
        # these methods consider too many neighbors which may lead to unphysical resutls
        'voro': VoronoiNN(tol=0.2),
        'econ': EconNN(),
        'brunner': BrunnerNN_relative(),

        # these two methods will consider motifs center at anions
        'crys': CrystalNN(),
        'jmol': JmolNN(),

        # not sure
        'minokeeffe': MinimumOKeeffeNN(),

        # maybe the best
        'mind': MinimumDistanceNN(),
        'minv': MinimumVIRENN()
    }

    nn = NN[nn_strategy]

    center_idxs = []
    neighbor_idxs = []
    offsets = []
    distances = []

    reformat_nbr_lst = []

    for n in range(len(struc.sites)):
        neigh = []
        neigh = [neighbor for neighbor in nn.get_nn(struc, n)]

        neighbor_reformat=[]
        for neighbor in neigh[:max_nn]:
            neighbor_index = neighbor.index
            offset = struc.frac_coords[neighbor_index] - struc.frac_coords[n] + neighbor.image
            m = struc.lattice.matrix
            offset = offset @ m
            distance = np.linalg.norm(offset)
            center_idxs.append(n)
            neighbor_idxs.append(neighbor_index)
            offsets.append(offset)
            distances.append(distance)

            neighbor_reformat.append((neighbor_index, offset, distance))
        reformat_nbr_lst.append((n,neighbor_reformat))

    return reformat_nbr_lst, nn_strategy
