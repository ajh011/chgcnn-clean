import numpy as np
import math

import os
import os.path as osp
import csv
import jsonpickle
import itertools
import time

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure


import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import math

try:
    from .hypergraph.hyperedges.bonds import Bonds
    from .hypergraph.hyperedges.triplets import Triplets
    from .hypergraph.hyperedges.motifs import Motifs
    from .hypergraph.hyperedges.unit_cell import UnitCell

    from .hypergraph.neighbor_list import get_nbrlist


    from .hypergraph.hypergraph import Crystal_Hypergraph

except:
    from hypergraph.hyperedges.bonds import Bonds
    from hypergraph.hyperedges.triplets import Triplets
    from hypergraph.hyperedges.motifs import Motifs
    from hypergraph.hyperedges.unit_cell import UnitCell

    from hypergraph.neighbor_list import get_nbrlist

    from hypergraph.hypergraph import Crystal_Hypergraph


##Build data structure in form of (vanilla) pytorch dataset (not PytorchGeometric!)
class MatbenchHypergraphDataset(Dataset):
    def __init__(self, dataframe, radius=8.0, n_nbr=20, motif_feat = 'csm'):
        super().__init__()

        self.radius = radius
        self.n_nbr = n_nbr
        self.motif_feat = motif_feat

        ##dataset_ratio for testing

        self.mbids = dataframe.index
        self.target_name = str(dataframe.columns[-1])
        self.targets = dataframe[self.target_name]
        self.structures = dataframe['structure']
        self.dataframe = dataframe

    def __len__(self):
        return int(len(list(self.mbids)))
    
    def __getitem__(self, index, report = True):
        mbid = self.mbids[index]
        target = self.targets[index]
        struc = self.structures[index]
        if report:
            start = time.time()
        hgraph = Crystal_Hypergraph(struc, mp_id=mbid, target_dict={f'{self.target_name}':torch.tensor(float(target))},
                                     motif_feat = self.motif_feat, n_nbr = self.n_nbr, radius = self.radius)
        #forget structure for 'efficiency'
        hgraph.struc = None
        if report:
            duration = time.time()-start
            print(f'Processed {mbid} in {round(duration,5)} sec')
        return {
            'hgraph': hgraph,
            'mp_id' : mbid
            }
    
class InMemoryCrystalHypergraphDataset(Dataset):
    def __init__(self, data_dir, csv_dir = '', motif_feat=['csm']):
        super().__init__()
        
        if type(motif_feat) == list:
            for mf in motif_feat:
                assert mf in set(['csm','lsop'])
        else: 
            assert motif_feat in set(['lsop','csm','none',None])
        
        self.motif_feat = motif_feat
        print(f'Loading motif feats: {motif_feat}')
        if csv_dir == '':
            csv_dir = data_dir

        self.csv_dir = csv_dir
        self.data_dir = data_dir

        with open(osp.join(csv_dir, 'processed_ids.csv')) as id_file:
            ids_csv = csv.reader(id_file)
            ids = [mp_id[0] for mp_id in ids_csv]
            self.ids = ids
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        mp_id = self.ids[index]
        file_dir = osp.join(self.data_dir, mp_id + '_hg.json')
        with open(file_dir,'r') as storage:
            data_read = storage.read()
            data_read = jsonpickle.decode(data_read)
        data_dict = dict(data_read)
        data = HeteroData(data_dict)
        num_motifs = list(data['motif'].hyperedge_attrs.shape)[0]
        if str(self.motif_feat) == 'csm':
            data['motif'].hyperedge_attrs = data['motif'].hyperedge_attrs[:,35:] 
        elif str(self.motif_feat) == 'lsop':
            data['motif'].hyperedge_attrs = data['motif'].hyperedge_attrs[:,:35] 
        elif self.motif_feat == None:
            data['motif'].hyperedge_attrs = torch.zeros([num_motifs, 0])
        #_global_store isn't being read correctly... here's a hack:
        data.y = data_dict['_global_store']['target']
        num_nodes = list(data['atom'].hyperedge_attrs.shape)[0]
        data['atom'].num_nodes = torch.tensor(num_nodes).long()
        data['atom'].batch = torch.tensor([0 for i in range(num_nodes)])

        return data
    

def process_data(idx):
    mp_id = dataframe.index[idx]
    if osp.exists(f'{processed_data_dir}/{mp_id}_hg.json'):
        print(f'Hypergraph for {mp_id} already found, skipping...')
    else:
        d = dataset[idx]
        data = d['hgraph']
        data_dict = data.to_dict()
        data_list = list(data_dict.items())
        with open(f'{processed_data_dir}/{mp_id}_hg.json','w') as storage:
            json_list = jsonpickle.encode(data_list)
            storage.write(json_list)



def run_process(N=None, processes=10):
    if N is None:
        N = len(dataset)

    pool = Pool(processes)

    for _ in tqdm(pool.imap_unordered(process_data, range(N)), total=N):
        pass


if __name__ == '__main__':
    from matbench.bench import MatbenchBenchmark
    from pathlib import Path
    mb = MatbenchBenchmark(autoload=False, subset= ['matbench_dielectric',
                                                    'matbench_log_gvrh',
                                                    'matbench_log_kvrh',
                                                    'matbench_mp_e_form',
                                                    'matbench_mp_gap',
                                                    'matbench_mp_is_metal',
                                                    'matbench_perovskites',
                                                    'matbench_phonons']) 
    for task in mb.tasks:
        task.load()
        dataframe = task.df
        dataset = MatbenchHypergraphDataset(dataframe)
        
        ##Directory for processed data (relative generate file)
        processed_data_dir = f'dataset_{str(task.df.columns[-1])}'
        if not osp.exists(processed_data_dir):
            Path(processed_data_dir).mkdir(parents=True, exist_ok=True)

        run_process()
