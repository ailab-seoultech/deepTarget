import torch
import numpy as np
import pandas as pd
from utils.sequence import make_input_pair, preprocess_data


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class Dataset(torch.utils.data.Dataset):
    def __init__(self, mirna_fasta_file, mrna_fasta_file, ground_truth_file, cts_size=30, seed_match='offset-9-mer-m7', header=True, train=True):
        self.dataset = make_input_pair(mirna_fasta_file, mrna_fasta_file, ground_truth_file, cts_size=cts_size, seed_match=seed_match, header=header, train=train)
        self.mirna, self.mrna = preprocess_data(self.dataset['query_seqs'], self.dataset['target_seqs'])
        self.labels = np.asarray(self.dataset['labels']).reshape(-1,)
        
        self.mirna = self.mirna.transpose((0, 2, 1))
        self.mrna = self.mrna.transpose((0, 2, 1))
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        mirna = self.mirna[index]
        mrna = self.mrna[index]
        label = self.labels[index]
        
        return (mirna, mrna), label
        
    def __len__(self):
        return len(self.labels)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, ground_truth_file, cts_size=30):
        self.records = pd.read_csv(ground_truth_file, header=0, sep='\t')
        mirna_seqs = self.records['MIRNA_SEQ'].values.tolist()
        mrna_seqs = self.records['MRNA_SEQ'].values.tolist()
        self.mirna, self.mrna = preprocess_data(mirna_seqs, mrna_seqs, cts_size=cts_size)
        self.labels = self.records['LABEL'].values.astype(int)
        
        self.mirna = self.mirna.transpose((0, 2, 1))
        self.mrna = self.mrna.transpose((0, 2, 1))
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        mirna = self.mirna[index]
        mrna = self.mrna[index]
        label = self.labels[index]
        
        return (mirna, mrna), label
        
    def __len__(self):
        return len(self.labels)
