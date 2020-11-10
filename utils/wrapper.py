import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm.auto import tqdm
bar_format = '{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

from core.model import deepTarget
from core.dataset import TrainDataset, Dataset
from torch.utils.data import DataLoader
from utils.sequence import postprocess_result


def train_model(mirna_fasta_file, mrna_fasta_file, train_file, model=None, cts_size=30, seed_match='offset-9-mer-m7', level='gene', batch_size=32, epochs=10, save_file=None, device='cpu'):
    if not isinstance(model, deepTarget):
        raise ValueError("'model' expected <nn.Module 'deepTarget'>, got {}".format(type(model)))
    
    print("\n[TRAIN] {}".format(model.name))
    
    if train_file.split('/')[-1] == 'train_set.csv':
        train_set = TrainDataset(train_file)
    else:
        train_set = Dataset(mirna_fasta_file, mrna_fasta_file, train_file, seed_match=seed_match, header=True, train=True)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    
    class_weight = torch.Tensor(compute_class_weight('balanced', classes=np.unique(train_set.labels), y=train_set.labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.Adam(model.parameters())
    
    model = model.to(device)
    for epoch in range(epochs):
        epoch_loss, corrects = 0, 0

        with tqdm(train_loader, desc="Epoch {}/{}".format(epoch+1, epochs), bar_format=bar_format) as tqdm_loader:
            for i, ((mirna, mrna), label) in enumerate(tqdm_loader):
                mirna, mrna, label = mirna.to(device, dtype=torch.float), mrna.to(device, dtype=torch.float), label.to(device)
                
                outputs = model(mirna, mrna)
                loss = criterion(outputs, label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * outputs.size(0)
                corrects += (torch.max(outputs, 1)[1] == label).sum().item()
                
                if (i+1) == len(train_loader):
                    tqdm_loader.set_postfix(dict(loss=(epoch_loss/len(train_set)), acc=(corrects/len(train_set))))
                else:
                    tqdm_loader.set_postfix(loss=loss.item())
    
    if save_file is None:
        time = datetime.now()
        save_file = "{}.pt".format(time.strftime('%Y%m%d_%H%M%S_weights'))
    torch.save(model.state_dict(), save_file)
    
    
def predict_result(mirna_fasta_file, mrna_fasta_file, query_file, model=None, weight_file=None, seed_match='offset-9-mer-m7', level='gene', output_file=None, device='cpu'):
    if not isinstance(model, deepTarget):
        raise ValueError("'model' expected <nn.Module 'deepTarget'>, got {}".format(type(model)))
    
    if not weight_file.endswith('.pt'):
        raise ValueError("'weight_file' expected '*.pt', got {}".format(weight_file))
    
    model.load_state_dict(torch.load(weight_file))
    
    test_set = Dataset(mirna_fasta_file, mrna_fasta_file, query_file, seed_match=seed_match, header=True, train=False)
    
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        
        mirna = torch.from_numpy(test_set.mirna).to(device, dtype=torch.float)
        mrna = torch.from_numpy(test_set.mrna).to(device, dtype=torch.float)
        label = torch.from_numpy(test_set.labels).to(device)
        
        outputs = model(mirna, mrna)
        _, predicts = torch.max(outputs.data, 1)
        probabilities = F.softmax(outputs, dim=1)
        
        y_probs = probabilities.cpu().numpy()[:, 1]
        y_predicts = predicts.cpu().numpy()
        y_truth = label.cpu().numpy()
        
        if output_file is None:
            time = datetime.now()
            output_file = "{}.csv".format(time.strftime('%Y%m%d_%H%M%S_results'))
        results = postprocess_result(test_set.dataset, y_probs, y_predicts,
                                     seed_match=seed_match, level=level, output_file=output_file)
        
        print(results)
