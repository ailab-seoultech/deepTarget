import torch
import torch.nn as nn
import torch.nn.functional as F
import json


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class HyperParam:
    def __init__(self, filters=None, kernels=None, model_json=None):
        self.dictionary = dict()
        self.name_postfix = str()

        if (filters is not None) and (kernels is not None) and (model_json is None):
            for i, (f, k) in enumerate(zip(filters, kernels)):
                setattr(self, 'f{}'.format(i+1), f)
                setattr(self, 'k{}'.format(i+1), k)
                self.dictionary.update({'f{}'.format(i+1): f, 'k{}'.format(i+1): k})
            self.len = i+1
                
            for key, value in self.dictionary.items():
                self.name_postfix = "{}_{}-{}".format(self.name_postfix, key, value)
        elif model_json is not None:
            self.dictionary = json.loads(model_json)
            for i, (key, value) in enumerate(self.dictionary.items()):
                setattr(self, key, value)
                self.name_postfix = "{}_{}-{}".format(self.name_postfix, key, value)
            self.len = (i+1)//2
    
    def __len__(self):
        return self.len


class deepTarget(nn.Module):
    def __init__(self, hparams=None, hidden_units=30, input_shape=(1, 4, 30), name_prefix="model"):
        super(deepTarget, self).__init__()
        
        if hparams is None:
            filters, kernels = [32, 16, 64, 16], [3, 3, 3, 3]
            hparams = HyperParam(filters, kernels)
        self.name = "{}{}".format(name_prefix, hparams.name_postfix)
        
        if (isinstance(hparams, HyperParam)) and (len(hparams) == 4):
            self.embd1 = nn.Conv1d(4, hparams.f1, kernel_size=hparams.k1, padding=((hparams.k1 - 1) // 2))
            self.conv2 = nn.Conv1d(hparams.f1*2, hparams.f2, kernel_size=hparams.k2)
            self.conv3 = nn.Conv1d(hparams.f2, hparams.f3, kernel_size=hparams.k3)
            self.conv4 = nn.Conv1d(hparams.f3, hparams.f4, kernel_size=hparams.k4)
            
            """ out_features = ((in_length - kernel_size + (2 * padding)) / stride + 1) * out_channels """
            flat_features = self.forward(torch.rand(input_shape), torch.rand(input_shape), flat_check=True)
            self.fc1 = nn.Linear(flat_features, hidden_units)
            self.fc2 = nn.Linear(hidden_units, 2)
        else:
            raise ValueError("not enough hyperparameters")
    
    def forward(self, x_mirna, x_mrna, flat_check=False):
        h_mirna = F.relu(self.embd1(x_mirna))
        h_mrna = F.relu(self.embd1(x_mrna))
        
        h = torch.cat((h_mirna, h_mrna), dim=1)
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        
        h = h.view(h.size(0), -1)
        if flat_check:
            return h.size(1)
        h = self.fc1(h)
        y = self.fc2(h) #y = F.softmax(self.fc2(h), dim=1)
        
        return y
    
    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)