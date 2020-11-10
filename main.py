#!/usr/bin/env python

import argparse
from datetime import datetime

import torch
from core.model import deepTarget
from utils.wrapper import train_model, predict_result


def main(configs):
    mirna_fasta_file = configs.MIRNA_FASTA_FILE if configs.MIRNA_FASTA_FILE is not None else 'data/mirna.fasta'
    mrna_fasta_file  = configs.MRNA_FASTA_FILE if configs.MRNA_FASTA_FILE is not None else 'data/mrna.fasta'
    seed_match       = configs.SEED_MATCH if configs.SEED_MATCH is not None else 'offset-9-mer-m7'
    level            = configs.LEVEL if configs.LEVEL is not None else 'gene'
    train_file       = configs.TRAIN_FILE
    save_file        = configs.SAVE_FILE
    query_file       = configs.QUERY_FILE
    weight_file      = configs.WEIGHT_FILE if configs.WEIGHT_FILE is not None else 'model/weights.pt'
    output_file      = configs.OUTPUT_FILE
    batch_size       = configs.BATCH_SIZE if configs.BATCH_SIZE is not None else 32
    epochs           = configs.EPOCHS if configs.EPOCHS is not None else 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if configs.MODE == 'train':
        if train_file is None:
            raise ValueError("'--train_file' expected '*.csv', got '{}'".format(configs.TRAIN_FILE))
            
        start_time = datetime.now()
        print("\n[START] {}".format(start_time.strftime('%Y-%m-%d @ %H:%M:%S')))
        
        model = deepTarget()
        train_model(mirna_fasta_file, mrna_fasta_file, train_file,
                    model=model,
                    seed_match=seed_match, level=level,
                    batch_size=batch_size, epochs=epochs,
                    save_file=save_file, device=device)
        
        finish_time = datetime.now()
        print("\n[FINISH] {} (user time: {})\n".format(finish_time.now().strftime('%Y-%m-%d @ %H:%M:%S'), (finish_time - start_time)))
    elif configs.MODE == 'predict':
        if query_file is None:
            raise ValueError("'--query_file' expected '*.csv', got '{}'".format(configs.QUERY_FILE))
            
        start_time = datetime.now()
        print("\n[START] {}".format(start_time.strftime('%Y-%m-%d @ %H:%M:%S')))
        
        model = deepTarget()
        results = predict_result(mirna_fasta_file, mrna_fasta_file, query_file,
                                 model=model, weight_file=weight_file,
                                 seed_match=seed_match, level=level,
                                 output_file=output_file, device=device)
        
        finish_time = datetime.now()
        print("\n[FINISH] {} (user time: {})\n".format(finish_time.now().strftime('%Y-%m-%d @ %H:%M:%S'), (finish_time - start_time)))
    else:
        raise ValueError("'--mode' expected 'train', or 'predict', got '{}'".format(configs.MODE))


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', dest='MODE', type=str, required=True,
                        help="run mode: [train|predict]")
    
    parser.add_argument('--mirna_file', dest='MIRNA_FASTA_FILE', type=str,
                        help="miRNA fasta file (default: data/miRNA.fasta)")
    parser.add_argument('--mrna_file', dest='MRNA_FASTA_FILE', type=str,
                        help="mRNA fasta file (default: data/mRNA.fasta)")
    parser.add_argument('--seed_match', dest='SEED_MATCH', type=str,
                        help="seed match type: [offset-9-mer-m7|10-mer-m7|10-mer-m6] (default: offset-9-mer-m7)")
    parser.add_argument('--level', dest='LEVEL', type=str,
                        help="prediction level: [gene|site] (default: gene)")
    
    parser.add_argument('--train_file', dest='TRAIN_FILE', type=str,
                        help="training file to be used in 'train' mode (sample: data/train_set.csv)")
    parser.add_argument('--save_file', dest='SAVE_FILE', type=str,
                        help="state_dict file to be saved in 'train' mode (default: yyyyMMdd_HHmmss_weights.pt)")
    parser.add_argument('--query_file', dest='QUERY_FILE', type=str,
                        help="query file to be queried in 'predict' mode (sample: templates/query_set.csv)")
    parser.add_argument('--weight_file', dest='WEIGHT_FILE', type=str,
                        help="state_dict file to be loaded in 'predict' mode (default: model/weights.pt)")
    parser.add_argument('--output_file', dest='OUTPUT_FILE', type=str,
                        help="output file to be saved in 'predict' mode (default: yyyyMMdd_HHmmss_results.csv)")
    
    parser.add_argument('--batch_size', dest='BATCH_SIZE', type=int,
                        help="batch size to be used in 'train' mode (default: 32)")
    parser.add_argument('--epochs', dest='EPOCHS', type=int,
                        help="epochs to be used in 'train' mode (default: 10)")
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    configs = parse_arguments()
    main(configs)
