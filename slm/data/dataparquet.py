from llama.generation import sample_top_p
from torchsummary import summary
from model import ModelArgs, Transformer
import torch
import torch.nn.functional as F
from tokenizer import Tokenizer
import re
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import subprocess as sp
import os
import random
import pandas as pd
from torch.utils.data import Dataset


class ParquetDataset(Dataset):
    def __init__(self, files, text_col='text',shuffle=True):

        dfs = []
        print('Preparing data.....')
    
        for filename in files:
            dfs.append(pd.read_parquet(filename))

        df = pd.concat(dfs)
        self.data = df['text'].sample(frac = 1).to_list()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]


def get_sents_gen(corpus_filename,bsz=32,max_sent_len=10):
    with open(corpus_filename) as f:
        lines = f.readlines()
    text = ' '.join(lines)
    len_text = len(text)
    res = re.finditer(r"\s", text)
    word_indices = []
    for obj in list(res):
        word_indices.append(obj.span()[1])
    word_indices = np.array(word_indices)
    len_indices = len(word_indices)
    shuffled_indices = np.arange(0,len_indices)
    np.random.shuffle(shuffled_indices)
    
    
    for i in range(0,len_indices,bsz):
        j = min(i+bsz,len_indices)
        start_indices = shuffled_indices[i:j]
        batch = []
        for index in start_indices:
            batch.append(text[word_indices[index]:word_indices[min(index+max_sent_len,len_indices-1)]])
        
        yield batch


def get_sents_gen_dir(directory,bsz=32,max_sent_len=10):
    files = os.listdir(directory)
    num_files = len(files)
    f_indices = list(range(num_files))
    random.shuffle(f_indices)
    for i in range(0,num_files,3):
        indices=f_indices[i:min(i+3,num_files)]
        text = ''
        for j in indices:
            df = pd.read_parquet(os.path.join(directory,files[f_indices[j]]))
            text = '\n\n'.join(df['text'].to_list())

        
        len_text = len(text)
        res = re.finditer(r"\s", text)
        word_indices = []
        for obj in list(res):
            word_indices.append(obj.span()[1])
        word_indices = np.array(word_indices)
        len_indices = len(word_indices)
        shuffled_indices = np.arange(0,len_indices)
        np.random.shuffle(shuffled_indices)
        
        
        for i in range(0,len_indices,bsz):
            j = min(i+bsz,len_indices)
            start_indices = shuffled_indices[i:j]
            batch = []
            for index in start_indices:
                batch.append(text[word_indices[index]:word_indices[min(index+max_sent_len,len_indices-1)]])
            
            yield batch
            
def get_sents_from_parquets(directory,bsz=10,max_sent_len=10):

    all_indices = []
    dfs = []

    print('Preparing data.....')
    
    for filename in ['data/0.parquet','data/1.parquet','data/2.parquet','data/3.parquet']:
        dfs.append(pd.read_parquet(filename))

    for filename in ['data/dolly/1.parquet','data/squadv2/1.parquet']:
        dfs.append(pd.read_parquet(filename,columns=['text']))
    

    df = pd.concat(dfs)
    len_df = len(df)
    indices = list(range(len_df))
    np.random.shuffle(indices)
    for i in range(0,len_df,bsz):
        yield (df['text'].iloc[indices[i:i+bsz]]).to_list()


