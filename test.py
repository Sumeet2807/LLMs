##  /apps/pytorch/2.0.1/bin/python
## /orange/h.azad/s.saini/.env/bin/activate


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
from transformers import AutoTokenizer, AutoModelForCausalLM
from slm.models.lm import LanguageModel
import deepspeed


model_args = ModelArgs()
model_args.dim= 8
model_args.n_layers= 2
model_args.n_heads= 2
model_args.n_kv_heads = None
model_args.vocab_size= None  # defined later by tokenizer
model_args.multiple_of = 32  # make SwiGLU hidden layer size multiple of large power of 2
model_args.ffn_dim_multiplier = None
model_args.norm_eps= 1e-5
model_args.max_batch_size= None
model_args.max_seq_len= None

device = 'cuda'
torch.autograd.set_detect_anomaly(True)
# tokenizer = Tokenizer('data/vocab/tinystories28000.model')

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.add_special_tokens({"pad_token":"<pad>"})

lm = LanguageModel(model_args,tokenizer,bsz=10,max_seq_len=10,device_index=5,verbose=False)
lm.train('data/tinystories_corpus/',epochs=20)

# LM_train(model_args,tokenizer,'data/alice_in_wonderland.txt',max_sent_len=128,bsz=32,epochs=1)

# model_engine, optimizer, _, _ = deepspeed.initialize(model=lm.model,
#                                                      model_parameters=lm.model.parameters())