##  /apps/pytorch/2.0.1/bin/python
## /orange/h.azad/s.saini/.env/bin/activate



from slm.models.llama import ModelArgs
import torch
import torch
from transformers import AutoTokenizer
from slm.models.lm import LanguageModel
from slm.data.dataparquet import ParquetDataset
from torch.utils.data import DataLoader
import os
import torch.distributed as dist
import argparse

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import functools


# import deepspeed

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    
    bsz = 64

    dset = ParquetDataset(['data/0.parquet','data/1.parquet','data/2.parquet','data/3.parquet',
                           'data/dolly/1.parquet','data/squadv2/1.parquet'])
    train_loader = DataLoader(dset, batch_size=bsz, shuffle=True)
    sampler = DistributedSampler(dset, rank=rank, num_replicas=world_size, shuffle=True)

    train_kwargs = {'batch_size': bsz, 'sampler': sampler}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dset,**train_kwargs)
    torch.cuda.set_device(rank)


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model_args = ModelArgs()
    model_args.dim= 512
    model_args.n_layers= 8
    model_args.n_heads= 8
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
    # print(args.batch_size)
    lm = LanguageModel(model_args,tokenizer,bsz=bsz,max_seq_len=1024,device=rank,device_index=5)
    lm.fsdp()
    # print(lm.model)
    init_start_event.record()
    print('start')
    lm.train(train_loader,sampler=sampler,epochs=1)
    init_end_event.record()

    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    # for epoch in range(1, args.epochs + 1):
    #     train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
    #     test(model, rank, world_size, test_loader)
    #     scheduler.step()

   
    torch.cuda.synchronize()
    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        # print(f"{model}")

    # if args.save_model:
    #     # use a barrier to make sure training is done on all ranks
    #     dist.barrier()
    #     states = model.state_dict()
    #     if rank == 0:
    #         torch.save(states, "mnist_cnn.pt")

    cleanup()



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)