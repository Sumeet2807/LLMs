from llama.generation import sample_top_p
from torchsummary import summary
from slm.models.llama import ModelArgs, Transformer
from slm.models.llama import TransformerBlock
import torch
import torch.nn.functional as F
from tokenizer import Tokenizer
import re
import numpy as np
import subprocess as sp
import os
import random
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from slm.data.dataparquet import get_sents_from_parquets
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import functools
import torch.distributed as dist
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from distutils.version import LooseVersion
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)


prompts = ['context:\nJohn was playing with his toy. He was very happy and smiling. He was tired, so he ate some food. He went to his room after playing.\nprompt:\nWhy did john eat some food?\nanswer:']



def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values




        
class LanguageModel():
    
    def __init__(self,model_args,tokenizer,max_seq_len=10,bsz=1,verbose=True,device='cuda',device_index=5):
        model_args.vocab_size = len(tokenizer)
        model_args.max_seq_len = max_seq_len
        model_args.max_batch_size = bsz
        self.bsz = bsz
        self.max_seq_len = max_seq_len
        self.model = Transformer(model_args)
        self.model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.device_index = device_index
        if verbose:
            summary(self.model,torch.ones((1,10),dtype=torch.long).to(device))

    def fsdp(self):

        bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )

        my_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock,},
        )
        
        bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
        and torch.cuda.nccl.version() >= (2, 10)
        )
        print('bf16 :' + str(bf16_ready))

        if bf16_ready:
            mp_policy = bfSixteen
        else:
            mp_policy = None # defaults to fp32

        self.model = FSDP(self.model,auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=mp_policy,
        #sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device())
    

        # self.model = FSDP(self.model)

    def compute_loss(self,batch,writer,cnt):
        pad_id = 32000
        sent_tokens = [self.tokenizer.encode(x) for x in batch]
        tokens = torch.full((len(batch), self.max_seq_len), pad_id, dtype=torch.long, device=self.device)
        for k, t in enumerate(sent_tokens):
            tokens[k, : min(self.max_seq_len,len(t))] = torch.tensor(t[:min(self.max_seq_len,len(t))], dtype=torch.long, device=self.device)
        X = tokens[:,:-1]
        y = tokens[:,1:]
        input_text_mask = (y != pad_id).to(self.device)
        writer.add_scalar("non pad ratio", torch.sum(input_text_mask.int())/(input_text_mask.shape[0]*input_text_mask.shape[0]), cnt)
        output = self.model(X)    
        # output = output*(input_text_mask.int()[...,None])
        # loss = self.loss_fn(torch.transpose(output,1,-1),y)

        relevant_output = output.flatten(end_dim=-2)[input_text_mask.flatten()]
        relevant_y = y.flatten()[input_text_mask.flatten()]
        loss = self.loss_fn(relevant_output,relevant_y)


        return loss
    
    def train(self,train_loader,test_corpus=None,epochs=20,sampler= None):
        
        self.model.train()
        writer = SummaryWriter()
        
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, betas=(0.9, 0.95), eps=1e-05, weight_decay=0.1)
        # print(self.model.parameters())
        
        cnt = 0
        for i in range(1,epochs+1):
            # print('total epochs - ' + str(i))
            # sent_gen = get_sents_gen_dir(train_corpus,self.bsz,self.max_seq_len)
            # sent_gen = get_sents_from_parquets(train_corpus,self.bsz,self.max_seq_len)


            if sampler:
                sampler.set_epoch(i)
            
            for j, batch in enumerate(train_loader):
                loss = self.compute_loss(batch,writer,cnt)
                loss.backward()#retain_graph=True)
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_val = loss.detach()
                cnt+=1
                if not j%100 and self.device == 0:
                    with torch.no_grad():
                        # if test_corpus is not None:
                        #     test_sent_gen = get_sents_gen(test_corpus,self.bsz,self.max_seq_len)
                        #     losses = []
                        #     for j, batch in enumerate(sent_gen): 
                        #         losses.append(self.compute_loss(batch))
                        #     val_loss = torch.mean(losses)
                        #     writer.add_scalar("Loss/test", val_loss, cnt)
                        print(len(batch),torch.cuda.memory_allocated(device=self.device))
                        print(loss_val)                    
                #         writer.add_scalar("Mem/train", get_gpu_memory()[self.device_index], cnt)
                #         prompt_tokens = [self.tokenizer.encode(x) for x in prompts]
                #         gens = self.tokenizer.batch_decode(self.generate(prompt_tokens,50)[0])
                #         for k,gen in enumerate(gens):
                #             writer.add_text("Gen/train",prompts[k]+'#####'+gen,cnt)
                    
                # # del loss
                # writer.add_scalar("Loss/train", loss_val, cnt)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ):
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        print('generating')
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = 32000
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos, use_att_cache= True)
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == 2
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if 2 in toks:
                eos_idx = toks.index(2)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)


