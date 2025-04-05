import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from .sioconvrs import SioConvRSLayer
from neural_memory import ChunkwiseNeuralMemory

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, device=None):
        super().__init__()
        self.normalized_shape = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, device=device))
        else:
            self.weight = None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

class FFNSwiGLU(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: float):
        super().__init__()
        self.fc = nn.Linear(dim, dim_ff_hidden)
        self.fc_act = nn.Linear(dim, dim_ff_hidden)
        self.fc_out = nn.Linear(dim_ff_hidden, dim)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.fc(x) * self.act(self.fc_act(x))
        x = self.fc_out(x)
        return x

class SioConvMemoryBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, num_head: int, base_lr: float, base_weight_decay: float, chunk_size: int, dropout: float):
        super().__init__()
        self.sioconv = SioConvRSLayer(dim)
        self.memory = ChunkwiseNeuralMemory(dim, dim_ff_hidden, num_head, base_lr, base_weight_decay, chunk_size)
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden)
        self.norm_sioconv = RMSNorm(dim)
        self.norm_memory = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.norm_sioconv(x)
        x = self.sioconv(x)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.norm_memory(x)
        x = self.memory(x)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.sioconv.reset_hidden()
        self.memory.reset_hidden()

    def set_is_refresh(self, is_refresh):
        self.sioconv.set_is_refresh(is_refresh)
        self.memory.set_is_refresh(is_refresh)

    def get_hidden(self):
        return {
            'sioconv': self.sioconv.get_hidden(),
            'memory': self.memory.get_hidden()
        }

    def set_hidden(self, hidden):
        if hidden is None:
            self.sioconv.set_hidden(None)
            self.memory.set_hidden(None)
        else:
            self.sioconv.set_hidden(hidden['sioconv'])
            self.memory.set_hidden(hidden['memory'])

class SioConvMemory(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        dim_ff_hidden: int,
        num_head: int,
        base_lr: float,
        base_weight_decay: float,
        dropout: float,
        vocab_size: int,
        chunk_size: int,
        devices,
        out_only_device: bool=True,
    ):
        super().__init__()
        self.devices = devices
        self.vocab_size = vocab_size
        self.token_in = nn.Embedding(vocab_size, dim, device=devices[0], max_norm=1)
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1])
        self.block_list = nn.ModuleList([SioConvMemoryBlock(dim, dim_ff_hidden, num_head, base_lr, base_weight_decay, chunk_size, dropout) for _ in range(depth)])
        self.norm_last = RMSNorm(dim, device=devices[-1])

        self.num_parameters_token_in = sum(p.numel() for p in self.token_in.parameters())
        self.num_parameters_per_block = sum(p.numel() for p in self.block_list[0].parameters())
        self.num_parameters_norm_last = sum(p.numel() for p in self.norm_last.parameters())
        self.num_parameters_token_out = sum(p.numel() for p in self.token_out.parameters())
        self.num_parameters = (self.num_parameters_per_block * depth) + self.num_parameters_norm_last + (self.num_parameters_token_in + self.num_parameters_token_out)
        self.out_only_device = out_only_device

        for i, block in enumerate(self.block_list):
            self.block_list[i] = block.to(devices[self.device_index(i)])

    def device_index(self, i):
        num_out_params = self.num_parameters_norm_last + self.num_parameters_token_out
        return (int)(
            (
                (len(self.devices)-(1 if self.out_only_device else 0)) *
                (i * self.num_parameters_per_block + self.num_parameters_token_in)
            ) /
            (self.num_parameters - (num_out_params if self.out_only_device else 0))
        )

    def forward(self, x):
        x = self.token_in(x)
        for i, block in enumerate(self.block_list):
            x = x.to(self.devices[self.device_index(i)])
            x = block(x)
        x = x.to(self.devices[-1])
        x = self.norm_last(x)
        x = self.token_out(x)
        return x 

    def reset_hidden(self):
        for block in self.block_list:
            block.reset_hidden()

    def set_is_refresh(self, is_refresh):
        for block in self.block_list:
            block.set_is_refresh(is_refresh)

    def get_hidden(self):
        hidden_list = []
        for block in self.block_list:
            hidden = block.get_hidden()
            if hidden is None:
                return None
            hidden_list.append(hidden.cpu())
        return hidden_list

    def set_hidden(self, hidden_list):
        for i, (block, hidden) in enumerate(zip(self.block_list, hidden_list)):
            block.set_hidden(hidden.to(self.devices[self.device_index(i)]) if hidden is not None else None)

    def module_list(self):
        blistlist = []
        for _ in self.devices:
            blistlist.append([])
        for i, block in enumerate(self.block_list):
            blistlist[self.device_index(i)].append(block)
        mlist = []
        for blist in blistlist:
            mlist.append(nn.Sequential(*blist))
        mlist[0] = nn.Sequential(self.token_in, mlist[0])
        mlist[-1] = nn.Sequential(mlist[-1], self.norm_last, self.token_out)
        return mlist
        
    