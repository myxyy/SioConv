import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

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

class NeuralMemory(nn.Module):
    def __init__(self, dim: int, dim_hidden:int, base_lr: float, base_weight_decay: float):
        super().__init__()
        self.log_base_lr = nn.Parameter(torch.tensor(np.log(base_lr)))
        self.log_base_weight_decay = nn.Parameter(torch.tensor(np.log(base_weight_decay)))
        self.linear_query = nn.Parameter(torch.randn(dim_hidden, dim) * dim ** -0.5)
        self.linear_key = nn.Parameter(torch.randn(dim_hidden, dim) * dim ** -0.5)
        self.linear_value = nn.Parameter(torch.randn(dim, dim) * dim ** -0.5)
        self.fc_lr = nn.Linear(dim, 1)
        self.fc_weight_decay = nn.Linear(dim, 1)

    def forward(self, x, hidden):
        batch, length, dim = x.shape
        W_prev = hidden # (batch, dim, dim_hidden)
        query = F.linear(x, self.linear_query) # (batch, length, dim_hidden)
        key = F.linear(x, self.linear_key) # (batch, length, dim_hidden)
        value = F.linear(x, self.linear_value) # (batch, length, dim_hidden)
        lr = torch.exp(self.log_base_lr) * F.sigmoid(self.fc_lr(x).squeeze(-1)) # (batch, length)
        log_weight_decay = torch.log(torch.exp(self.log_base_weight_decay) * F.sigmoid(self.fc_weight_decay(x).squeeze(-1))) # (batch, length)
        weight_decay_cross_chunk = torch.exp(torch.cumsum(log_weight_decay, dim=1)) # (batch, length)
        weight_decay_inner_chunk = torch.exp(torch.cumsum(einops.repeat(log_weight_decay, "b l -> b m l", m=length).triu(1), dim=2)).triu() # (batch, length, length)
        kq = torch.einsum("b l d, b m d -> b l m", key, query) # (batch, length, length)
        mask_kq = kq * weight_decay_inner_chunk # (batch, length, length)
        WpK_V = torch.einsum("b d h, b l h -> b l d", W_prev, key) - value # (batch, length, dim)
        y_inner_chunk = -torch.einsum("b l d, b l, b l m -> b l d", WpK_V, lr, mask_kq) # (batch, length, dim)
        y_cross_chunk = torch.einsum("b d h, b l h, b l -> b l d", W_prev, key, weight_decay_cross_chunk) # (batch, length, dim)
        y = y_inner_chunk + y_cross_chunk # (batch, length, dim)
        W_next_inner_chunk = -torch.einsum("b l d, b l, b l h -> b d h", WpK_V, lr * weight_decay_inner_chunk[:,:,-1], key) # (batch, dim, dim_hidden)
        W_next_cross_chunk = W_prev * weight_decay_cross_chunk[:,-1][:,None,None] # (batch, dim, dim_hidden)
        W_next = W_next_inner_chunk + W_next_cross_chunk # (batch, dim, dim_hidden)
        return y, W_next
        

class ChunkwiseNeuralMemory(nn.Module):
    def __init__(self, dim: int, dim_hidden: int, base_lr: float, base_weight_decay: float, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.memory = NeuralMemory(dim, dim_hidden, base_lr, base_weight_decay)
        self.last_hidden = None
        self.W = None
        self.W_init = nn.Parameter(torch.randn(dim, dim_hidden) * dim_hidden ** -0.5)
        self.is_refresh = True

    def forward(self, x):
        batch, length, dim = x.shape

        if self.last_hidden is None:
            hidden = einops.repeat(self.W_init, "d h -> b d h", b=batch)
        else:
            hidden = self.last_hidden.detach()

        input_chunks = x.split(self.chunk_size, dim=1)
        output_chunks = []
        for input_chunk in input_chunks:
            output_chunk, hidden = self.memory(input_chunk, hidden)
            output_chunks.append(output_chunk)

        if self.is_refresh:
            self.last_hidden = hidden

        return torch.cat(output_chunks, dim=1)
 
    def reset_hidden(self):
        self.last_hidden = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

    def get_hidden(self):
        return self.last_hidden

    def set_hidden(self, hidden):
        self.last_hidden = hidden

class NeuralMemoryBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, base_lr: float, base_weight_decay: float, chunk_size: int, dropout: float):
        super().__init__()
        self.memory = ChunkwiseNeuralMemory(dim, dim_ff_hidden, base_lr, base_weight_decay, chunk_size)
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden)
        self.norm_memory = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
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
        self.memory.reset_hidden()

    def set_is_refresh(self, is_refresh):
        self.memory.set_is_refresh(is_refresh)

    def get_hidden(self):
        return self.memory.get_hidden()

    def set_hidden(self, hidden):
        self.memory.set_hidden(hidden)

class NeuralMemoryLM(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        dim_ff_hidden: int,
        base_lr: float,
        base_weight_decay: float,
        dropout: float,
        vocab_size: int,
        chunk_size: int,
        devices,
        token_in_out_parameter_corr = 3.0,
        out_only_device: bool=True,
    ):
        super().__init__()
        self.devices = devices
        self.vocab_size = vocab_size
        self.token_in = nn.Embedding(vocab_size, dim, device=devices[0], max_norm=1)
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1])
        self.block_list = nn.ModuleList([NeuralMemoryBlock(dim, dim_ff_hidden, base_lr, base_weight_decay, chunk_size, dropout) for _ in range(depth)])
        self.norm_last = RMSNorm(dim, device=devices[-1])

        self.token_in_out_parameter_corr = token_in_out_parameter_corr
        self.num_parameters_token_in = sum(p.numel() for p in self.token_in.parameters())
        self.num_parameters_per_block = sum(p.numel() for p in self.block_list[0].parameters())
        self.num_parameters_norm_last = sum(p.numel() for p in self.norm_last.parameters())
        self.num_parameters_token_out = sum(p.numel() for p in self.token_out.parameters())
        self.num_parameters = (self.num_parameters_per_block * depth) + self.num_parameters_norm_last + (self.num_parameters_token_in + self.num_parameters_token_out)
        self.out_only_device = out_only_device

        for i, block in enumerate(self.block_list):
            self.block_list[i] = block.to(devices[self.device_index(i)])

    def device_index(self, i):
        return (int)(((len(self.devices)-(1 if self.out_only_device else 0)) * (i * self.num_parameters_per_block + self.num_parameters_token_in)) / self.num_parameters)

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
        for i, block in enumerate(self.block_list):
            block.set_hidden(hidden_list[i].to(self.devices[self.device_index(i)]) if hidden_list is not None else None)

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
        