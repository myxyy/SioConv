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

# [a0, a1, a2, ...], [b0, b1, b2, ...] -> [b0, a1 * b0 + b1, a2 * a1 * b0 + a2 * b1, b2, ...]
def scan(a, b):
    _, length = a.shape
    if length == 1:
        return b
    is_odd = length % 2 == 1
    a_even = a[:,:-1 if is_odd else None:2]
    a_odd = a[:,1::2]
    b_even = b[:,:-1 if is_odd else None:2]
    b_odd = b[:,1::2]
    a_next = a_odd * a_even
    b_next = a_even * b_even + b_odd
    b_new = b.clone()
    mask_odd = torch.zeros(length, device=a.device)
    mask_odd[1::2] = 1
    mask_odd = mask_odd[None,:]
    b_new = b * (1-mask_odd)
    b_new += F.pad(scan(a_next, b_next).repeat_interleave(2, dim=1), (0,1) if is_odd else (0,0), value=0) * mask_odd
    b_odd_new = b_new[:,1:None if is_odd else -1:2]
    a_even_new = a[:,2::2]
    mask_even = torch.zeros(length, device=a.device)
    mask_even[2::2] = 1
    mask_even = mask_even[None,:]
    b_new = b_new + F.pad((a_even_new * b_odd_new).repeat_interleave(2, dim=1), (0,1) if is_odd else (0,2), value=0).roll(1, dims=1) * mask_even
    return b_new

class QuasiLSTM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fc_forget = nn.Linear(dim, dim)
        self.fc_input = nn.Linear(dim, dim)
        self.fc_input_gate = nn.Linear(dim, dim)
        self.fc_output_gate = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.last_hidden = None
        self.last_hidden_init = nn.Parameter(torch.randn(dim)) 
        self.is_refresh = True

    # (batch, len, dim), (batch, dim) -> (batch, len, dim), (batch, dim)
    def forward(self, x):
        batch, len, dim = x.shape

        if self.last_hidden is None:
            last_hidden = self.last_hidden_init.unsqueeze(0).expand(batch, dim)
        else:
            last_hidden = self.last_hidden.detach()

        forget = F.sigmoid(self.fc_forget(x)) # (batch, len, dim)
        h_cross_chunk = last_hidden[:,None,:] * forget.cumprod(1) # (batch, len, dim)

        input = self.tanh(self.fc_input(x)) * self.sigmoid(self.fc_input_gate(x))
        h_inner_chunk = scan(forget.transpose(2,1).reshape(batch * dim, len), input.transpose(2,1).reshape(batch * dim, len)).reshape(batch, dim, len).transpose(2,1)

        h = h_inner_chunk + h_cross_chunk

        if self.is_refresh:
            self.last_hidden = h[:,-1,:]

        y = self.tanh(h) * self.sigmoid(self.fc_output_gate(x))
        return y

    def reset_hidden(self):
        self.last_hidden = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

    def get_hidden(self):
        return self.last_hidden

    def set_hidden(self, hidden):
        self.last_hidden = hidden

class QuasiLSTMBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, dropout: float):
        super().__init__()
        self.sioconv = QuasiLSTM(dim)
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden)
        self.norm_sioconv = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.norm_sioconv(x)
        x = self.sioconv(x)
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

    def set_is_refresh(self, is_refresh):
        self.sioconv.set_is_refresh(is_refresh)

    def get_hidden(self):
        return self.sioconv.get_hidden()

    def set_hidden(self, hidden):
        self.sioconv.set_hidden(hidden)

class QuasiLSTMLM(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        dim_ff_hidden: int,
        dropout: float,
        vocab_size: int,
        devices,
        out_only_device: bool=True,
    ):
        super().__init__()
        self.devices = devices
        self.vocab_size = vocab_size
        self.token_in = nn.Embedding(vocab_size, dim, device=devices[0], max_norm=1)
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1])
        self.block_list = nn.ModuleList([QuasiLSTMBlock(dim, dim_ff_hidden, dropout) for _ in range(depth)])
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
        return torch.stack(hidden_list, dim=1).detach()

    def set_hidden(self, hidden_stack):
        for i, block in enumerate(self.block_list):
            block.set_hidden(hidden_stack[:,i].to(self.devices[self.device_index(i)]) if hidden_stack is not None else None)

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
        
    