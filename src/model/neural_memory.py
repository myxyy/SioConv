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

def silu_backward(x):
    return F.silu(x) + F.sigmoid(x) * (1 - F.silu(x))

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

class NeuralMemory(nn.Module):
    def __init__(self, dim: int, dim_hidden:int, base_lr: float):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.base_lr = base_lr
        self.fc_query = nn.Linear(dim, dim)
        self.fc_key = nn.Linear(dim, dim)
        self.fc_value = nn.Linear(dim, dim)
        self.fc_lr = nn.Linear(dim, 1)
        self.fc_momentum = nn.Linear(dim, 1)
        self.fc_weight_decay = nn.Linear(dim, 1)

        self.is_refresh = True

    def forward(self, x, hidden):
        dim_hidden = self.dim_hidden

        batch, length, dim = x.shape
        query = self.fc_query(x)
        key = self.fc_key(x)
        value = self.fc_value(x)

        W1 = hidden["W1"]
        b1 = hidden["b1"]
        W2 = hidden["W2"]
        b2 = hidden["b2"]
        momentum_W1_grad = hidden["momentum_W1_grad"]
        momentum_b1_grad = hidden["momentum_b1_grad"]
        momentum_W2_grad = hidden["momentum_W2_grad"]
        momentum_b2_grad = hidden["momentum_b2_grad"]

        X1 = key # (batch, length, dim)
        Z1 = torch.einsum("bhd,bld->blh", W1, X1) + b1 # (batch, length, dim_hidden)
        X2 = F.silu(Z1) # (batch, length, dim_hidden)
        Z2 = torch.einsum("bdh,blh->bld", W2, X2) + b2 # (batch, length, dim)

        # loss = 0.5 * ((value - Z2) ** 2)
        grad_Z2 = Z2 - value # (batch, length, dim)
        grad_b2 = grad_Z2 # (batch, length, dim)
        grad_W2 = torch.einsum("bld,blh->bldh", grad_Z2, X2) # (batch, length, dim, dim_hidden)
        grad_X2 = torch.einsum("bld,bdh->blh", grad_Z2, W2) # (batch, length, dim_hidden)
        grad_Z1 = grad_X2 * silu_backward(Z1) # (batch, length, dim_hidden)
        grad_b1 = grad_Z1 # (batch, length, dim_hidden)
        grad_W1 = torch.einsum("blh,bld->blhd", grad_Z1, X1) # (batch, length, dim_hidden, dim)

        lr = F.softplus(self.fc_lr(x).squeeze(-1) + np.log(np.expm1(self.base_lr))) # (batch, length)
        grad_W1_lr = torch.einsum("blhd,bl->blhd", grad_W1, lr) # (batch, length, dim_hidden, dim)
        grad_b1_lr = grad_b1 * lr.unsqueeze(-1) # (batch, length, dim_hidden)
        grad_W2_lr = torch.einsum("bldh,bl->bldh", grad_W2, lr) # (batch, length, dim, dim_hidden)
        grad_b2_lr = grad_b2 * lr.unsqueeze(-1) # (batch, length, dim)

        fc_momentum = self.fc_momentum(x).squeeze(-1) # (batch, length)

        momentum_grad_W1_lr_inner_chunk = scan(
            fc_momentum[:,:,None,None].expand(batch, length, dim_hidden, dim).transpose(3,1).reshape(-1,length),
            grad_W1_lr.transpose(3,1).reshape(-1,length)
        ).reshape(batch, dim, dim_hidden, length).transpose(3,1)
        momentum_grad_b1_lr_inner_chunk = scan(
            fc_momentum[:,:,None].expand(batch, length, dim_hidden).transpose(2,1).reshape(-1,length),
            grad_b1_lr.transpose(2,1).reshape(-1,length)
        ).reshape(batch, dim_hidden, length).transpose(2,1)
        momentum_grad_W2_lr_inner_chunk = scan(
            fc_momentum[:,:,None,None].expand(batch, length, dim, dim_hidden).transpose(3,1).reshape(-1,length),
            grad_W2_lr.transpose(3,1).reshape(-1,length)
        ).reshape(batch, dim_hidden, dim, length).transpose(3,1)
        momentum_grad_b2_lr_inner_chunk = scan(
            fc_momentum[:,:,None].expand(batch, length, dim).transpose(2,1).reshape(-1,length),
            grad_b2_lr.transpose(2,1).reshape(-1,length)
        ).reshape(batch, dim, length).transpose(2,1)

        log_momentum = -F.softplus(-fc_momentum) # (batch, length)
        log_momentum_cumsum = torch.cumsum(log_momentum, dim=1) # (batch, length)
        momentum_cumsum = torch.exp(log_momentum_cumsum) # (batch, length)

        momentum_grad_W1_lr_cross_chunk = torch.einsum("bl,bhd->blhd", momentum_cumsum, momentum_W1_grad) # (batch, length, dim_hidden, dim)
        momentum_grad_b1_lr_cross_chunk = torch.einsum("bl,bh->blh", momentum_cumsum, momentum_b1_grad) # (batch, length, dim_hidden)
        momentum_grad_W2_lr_cross_chunk = torch.einsum("bl,bdh->bldh", momentum_cumsum, momentum_W2_grad) # (batch, length, dim, dim_hidden)
        momentum_grad_b2_lr_cross_chunk = torch.einsum("bl,bd->bld", momentum_cumsum, momentum_b2_grad) # (batch, length, dim)

        momentum_grad_W1_lr = momentum_grad_W1_lr_inner_chunk - momentum_grad_W1_lr_cross_chunk # (batch, length, dim_hidden, dim)
        momentum_grad_b1_lr = momentum_grad_b1_lr_inner_chunk - momentum_grad_b1_lr_cross_chunk # (batch, length, dim_hidden)
        momentum_grad_W2_lr = momentum_grad_W2_lr_inner_chunk - momentum_grad_W2_lr_cross_chunk # (batch, length, dim, dim_hidden)
        momentum_grad_b2_lr = momentum_grad_b2_lr_inner_chunk - momentum_grad_b2_lr_cross_chunk # (batch, length, dim)

        weight_decay = self.fc_weight_decay(x).squeeze(-1)

        W1_progress_inner_chunk = scan(
            weight_decay[:,:,None,None].expand(batch, length, dim_hidden, dim).transpose(3,1).reshape(-1,length),
            momentum_grad_W1_lr.transpose(3,1).reshape(-1,length)
        ).reshape(batch, dim, dim_hidden, length).transpose(3,1)
        b1_progress_inner_chunk = scan(
            weight_decay[:,:,None].expand(batch, length, dim_hidden).transpose(2,1).reshape(-1,length),
            momentum_grad_b1_lr.transpose(2,1).reshape(-1,length)
        ).reshape(batch, dim_hidden, length).transpose(2,1)
        W2_progress_inner_chunk = scan(
            weight_decay[:,:,None,None].expand(batch, length, dim, dim_hidden).transpose(3,1).reshape(-1,length),
            momentum_grad_W2_lr.transpose(3,1).reshape(-1,length)
        ).reshape(batch, dim_hidden, dim, length).transpose(3,1)
        b2_progress_inner_chunk = scan(
            weight_decay[:,:,None].expand(batch, length, dim).transpose(2,1).reshape(-1,length),
            momentum_grad_b2_lr.transpose(2,1).reshape(-1,length)
        ).reshape(batch, dim, length).transpose(2,1)

        log_weight_decay = -F.softplus(weight_decay) # (batch, length)
        log_weight_decay_cumsum = torch.cumsum(log_weight_decay, dim=1) # (batch, length)
        weight_decay_cumsum = torch.exp(log_weight_decay_cumsum) # (batch, length)

        W1_progress_cross_chunk = torch.einsum("bl,bhd->blhd", weight_decay_cumsum, W1) # (batch, length, dim_hidden, dim)
        b1_progress_cross_chunk = torch.einsum("bl,bh->blh", weight_decay_cumsum, b1) # (batch, length, dim_hidden)
        W2_progress_cross_chunk = torch.einsum("bl,bdh->bldh", weight_decay_cumsum, W2) # (batch, length, dim, dim_hidden)
        b2_progress_cross_chunk = torch.einsum("bl,bd->bld", weight_decay_cumsum, b2) # (batch, length, dim)

        W1_progress = W1_progress_inner_chunk + W1_progress_cross_chunk # (batch, length, dim_hidden, dim)
        b1_progress = b1_progress_inner_chunk + b1_progress_cross_chunk # (batch, length, dim_hidden)
        W2_progress = W2_progress_inner_chunk + W2_progress_cross_chunk # (batch, length, dim, dim_hidden)
        b2_progress = b2_progress_inner_chunk + b2_progress_cross_chunk # (batch, length, dim)

        Xq1 = query # (batch, length, dim)
        Zq1 = torch.einsum("blhd,bld->blh", W1_progress, Xq1) + b1_progress # (batch, length, dim_hidden)
        Xq2 = F.silu(Zq1) # (batch, length, dim_hidden)
        Zq2 = torch.einsum("bldh,blh->bld", W2_progress, Xq2) + b2_progress # (batch, length, dim)

        next_hidden = {
            "W1": W1_progress[:,-1,:,:],
            "b1": b1_progress[:,-1,:],
            "W2": W2_progress[:,-1,:,:],
            "b2": b2_progress[:,-1,:],
            "momentum_W1_grad": momentum_grad_W1_lr[:,-1,:,:],
            "momentum_b1_grad": momentum_grad_b1_lr[:,-1,:],
            "momentum_W2_grad": momentum_grad_W2_lr[:,-1,:,:],
            "momentum_b2_grad": momentum_grad_b2_lr[:,-1,:],
        }

        return Zq2, next_hidden

class ChunkwiseNeuralMemory(nn.Module):
    def __init__(self, dim: int, dim_hidden: int, base_lr: float, chunk_size: int):
        super().__init__()
        self.memory = NeuralMemory(dim, dim_hidden, base_lr)
        self.chunk_size = chunk_size
        self.last_hidden = None
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.W1_init = nn.Parameter(torch.randn(dim_hidden, dim) * dim_hidden ** -0.5)
        self.b1_init = nn.Parameter(torch.zeros(dim_hidden))
        self.W2_init = nn.Parameter(torch.randn(dim, dim_hidden) * dim ** -0.5)
        self.b2_init = nn.Parameter(torch.zeros(dim))
        self.momentum_W1_grad = None
        self.momentum_b1_grad = None
        self.momentum_W2_grad = None
        self.momentum_b2_grad = None
        self.is_refresh = True

    def forward(self, x):
        batch, length, dim = x.shape

        if self.last_hidden is None:
            W1 = einops.repeat(self.W1_init, "h d -> b h d", b=batch)
            b1 = einops.repeat(self.b1_init, "h -> b h", b=batch)
            W2 = einops.repeat(self.W2_init, "d h -> b d h", b=batch)
            b2 = einops.repeat(self.b2_init, "d -> b d", b=batch)
            hidden = {
                "W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2,
                "momentum_W1_grad": torch.zeros_like(W1),
                "momentum_b1_grad": torch.zeros_like(b1),
                "momentum_W2_grad": torch.zeros_like(W2),
                "momentum_b2_grad": torch.zeros_like(b2),
            }
        else:
            hidden = {k: v.detach() for k, v in self.last_hidden.items()}
        
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
    def __init__(self, dim: int, dim_ff_hidden: int, base_lr: float, dropout: float, chunk_size: int):
        super().__init__()
        self.memory = ChunkwiseNeuralMemory(dim, dim_ff_hidden, base_lr, chunk_size)
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
        dropout: float,
        vocab_size: int,
        devices,
        chunk_size: int=512,
        token_in_out_parameter_corr = 3.0,
        out_only_device: bool=True,
    ):
        super().__init__()
        self.devices = devices
        self.vocab_size = vocab_size
        self.token_in = nn.Embedding(vocab_size, dim, device=devices[0], max_norm=1)
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1])
        self.block_list = nn.ModuleList([NeuralMemoryBlock(dim, dim_ff_hidden, base_lr, dropout, chunk_size) for _ in range(depth)])
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
        