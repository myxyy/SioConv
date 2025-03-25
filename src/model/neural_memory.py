import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops


def silu_backward(x):
    return F.silu(x) + F.sigmoid(x) * (1 - F.silu(x))

class NeuralMemory(nn.Module):
    def __init__(self, dim: int, dim_hidden: int, base_lr: float):
        super().__init__()
        self.base_lr = base_lr
        self.fc_query = nn.Linear(dim, dim)
        self.fc_key = nn.Linear(dim, dim)
        self.fc_value = nn.Linear(dim, dim)
        self.fc_lr = nn.Linear(dim, 1)
        self.fc_momentum = nn.Linear(dim, 1)
        self.fc_weight_decay = nn.Linear(dim, 1)

        self.is_refresh = True

    def forward(self, x, hidden):
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
        log_momentum = -F.softplus(-fc_momentum) # (batch, length)
        log_momentum_masked = einops.repeat(log_momentum, "b l -> b l m", m=length).tril(-1) # (batch, length, length)
        log_momentum_masked_cumsum = torch.cumsum(log_momentum_masked, dim=1) # (batch, length, length)
        log_om_momentum = -F.softplus(fc_momentum) # (batch, length)
        log_momentum_masked += log_om_momentum[:,:,None] # (batch, length, length)
        momentum_masked_cumsum = torch.exp(log_momentum_masked_cumsum).tril() # (batch, length, length)

        momentum_grad_W1_lr_inner_chunk = torch.einsum("blm,bmhd->blhd", momentum_masked_cumsum, grad_W1_lr) # (batch, length, dim_hidden, dim)
        momentum_grad_b1_lr_inner_chunk = torch.einsum("blm,bmh->blh", momentum_masked_cumsum, grad_b1_lr) # (batch, length, dim_hidden)
        momentum_grad_W2_lr_inner_chunk = torch.einsum("blm,bmdh->bldh", momentum_masked_cumsum, grad_W2_lr) # (batch, length, dim, dim_hidden)
        momentum_grad_b2_lr_inner_chunk = torch.einsum("blm,bmd->bld", momentum_masked_cumsum, grad_b2_lr) # (batch, length, dim)

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

        log_weight_decay = -F.softplus(self.fc_weight_decay(x).squeeze(-1)) # (batch, length)
        log_weight_decay_masked = einops.repeat(log_weight_decay, "b l -> b l m", m=length).tril(-1) # (batch, length, length)
        log_weight_decay_masked_cumsum = torch.cumsum(log_weight_decay_masked, dim=1) # (batch, length, length)
        weight_decay_masked_cumsum = torch.exp(log_weight_decay_masked_cumsum).tril() # (batch, length, length)

        W1_progress_inner_chunk = torch.einsum("blm,bmhd->blhd", weight_decay_masked_cumsum, momentum_grad_W1_lr) # (batch, length, dim_hidden, dim)
        b1_progress_inner_chunk = torch.einsum("blm,bmh->blh", weight_decay_masked_cumsum, momentum_grad_b1_lr) # (batch, length, dim_hidden)
        W2_progress_inner_chunk = torch.einsum("blm,bmdh->bldh", weight_decay_masked_cumsum, momentum_grad_W2_lr) # (batch, length, dim, dim_hidden)
        b2_progress_inner_chunk = torch.einsum("blm,bmd->bld", weight_decay_masked_cumsum, momentum_grad_b2_lr) # (batch, length, dim)

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
