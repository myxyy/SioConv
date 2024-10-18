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

class BandMHA(nn.Module):
    def __init__(self, dim: int, num_head: int, band_width: int, is_bias: bool=False):
        super().__init__()
        assert dim % num_head == 0, "dim must be multiple of num_head"
        self.dim = dim
        self.num_head = num_head
        self.band_width = band_width
        self.linear_q = nn.Linear(dim, dim, bias=is_bias)
        self.linear_k = nn.Linear(dim, dim, bias=is_bias)
        self.linear_v = nn.Linear(dim, dim, bias=is_bias)
        self.linear_out = nn.Linear(dim, dim, bias=is_bias)
        head_dim = dim // num_head
        self.last_k = None
        self.last_k_init = nn.Parameter(torch.randn(band_width-1, num_head, head_dim))
        self.last_v = None
        self.last_v_init = nn.Parameter(torch.randn(band_width-1, num_head, head_dim))

        self.is_refresh = True

    def forward(self, x):
        batch, length, dim = x.shape
        assert dim == self.dim, "dim missmatch"
        num_head = self.num_head
        head_dim = dim // num_head
        band_width = self.band_width
        q = self.linear_q(x).view(batch, length, num_head, head_dim)
        k = self.linear_k(x).view(batch, length, num_head, head_dim)
        v = self.linear_v(x).view(batch, length, num_head, head_dim)
        if self.last_k is None:
            last_k = self.last_k_init.unsqueeze(0).expand(batch, band_width-1, num_head, head_dim)
        else:
            last_k = self.last_k.detach()
        if self.last_v is None:
            last_v = self.last_v_init.unsqueeze(0).expand(batch, band_width-1, num_head, head_dim)
        else:
            last_v = self.last_v.detach()
        k_with_last_k = torch.cat((last_k, k), dim=1)
        v_with_last_v = torch.cat((last_v, v), dim=1)
        attention_matrix = torch.zeros(batch, length, band_width, num_head, device=x.device, dtype=x.dtype)
        for i in range(band_width):
            attention_matrix[:,:,i,:] = torch.einsum("blhd,blhd->blh", q, k_with_last_k[:,i:i+length,:,:])
        attention_matrix *= head_dim ** -0.5
        attention_matrix = nn.functional.softmax(attention_matrix, dim=2)
        out_acc = torch.zeros(batch, length, num_head, head_dim, device=x.device, dtype=x.dtype)
        for i in range(band_width):
            out_acc += torch.einsum("blh,blhd->blhd", attention_matrix[:,:,i,:], v_with_last_v[:,i:i+length,:,:])
        out = self.linear_out(out_acc.reshape(batch, length, dim))

        if self.is_refresh:
            self.last_k = k_with_last_k[:,-self.band_width:,:,:]
            self.last_v = v_with_last_v[:,-self.band_width:,:,:]

        return out

    def reset_hidden(self):
        self.last_k = None
        self.last_v = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

    def get_hidden(self):
        if self.last_k is None or self.last_v is None:
            return None
        return torch.stack((self.last_k, self.last_v))

    def set_hidden(self, hidden):
        self.last_k = hidden[0]
        self.last_v = hidden[1]
        
class BTBlock(nn.Module):
    def __init__(self, dim: int, num_head: int, dim_ff_hidden: int, band_width: int, dropout: float):
        super().__init__()
        self.band_mha = BandMHA(dim, num_head, band_width)
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden)
        self.norm_band_mha = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.norm_band_mha(x)
        x = self.band_mha(x)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.band_mha.reset_hidden()

    def set_is_refresh(self, is_refresh):
        self.band_mha.set_is_refresh(is_refresh)

    def get_hidden(self):
        return self.band_mha.get_hidden()

    def set_hidden(self, hidden):
        self.band_mha.set_hidden(hidden)

class BandTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        num_head: int,
        dim_ff_hidden: int,
        dropout: float,
        vocab_size: int,
        devices,
        band_width: int,
        token_in_out_parameter_corr = 3.0,
        out_only_device: bool=True,
    ):
        super().__init__()
        self.devices = devices
        self.vocab_size = vocab_size
        self.token_in = nn.Embedding(vocab_size, dim, device=devices[0], max_norm=1)
        self.block_list = nn.ModuleList([BTBlock(dim, num_head, dim_ff_hidden, band_width, dropout) for _ in range(depth)])
        self.norm_last = RMSNorm(dim, device=devices[-1])
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1])

        self.token_in_out_parameter_corr = token_in_out_parameter_corr
        self.num_parameters_token_in = sum(p.numel() for p in self.token_in.parameters())
        self.num_parameters_per_block = sum(p.numel() for p in self.block_list[0].parameters())
        self.num_parameters_norm_last = sum(p.numel() for p in self.norm_last.parameters())
        self.num_parameters_token_out = sum(p.numel() for p in self.token_out.parameters())
        self.num_parameters = self.num_parameters_token_in + (self.num_parameters_per_block * depth) + self.num_parameters_norm_last + self.num_parameters_token_out
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
        
    