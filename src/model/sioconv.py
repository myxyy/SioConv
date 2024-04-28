import torch
import torch.nn as nn
import numpy as np
from ..helper import calc_num_parameters

class FFN(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: float, dtype):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim_ff_hidden, bias=True, dtype=dtype)
        self.linear_2 = nn.Linear(dim_ff_hidden, dim, bias=True, dtype=dtype)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x

class SioConvLayer(nn.Module):
    def __init__(self, dim: int, inner_dim: int, num_head: int, dtype, a_scale_min: float=0.125, a_scale_max: float=4):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim 
        self.num_head = num_head
        self.fc_q = nn.Linear(dim, num_head * inner_dim * 2)
        self.fc_k = nn.Linear(dim, num_head * inner_dim * 2)
        self.fc_v = nn.Linear(dim, num_head * inner_dim * 2)
        self.fc_a_angle = nn.Linear(dim, num_head)
        self.fc_g = nn.Linear(dim, num_head * inner_dim * 2)
        self.fc_y = nn.Linear(num_head * inner_dim * 2, dim)
        self.p_angle = nn.Parameter((1e-4 ** (torch.arange(inner_dim)/inner_dim)).unsqueeze(0).expand(num_head, inner_dim), requires_grad=False)
        self.act = nn.SiLU()
        self.group_norm = nn.GroupNorm(num_head, num_head)
        self.a_scale = 1 - 1e-5

    #(batch, len, dim),(batch, num_head, inner_dim, inner_dim) -> (batch, len, dim),(batch, num_head, inner_dim, inner_dim)
    def forward(self, x, hidden):
        batch = x.shape[0]
        len = x.shape[1]
        inner_dim = self.inner_dim
        num_head = self.num_head
        dtype = x.dtype

        x = x.float()
        q, k, v, g = self.fc_q(x), self.fc_k(x), self.fc_v(x), self.fc_g(x) # (batch, len, num_head * inner_dim * 2)
        q = torch.view_as_complex(q.view(batch, len, num_head, inner_dim, 2))  # (batch, len, num_head, inner_dim)
        k = torch.view_as_complex(k.view(batch, len, num_head, inner_dim, 2))  # (batch, len, num_head, inner_dim)
        v = torch.view_as_complex(v.view(batch, len, num_head, inner_dim, 2))  # (batch, len, num_head, inner_dim)
        a_angle = self.fc_a_angle(x) # (batch, len, num_head)
        a = torch.exp(a_angle * 1j) * self.a_scale
        p_angle = self.p_angle # (num_head, inner_dim)

        len_arange = torch.arange(len, device=x.device)
        p_pow_len = torch.exp(torch.einsum("hi,l->lhi", p_angle, len_arange) * 1j) # (len, num_head, inner_dim)
        q = torch.einsum("blhi,lhi->blhi", q, p_pow_len)
        k = torch.einsum("blhi,lhi->blhi", k, torch.conj(p_pow_len))

        ln_a = torch.log(a) # (batch, len, num_head)
        ln_a_tri = ln_a.permute(0,2,1).unsqueeze(2).expand(batch, num_head, len, len).tril(-1) # (batch, num_head, len, len)
        ln_a_tri_fft = torch.fft.fft(ln_a_tri, n=len*2, dim=2)
        ones_fft = torch.fft.fft(torch.ones(len, device=x.device), n=len*2)
        ln_a_tri_conv = torch.fft.ifft(torch.einsum("bhlm,l->bhlm", ln_a_tri_fft, ones_fft), dim=2).narrow(2,0,len) # (batch, num_head, len, len)
        c = torch.exp(ln_a_tri_conv).tril() # (batch, num_head, len, len)

        qk = torch.einsum("blhi,bmhi->bhlm", q, k) # (batch, num_head, len, len)
        h_inner_chunk = torch.einsum("bhlm,bmhi->blhi", qk * c, v)

        ln_a_fft = torch.fft.fft(ln_a, n=len*2, dim=1)
        ln_a_conv = torch.fft.ifft(torch.einsum("blh,l->blh", ln_a_fft, ones_fft), dim=1).narrow(1,0,len) # (batch, len, num_head)
        d = torch.exp(ln_a_conv) # (batch, len, num_head)
        p = torch.exp(p_angle * 1j) # (num_head, inner_dim)
        h_cross_chunk = torch.einsum("blhi,blh,bhij->blhj", q, d, torch.einsum("bhij,hi->bhij", hidden, p))

        h = h_inner_chunk +  h_cross_chunk

        hidden_next_inner_chunk = torch.einsum("blhi,bhl,blhj,hi->bhij", k, c[:,:,-1,:], v, torch.exp(p_angle * (len-1) * 1j))
        hidden_next_cross_chunk = torch.einsum("bh,bhij,hi->bhij", torch.exp(ln_a.sum(dim=1)), hidden, torch.exp(p_angle * len * 1j))
        hidden_next = hidden_next_inner_chunk + hidden_next_cross_chunk

        h = torch.view_as_real(h).reshape(batch*len, num_head, inner_dim*2)
        h = self.group_norm(h)
        h = h.view(batch, len, num_head*inner_dim*2)
        y = self.fc_y(h * self.act(g))
        return y.to(dtype), hidden_next


class ChunkWiseSioConvLayer(nn.Module):
    def __init__(self, dim: int, inner_dim: int, num_head: int, chunk_size: int, dtype):
        super().__init__()
        self.sioconv = SioConvLayer(dim, inner_dim, num_head, dtype)
        self.last_hidden = None
        self.last_hidden_init = nn.Parameter(torch.randn(num_head, inner_dim, inner_dim, dtype=torch.cfloat))
        self.is_refresh = True
        self.inner_dim = inner_dim 
        self.num_head = num_head
        self.chunk_size = chunk_size

    def forward(self, x):
        batch = x.shape[0]
        inner_dim = self.inner_dim
        num_head = self.num_head

        if self.last_hidden is None:
            hidden = self.last_hidden_init.unsqueeze(0).expand(batch, num_head, inner_dim, inner_dim)
        else:
            hidden = self.last_hidden.detach()

        input_chunks = x.split(self.chunk_size, dim=1)
        output_chunks = []
        for input_chunk in input_chunks:
            output_chunk, hidden = self.sioconv(input_chunk, hidden)
            output_chunks.append(output_chunk)

        if self.is_refresh:
            self.last_hidden = hidden

        return torch.cat(output_chunks, dim=1)
 
    def reset_hidden(self):
        self.last_hidden = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

class SioConvBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, inner_dim: int, num_head: int, chunk_size:int, dropout: float, dtype):
        super().__init__()
        self.dtype = dtype 

        self.layer_norm_sioconv = nn.LayerNorm(dim, dtype=dtype)
        self.sioconv = ChunkWiseSioConvLayer(dim, inner_dim, num_head, chunk_size, dtype)

        self.layer_norm_ffn = nn.LayerNorm(dim, dtype=dtype)
        self.ffn_sc = FFN(dim, dim_ff_hidden, dtype)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.layer_norm_sioconv(x)
        x = self.sioconv(x)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.layer_norm_ffn(x)
        x = self.ffn_sc(x)
        x = self.dropout(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.sioconv.reset_hidden()

    def set_is_refresh(self, is_refresh):
        self.sioconv.set_is_refresh(is_refresh)

class SioConv(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        dim_ff_hidden: int,
        inner_dim: int,
        num_head: int,
        chunk_size: int,
        dropout: float,
        vocab_size: int,
        devices,
        out_only_device: bool=True,
        dtype=torch.float,
    ):
        super().__init__()
        self.devices = devices
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.token_in = nn.Embedding(vocab_size, dim, device=devices[0], max_norm=1, dtype=dtype)
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1], dtype=dtype)
        self.block_list = nn.ModuleList([SioConvBlock(dim, dim_ff_hidden, inner_dim, num_head, chunk_size, dropout, dtype) for _ in range(depth)])
        self.layer_norm_last = nn.LayerNorm(dim, device=devices[-1], dtype=dtype)

        self.num_parameters_token_in = calc_num_parameters(self.token_in)
        self.num_parameters_per_block = calc_num_parameters(self.block_list[0])
        self.num_parameters_layer_norm_last = calc_num_parameters(self.layer_norm_last)
        self.num_parameters_token_out = calc_num_parameters(self.token_out)
        self.num_parameters = (self.num_parameters_per_block * depth) + self.num_parameters_layer_norm_last + (self.num_parameters_token_in + self.num_parameters_token_out)
        self.out_only_device = out_only_device

        for i, block in enumerate(self.block_list):
            self.block_list[i] = block.to(devices[self.device_index(i)])

    def device_index(self, i):
        return (int)(((len(self.devices)-(1 if self.out_only_device else 0)) * (i * self.num_parameters_per_block + self.num_parameters_token_in)) / self.num_parameters)

    def forward(self, x):
        x = self.token_in(x).to(self.dtype)
        for i, block in enumerate(self.block_list):
            x = x.to(self.devices[self.device_index(i)])
            x = block(x)
        x = x.to(self.devices[-1])
        x = self.layer_norm_last(x)
        x = self.token_out(x)
        return x 

    def reset_hidden(self):
        for block in self.block_list:
            block.reset_hidden()

    def set_is_refresh(self, is_refresh):
        for block in self.block_list:
            block.set_is_refresh(is_refresh)

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
        mlist[-1] = nn.Sequential(mlist[-1], self.layer_norm_last, self.token_out)
        return mlist
        
    