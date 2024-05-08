import torch
import torch.nn as nn
import numpy as np
from ..helper import calc_num_parameters
import einops

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
    def __init__(self, dim: int, inner_dim: int, num_head: int, dtype, angle_scale_start=32, angle_scale_end=1024):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim 
        self.num_head = num_head
        self.fc_qkv = nn.Linear(dim, num_head * inner_dim * 3 * 2)
        self.fc_g = nn.Linear(dim, num_head * inner_dim * 2)
        self.fc_a_angle = nn.Linear(dim, num_head)
        self.fc_ln_minus_ln_a_abs = nn.Linear(dim, num_head)
        self.fc_y = nn.Linear(num_head * inner_dim * 2, dim)
        self.angle_base = 1e-4
        self.p_angle = nn.Parameter((self.angle_base ** (torch.arange(num_head*inner_dim)/(num_head*inner_dim))).view(num_head, inner_dim), requires_grad=False)
        self.p_angle_scale = nn.Parameter(torch.exp(torch.linspace(np.log(angle_scale_start), np.log(angle_scale_end), num_head)), requires_grad=False)
        self.fc_p_angle_diff_scale_qk = nn.Linear(dim, num_head)
        self.act = nn.SiLU()
        self.group_norm = nn.GroupNorm(num_head, num_head)
        self.register_buffer('p_angle_diff_q_mask', einops.repeat(torch.Tensor([1,0]), "a -> h a", h=inner_dim//2).reshape(inner_dim))
        self.register_buffer('p_angle_diff_k_mask', einops.repeat(torch.Tensor([0,1]), "a -> h a", h=inner_dim//2).reshape(inner_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_a_angle.weight, gain=1e-2)
        nn.init.zeros_(self.fc_a_angle.bias)
        nn.init.xavier_uniform_(self.fc_ln_minus_ln_a_abs.weight, gain=1e-2)
        nn.init.zeros_(self.fc_ln_minus_ln_a_abs.bias)
        nn.init.xavier_uniform_(self.fc_p_angle_diff_scale_qk.weight, gain=1e-2)
        nn.init.zeros_(self.fc_p_angle_diff_scale_qk.bias)

    #(batch, len, dim),(batch, num_head, inner_dim, inner_dim) -> (batch, len, dim),(batch, num_head, inner_dim, inner_dim)
    def forward(self, x, hidden):
        batch = x.shape[0]
        len = x.shape[1]
        inner_dim = self.inner_dim
        num_head = self.num_head
        dtype = x.dtype

        p_angle = self.p_angle # (num_head, inner_dim)

        x = x.float()
        qkv = torch.view_as_complex(self.fc_qkv(x).view(batch, len, num_head, inner_dim, 3, 2))
        qkv = qkv / (1 + qkv.abs())
        q, k, v = qkv[:,:,:,:,0], qkv[:,:,:,:,1], qkv[:,:,:,:,2] # (batch, len, num_head, inner_dim)

        a_angle = nn.functional.tanh(self.fc_a_angle(x)) # (batch, len, num_head)
        ln_a_abs = - torch.exp(self.fc_ln_minus_ln_a_abs(x)) # (batch, len, num_head)
        ln_a = torch.einsum("blh,h->blh", a_angle * 1j + ln_a_abs, self.p_angle.view(num_head, inner_dim)[:,0]) # (batch, len, num_head)

        len_arange = torch.arange(len, device=x.device)

        p_angle_diff_scale_qk = - nn.functional.sigmoid(self.fc_p_angle_diff_scale_qk(x).view(batch, len, num_head)) # (batch, len, num_head)
        p_angle_diff_qk = torch.einsum("blh,h,hi->blhi", p_angle_diff_scale_qk, self.p_angle_scale, p_angle) # (batch, len, num_head, inner_dim)
        p_angle_diff_q = p_angle_diff_qk * self.p_angle_diff_q_mask
        p_angle_diff_k = p_angle_diff_qk * self.p_angle_diff_k_mask

        p_pow_len_q = torch.exp((torch.einsum("hi,l->lhi", p_angle,  len_arange).unsqueeze(0) + p_angle_diff_q) * 1j) # (batch, len, num_head, inner_dim)
        p_pow_len_k = torch.exp((torch.einsum("hi,l->lhi", p_angle, -len_arange).unsqueeze(0) + p_angle_diff_k) * 1j) # (batch, len, num_head, inner_dim)

        qp = torch.einsum("blhi,blhi->blhi", q, p_pow_len_q)
        kp = torch.einsum("blhi,blhi->blhi", k, p_pow_len_k)

        ones_fft = torch.fft.fft(torch.ones(len, device=x.device), n=len*2)

        ln_a_tri = einops.repeat(ln_a, "b l h -> b h l m", m=len).tril(-1) # (batch, num_head, len, len)
        ln_a_tri_fft = torch.fft.fft(ln_a_tri, n=len*2, dim=2)
        ln_a_tri_conv = torch.fft.ifft(torch.einsum("bhlm,l->bhlm", ln_a_tri_fft, ones_fft), dim=2).narrow(2,0,len) # (batch, num_head, len, len)
        c = torch.exp(ln_a_tri_conv).tril() # (batch, num_head, len, len)

        qck = torch.einsum("blhi,bmhi->bhlm", qp, kp) * c # (batch, num_head, len, len)
        h_inner_chunk = torch.einsum("bhlm,bmhi->blhi", qck, v)

        ln_a_fft = torch.fft.fft(ln_a, n=len*2, dim=1)
        ln_a_conv = torch.fft.ifft(torch.einsum("blh,l->blh", ln_a_fft, ones_fft), dim=1).narrow(1,0,len) # (batch, len, num_head)
        d = torch.exp(ln_a_conv) # (batch, len, num_head)
        p = torch.exp(p_angle * 1j) # (num_head, inner_dim)
        h_cross_chunk = torch.einsum("blhi,bhij->blhj", torch.einsum("blhi,blh->blhi", qp, d), torch.einsum("bhij,hi->bhij", hidden, p))

        h = h_inner_chunk +  h_cross_chunk

        p_pow_len_inverse_pk = torch.exp((torch.einsum("hi,l->lhi", p_angle, len - 1 - len_arange).unsqueeze(0) + p_angle_diff_k) * 1j) # (batch, len, num_head, inner_dim)
        hidden_next_inner_chunk = torch.einsum("blhi,bhl,blhi,blhj->bhij", k, c[:,:,-1,:], p_pow_len_inverse_pk, v)
        hidden_next_cross_chunk = torch.einsum("bh,bhij,hi->bhij", torch.exp(ln_a.sum(dim=1)), hidden, torch.exp(p_angle * len * 1j))
        hidden_next = hidden_next_inner_chunk + hidden_next_cross_chunk

        h = torch.view_as_real(h).reshape(batch*len, num_head, inner_dim*2)
        h = self.group_norm(h)
        h = h.view(batch, len, num_head*inner_dim*2)
        g = self.fc_g(x)
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

    def get_hidden(self):
        return self.last_hidden

    def set_hidden(self, hidden):
        self.last_hidden = hidden
        

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
    
    def get_hidden(self):
        return self.sioconv.get_hidden()

    def set_hidden(self, hidden):
        self.sioconv.set_hidden(hidden)

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
            block.to(devices[self.device_index(i)])

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
            block.set_hidden(hidden_stack[:,i].to(self.devices[self.device_index(i)]))

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
        
    