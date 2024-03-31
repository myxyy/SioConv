import torch
import torch.nn as nn
import numpy as np

class FFN(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: float, dtype):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim_ff_hidden, bias=True, dtype=dtype)
        nn.init.normal_(self.linear_1.weight, std=dim**-0.5)
        nn.init.constant_(self.linear_1.bias, 0)
        self.linear_2 = nn.Linear(dim_ff_hidden, dim, bias=True, dtype=dtype)
        nn.init.normal_(self.linear_2.weight, std=dim_ff_hidden**-0.5)
        nn.init.constant_(self.linear_2.bias, 0)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x

class SioConvLayer(nn.Module):
    def __init__(self, dim: int, inner_dim: int, num_head: int, dtype):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim 
        self.num_head = num_head
        self.fc_in = nn.Linear(dim, dim)
        self.fc_qkva = nn.Linear(dim, num_head * inner_dim * 8)
        self.fc_y = nn.Linear(num_head * inner_dim * 2, dim)
        self.fc_out = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.last_hidden = None
        self.last_hidden_init = nn.Parameter(torch.randn(num_head, inner_dim, inner_dim, dtype=torch.cfloat))
        self.is_refresh = True

    #(batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        dim = self.dim
        inner_dim = self.inner_dim
        num_head = self.num_head
        dtype = x.dtype

        if self.last_hidden is None:
            self.last_hidden = self.last_hidden_init.unsqueeze(0).expand(batch, num_head, inner_dim, inner_dim)
        else:
            self.last_hidden = self.last_hidden.detach()
        
        x = x.float()
        x = self.fc_in(x)
        x = self.act(x)
        x = self.fc_qkva(x) # (batch, len, num_head * inner_dim * 4)
        x = torch.view_as_complex(x.view(batch, len, num_head, inner_dim, 4, 2))  # (batch, len, num_head, inner_dim, 4)
        q, k, v, a = x[:,:,:,:,0], x[:,:,:,:,1], x[:,:,:,:,2], x[:,:,:,:,3] # (batch, len, num_head, inner_dim)
        a_sqr_mag = a.real * a.real + a.imag * a.imag
        a = a * torch.rsqrt(a_sqr_mag) * torch.sigmoid(torch.log(a_sqr_mag)) # (batch, len, num_head, inner_dim)

        if len == 1:
            ah = torch.einsum("blhd,bhde->blhde", a, self.last_hidden)
            kv = torch.einsum("blhd,blhe->blhde", k, v)
            h = ah + kv
        else:
            a_ln = torch.log(a)
            a_ln_tri = a_ln.permute(0,2,3,1).unsqueeze(3).expand(batch, num_head, inner_dim, len, len).triu() # (batch, num_head, inner_dim, len, len)
            a_ln_tri_fft = torch.fft.fft(a_ln_tri, n=len*2, dim=4)
            ones_fft = torch.fft.fft(torch.ones(len, len, device=x.device), n=len*2, dim=1)
            a_ln_tri_fft_ones_fft = a_ln_tri_fft * ones_fft.unsqueeze(0).unsqueeze(1).unsqueeze(2)
            a_ln_tri_conv = torch.fft.ifft(a_ln_tri_fft_ones_fft).narrow(4,0,len) # (batch, num_head, inner_dim, len, len)
            a_mat = torch.exp(a_ln_tri_conv).triu(diagonal=-1) # (batch, num_head, inner_dim, len, len)

            kv = torch.einsum("blhd,blhe->blhde", k, v) # (batch, len, num_head, inner_dim, inner_dim)
            kv_h = kv.roll(1, dims=1)
            kv_h[:,0,:,:,:] = self.last_hidden
            akv_h = torch.einsum("bhdml,bmhde->blhde", a_mat, kv_h) # (batch, len, num_head, inner_dim, inner_dim)
            akv_h[:,-1,:,:,:] += kv[:,-1,:,:,:]
            h = akv_h

        y = torch.einsum("blhd,blhde->blhe", q, h) # (batch, len, num_head, inner_dim)
        if self.is_refresh:
            self.last_hidden = h[:,-1,:,:,:]
        y = self.fc_y(torch.view_as_real(y).reshape(batch, len, num_head*inner_dim*2))
        y = self.act(y)
        y = self.fc_out(y)
        return y.to(dtype)

    def reset_hidden(self):
        self.last_hidden = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

class SioConvBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, inner_dim: int, num_head: int, dropout: float, dtype):
        super().__init__()
        self.dtype = dtype 

        self.layer_norm_sc_in = nn.LayerNorm(dim, elementwise_affine=True, bias=True, dtype=dtype)
        self.sioconv = SioConvLayer(dim, inner_dim, num_head, dtype)

        self.layer_norm_ffn_sc_in = nn.LayerNorm(dim, elementwise_affine=True, bias=True, dtype=dtype)
        self.ffn_sc = FFN(dim, dim_ff_hidden, dtype)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.layer_norm_sc_in(x)
        x = self.sioconv(x)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.layer_norm_ffn_sc_in(x)
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
        dropout: float,
        vocab_size: int,
        devices,
        dtype=torch.float,
    ):
        super().__init__()
        self.devices = devices
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.token_in = nn.Linear(vocab_size, dim, device=devices[0], dtype=dtype)
        nn.init.normal_(self.token_in.weight, std=vocab_size**-0.5)
        nn.init.constant_(self.token_in.bias, 0)
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1], dtype=dtype)
        nn.init.normal_(self.token_out.weight, std=dim**-0.5)
        nn.init.constant_(self.token_out.bias, 0)
        self.block_list = nn.ModuleList([SioConvBlock(dim, dim_ff_hidden, inner_dim, num_head, dropout, dtype) for _ in range(depth)])
        self.layer_norm_last = nn.LayerNorm(dim, elementwise_affine=True, bias=True, device=devices[-1], dtype=dtype)

        self.num_parameters_token_in = sum(p.numel() for p in self.token_in.parameters())
        self.num_parameters_per_block = sum(p.numel() for p in self.block_list[0].parameters())
        self.num_parameters_layer_norm_last = sum(p.numel() for p in self.layer_norm_last.parameters())
        self.num_parameters_token_out = sum(p.numel() for p in self.token_out.parameters())
        self.num_parameters = (self.num_parameters_per_block * depth) + self.num_parameters_layer_norm_last + (self.num_parameters_token_in + self.num_parameters_token_out)

        for i, block in enumerate(self.block_list):
            self.block_list[i] = block.to(devices[self.device_index(i)])

    def device_index(self, i):
        return (int)((len(self.devices) * (i * self.num_parameters_per_block + self.num_parameters_token_in)) / self.num_parameters)

    def forward(self, x):
        x = self.token_in(x)
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
        
    