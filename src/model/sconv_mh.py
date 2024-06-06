import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
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
    def __init__(self, dim: int, dim_ff_hidden: float, dtype):
        super().__init__()
        self.fc = nn.Linear(dim, dim_ff_hidden, dtype=dtype)
        self.fc_act = nn.Linear(dim, dim_ff_hidden, dtype=dtype)
        self.fc_out = nn.Linear(dim_ff_hidden, dim, dtype=dtype)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.fc(x) * self.act(self.fc_act(x))
        x = self.fc_out(x)
        return x

class SConvLayer(nn.Module):
    def __init__(self, dim: int, num_head: int, dtype, a_init_range=(1,16), dt_init_range=(0.001,0.1)):
        super().__init__()
        assert dim % num_head == 0, 'dim must be multiple of num_head'
        self.dim = dim
        self.num_head = num_head
        self.fc_z= nn.Linear(dim, dim)
        self.fc_z_act = nn.Linear(dim, dim)
        self.fc_y = nn.Linear(dim, dim)
        self.fc_y_act = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.ln_a = nn.Parameter(torch.log(torch.empty(num_head).uniform_(*a_init_range)))
        self.fc_dt = nn.Linear(dim, num_head)
        dt = torch.exp(torch.empty(num_head).uniform_(np.log(dt_init_range[0]), np.log(dt_init_range[1])))
        # inv_softplus_dt = torch.log(torch.exp(dt)-1) equals
        inv_softplus_dt = dt + torch.log(1-torch.exp(-dt))
        self.fc_dt.bias = nn.Parameter(inv_softplus_dt)
        self.last_conv = None # (batch, dim)
        self.last_conv_init = nn.Parameter(torch.randn(num_head, dim//num_head))
        self.norm = nn.GroupNorm(num_head, num_head)
        self.is_refresh = True
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc_dt.weight, gain=1e-2)
        nn.init.xavier_normal_(self.fc_z.weight, gain=1e-2)
        nn.init.zeros_(self.fc_z.bias)
        nn.init.xavier_normal_(self.fc_z_act.weight, gain=1e-2)
        nn.init.zeros_(self.fc_z_act.bias)
        nn.init.xavier_normal_(self.fc_y.weight, gain=1e-2)
        nn.init.zeros_(self.fc_y.bias)
        nn.init.xavier_normal_(self.fc_y_act.weight, gain=1e-2)
        nn.init.zeros_(self.fc_y_act.bias)

    # (batch, len, dim), (batch, num_head, inner_dim) -> (batch, len, dim), (batch, num_head, inner_dim)
    def forward(self, x, hidden):
        batch = x.shape[0]
        len = x.shape[1]
        dim = x.shape[2]
        dtype = x.dtype
        num_head = self.num_head
        inner_dim = dim//num_head

        x = x.to(torch.float)
        z = (self.fc_z(x) * self.act(self.fc_z_act(x))).view(batch, len, num_head, inner_dim) # (batch, len, num_head, inner_dim)

        ones = torch.ones(len, device=x.device)
        ones_fft = torch.fft.rfft(ones, n=len*2)

        ln_da = - torch.exp(self.ln_a) * F.softplus(self.fc_dt(x)) # (batch, len, num_head)
        ln_da_masked = einops.repeat(ln_da, "b l h ->b l m h", m=len).tril(-1) # (batch, len, len, num_head)
        ln_da_masked_fft = torch.fft.rfft(ln_da_masked, n=len*2, dim=1) # (batch, len, len, num_head)
        ln_da_masked_conv = torch.fft.irfft(torch.einsum("blmh,l->blmh", ln_da_masked_fft, ones_fft), dim=1).narrow(1,0,len) # (batch, len, len, num_head)
        da_masked_conv = torch.exp(ln_da_masked_conv).tril() # (batch, len, len, num_head)

        h_inner_chunk = torch.einsum("blmh,bmhi->blhi", da_masked_conv, z)

        ln_da_fft = torch.fft.rfft(ln_da, n=len*2, dim=1)
        ln_da_conv = torch.fft.irfft(torch.einsum("blh,l->blh", ln_da_fft, ones_fft), dim=1).narrow(1,0,len) # (batch, len, num_head)
        
        h_cross_chunk = torch.einsum("blh,bhi->blhi", torch.exp(ln_da_conv), hidden)

        h = h_inner_chunk + h_cross_chunk

        hidden_next = h[:,-1,:,:]

        h_norm = self.norm(h.view(batch*len, num_head, inner_dim)).view(batch, len, dim)
        y = self.fc_y(h_norm) * self.act(self.fc_y_act(x))
        return y.to(dtype), hidden_next

class ChunkWiseSConvLayer(nn.Module):
    def __init__(self, dim: int, num_head: int, chunk_size: int, dtype):
        super().__init__()
        self.sconv = SConvLayer(dim, num_head, dtype)
        self.last_hidden = None
        self.last_hidden_init = nn.Parameter(torch.randn(num_head, dim//num_head))
        self.is_refresh = True
        self.dim = dim
        self.num_head = num_head
        self.chunk_size = chunk_size

    def forward(self, x):
        batch = x.shape[0]
        num_head = self.num_head
        dim = self.dim

        if self.last_hidden is None:
            hidden = self.last_hidden_init.unsqueeze(0).expand(batch, num_head, dim//num_head)
        else:
            hidden = self.last_hidden.detach()

        input_chunks = x.split(self.chunk_size, dim=1)
        output_chunks = []
        for input_chunk in input_chunks:
            output_chunk, hidden = self.sconv(input_chunk, hidden)
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

class SConvBlock(nn.Module):
    def __init__(self, dim: int, num_head: int, dim_ff_hidden: int, dropout: float, chunk_size: int, dtype):
        super().__init__()
        self.dtype = dtype 
        self.sconv = ChunkWiseSConvLayer(dim, num_head, chunk_size, dtype)
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden, dtype)
        self.norm_sconv = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.norm_sconv(x)
        x = self.sconv(x)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.sconv.reset_hidden()

    def set_is_refresh(self, is_refresh):
        self.sconv.set_is_refresh(is_refresh)

    def get_hidden(self):
        return self.sconv.get_hidden()

    def set_hidden(self, hidden):
        self.sconv.set_hidden(hidden)

class SConvMH(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        num_head: int,
        dim_ff_hidden: int,
        dropout: float,
        vocab_size: int,
        devices,
        chunk_size: int=512,
        dtype=torch.float,
        token_in_out_parameter_corr = 3.0,
        out_only_device: bool=True,
    ):
        super().__init__()
        self.devices = devices
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.token_in = nn.Embedding(vocab_size, dim, device=devices[0], max_norm=1, dtype=dtype)
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1], dtype=dtype)
        self.block_list = nn.ModuleList([SConvBlock(dim, num_head, dim_ff_hidden, dropout, chunk_size, dtype) for _ in range(depth)])
        self.layer_norm_last = nn.LayerNorm(dim, elementwise_affine=True, bias=True, device=devices[-1], dtype=dtype)

        self.token_in_out_parameter_corr = token_in_out_parameter_corr
        self.num_parameters_token_in = sum(p.numel() for p in self.token_in.parameters())
        self.num_parameters_per_block = sum(p.numel() for p in self.block_list[0].parameters())
        self.num_parameters_layer_norm_last = sum(p.numel() for p in self.layer_norm_last.parameters())
        self.num_parameters_token_out = sum(p.numel() for p in self.token_out.parameters())
        self.num_parameters = (self.num_parameters_per_block * depth) + self.num_parameters_layer_norm_last + (self.num_parameters_token_in + self.num_parameters_token_out)
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
        
    