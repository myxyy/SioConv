import torch
import torch.nn as nn
import numpy as np

class FFN(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: float, dtype):
        super().__init__()
        self.fc_1 = nn.Linear(dim, dim_ff_hidden, bias=True, dtype=dtype)
        self.fc_2 = nn.Linear(dim_ff_hidden, dim, bias=True, dtype=dtype)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.fc_1(x)
        x = self.act(x)
        x = self.fc_2(x)
        return x

class SConvLayer(nn.Module):
    def __init__(self, dim: int, dtype):
        super().__init__()
        self.dim = dim
        self.fc_phazor_angle = nn.Linear(dim, dim)
        self.phazor_angle_scale = nn.Parameter(1e-3 ** torch.linspace(0, 1, dim), requires_grad=False)
        self.last_conv = None # (batch, dim)
        self.last_conv_init = nn.Parameter(torch.randn(dim, dtype=torch.cfloat))
        self.layer_norm = nn.LayerNorm(dim)
        self.is_refresh = True

    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        dim = x.shape[2]
        dtype = x.dtype

        x = x.to(torch.float)

        if self.last_conv is None:
            self.last_conv = self.last_conv_init.unsqueeze(0).expand(batch, dim)
        else:
            self.last_conv = self.last_conv.detach()

        ones_fft = torch.fft.fft(torch.ones(len, device=x.device), n=len*2)

        ln_phazor = self.fc_phazor_angle(x) * 1j * self.phazor_angle_scale - 1e-3 # (batch, len, dim)
        ln_phazor_mask = torch.ones(len, device=x.device)
        ln_phazor_mask[0] = 0
        ln_phazor_masked = torch.einsum("bld,l->bld", ln_phazor, ln_phazor_mask)
        ln_phazor_masked_fft = torch.fft.fft(ln_phazor_masked, n=len*2, dim=1)
        ln_phazor_masked_conv = torch.fft.ifft(torch.einsum("bld,l->bld", ln_phazor_masked_fft, ones_fft), dim=1).narrow(1,0,len)
        phazor_masked_row = torch.exp(ln_phazor_masked_conv) # (batch, len, dim)
        phazor_masked_col = 1/phazor_masked_row # (batch, len, dim)
        tri_mask = torch.ones(len, len, device=x.device, dtype=torch.cfloat).tril() # (len, len)

        h_inner_chunk = torch.einsum("bld,bld->bld", phazor_masked_row, torch.einsum("lm,bmd->bld", tri_mask, phazor_masked_col * x.cfloat()))

        ln_phazor_fft = torch.fft.fft(ln_phazor, n=len*2, dim=1)
        ln_phazor_conv = torch.fft.ifft(torch.einsum("bld,l->bld", ln_phazor_fft, ones_fft), dim=1).narrow(1,0,len)
        
        h_cross_chunk = torch.einsum("bld,bd->bld", torch.exp(ln_phazor_conv), self.last_conv)

        h = h_inner_chunk + h_cross_chunk

        if self.is_refresh:
            self.last_conv = h[:,-1,:]

        y = self.layer_norm(h.real)
        return y.to(dtype)

    def reset_hidden(self):
        self.last_conv = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

    def get_hidden(self):
        return self.last_conv

    def set_hidden(self, hidden):
        self.last_conv = hidden
 

class SConvBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, dropout: float, dtype):
        super().__init__()
        self.dtype = dtype 
        self.spiral_conv = SConvLayer(dim, dtype)
        self.ffn_sc = FFN(dim, dim_ff_hidden, dtype)
        self.layer_norm_sc_in = nn.LayerNorm(dim, elementwise_affine=True, bias=True, dtype=dtype)
        self.layer_norm_ffn_sc_in = nn.LayerNorm(dim, elementwise_affine=True, bias=True, dtype=dtype)
        self.act = nn.SiLU()
        self.fc = nn.Linear(dim, dim, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.layer_norm_sc_in(x)
        y = self.fc(x)
        y = self.act(y)
        x = self.spiral_conv(x)
        x = x * y
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.layer_norm_ffn_sc_in(x)
        x = self.ffn_sc(x)
        x = self.dropout(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.spiral_conv.reset_hidden()

    def set_is_refresh(self, is_refresh):
        self.spiral_conv.set_is_refresh(is_refresh)

    def get_hidden(self):
        return self.spiral_conv.get_hidden()

    def set_hidden(self, hidden):
        self.spiral_conv.set_hidden(hidden)

class SConv(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        dim_ff_hidden: int,
        dropout: float,
        vocab_size: int,
        devices,
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
        self.block_list = nn.ModuleList([SConvBlock(dim, dim_ff_hidden, dropout, dtype) for _ in range(depth)])
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
        
    