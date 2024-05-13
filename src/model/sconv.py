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
    def __init__(self, dim: int, dtype, phazor_scale_start: float=32, phazor_scale_end: float=2048):
        super().__init__()
        self.dim = dim
        self.phazor_init = nn.Parameter(torch.randn(dim, dtype=torch.cfloat))
        self.phazor_angle = nn.Parameter(torch.randn(dim))
        self.ln_minus_ln_phazor_abs = nn.Parameter(torch.randn(dim))
        self.ln_phazor_scale = nn.Parameter(torch.exp(torch.linspace(np.log(1/phazor_scale_start), np.log(1/phazor_scale_end), dim)), requires_grad=False)
        self.last_conv = None # (batch, dim)
        self.last_conv_init = nn.Parameter(torch.randn(dim, dtype=torch.cfloat))
        self.is_refresh = True

    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        dtype = x.dtype

        x = x.to(torch.cfloat)
        if self.last_conv is None:
            self.last_conv = self.last_conv_init.unsqueeze(0).expand(batch, self.dim)
        else:
            self.last_conv = self.last_conv.detach()
        phazor = torch.exp((self.phazor_angle * 1j - torch.exp(self.ln_minus_ln_phazor_abs)) * self.ln_phazor_scale)
        phazor_progression = torch.pow(phazor.unsqueeze(0), torch.arange(len, device=x.device).unsqueeze(1)) # (len, dim)
        filter = phazor_progression * self.phazor_init.unsqueeze(0)
        filter_fft = torch.fft.fft(filter, n=len*2, dim=0) # (len*2, dim)
        x_fft = torch.fft.fft(x, n=len*2, dim=1) # (batch, len*2, dim)
        conv_filter_x = torch.fft.ifft(filter_fft.unsqueeze(0) * x_fft, dim=1).narrow(1,0,len) # (batch, len, dim)
        conv_with_past = conv_filter_x + self.last_conv.unsqueeze(1)*phazor_progression.unsqueeze(0)*phazor.unsqueeze(0).unsqueeze(0)
        if self.is_refresh:
            self.last_conv = conv_with_past[:,-1,:]

        y = conv_with_past * (1 - torch.exp(- torch.exp(self.ln_minus_ln_phazor_abs) * self.ln_phazor_scale))
        y = y.real.to(dtype)
        return y

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
        
    