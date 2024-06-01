import torch
import torch.nn as nn
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
    def __init__(self, dim: int, num_head: int, dtype):
        super().__init__()
        assert dim % num_head == 0, 'dim must be multiple of num_head'
        self.dim = dim
        self.num_head = num_head
        self.fc_phazor_angle = nn.Linear(dim, num_head)
        self.fc_phazor_abs = nn.Linear(dim, num_head)
        self.fc_z= nn.Linear(dim, dim)
        self.fc_z_act = nn.Linear(dim, dim)
        self.fc_y = nn.Linear(dim, dim)
        self.fc_y_act = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.phazor_angle_scale = nn.Parameter(1e-3 ** torch.linspace(0, 1, num_head), requires_grad=False)
        self.last_conv = None # (batch, dim)
        self.last_conv_init = nn.Parameter(torch.randn(num_head, dim//num_head, dtype=torch.cfloat))
        self.norm = nn.GroupNorm(num_head, num_head)
        self.is_refresh = True
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc_phazor_angle.weight, gain=1e-2)
        nn.init.zeros_(self.fc_phazor_angle.bias)
        nn.init.xavier_normal_(self.fc_phazor_abs.weight, gain=1e-2)
        nn.init.zeros_(self.fc_phazor_abs.bias)
        nn.init.xavier_normal_(self.fc_z.weight, gain=1e-2)
        nn.init.zeros_(self.fc_z.bias)
        nn.init.xavier_normal_(self.fc_z_act.weight, gain=1e-2)
        nn.init.zeros_(self.fc_z_act.bias)
        nn.init.xavier_normal_(self.fc_y.weight, gain=1e-2)
        nn.init.zeros_(self.fc_y.bias)
        nn.init.xavier_normal_(self.fc_y_act.weight, gain=1e-2)
        nn.init.zeros_(self.fc_y_act.bias)

    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        dim = x.shape[2]
        dtype = x.dtype
        num_head = self.num_head
        inner_dim = dim//num_head

        x = x.to(torch.float)
        z = (self.fc_z(x) * self.act(self.fc_z_act(x))).cfloat().view(batch, len, num_head, inner_dim) # (batch, len, num_head, inner_dim)

        if self.last_conv is None:
            self.last_conv = einops.repeat(self.last_conv_init, "n i -> b n i", b=batch) # (batch, num_head, inner_dim)
        else:
            self.last_conv = self.last_conv.detach()

        ones = torch.ones(len, device=x.device)
        ones_fft = torch.fft.fft(ones, n=len*2)

        ln_phazor = (self.act(self.fc_phazor_angle(x)) * 1j - nn.functional.sigmoid(self.fc_phazor_abs(x))) * self.phazor_angle_scale # (batch, len, num_head)
        ln_phazor_masked = einops.repeat(ln_phazor, "b l h ->b l m h", m=len).tril(-1) # (batch, len, len, num_head)
        ln_phazor_masked_fft = torch.fft.fft(ln_phazor_masked, n=len*2, dim=1) # (batch, len, len, num_head)
        ln_phazor_masked_conv = torch.fft.ifft(torch.einsum("blmh,l->blmh", ln_phazor_masked_fft, ones_fft), dim=1).narrow(1,0,len) # (batch, len, len, num_head)
        phazor_masked_conv = torch.exp(ln_phazor_masked_conv).tril() # (batch, len, len, num_head)

        h_inner_chunk = torch.einsum("blmh,bmhi->blhi", phazor_masked_conv, z)

        ln_phazor_fft = torch.fft.fft(ln_phazor, n=len*2, dim=1)
        ln_phazor_conv = torch.fft.ifft(torch.einsum("blh,l->blh", ln_phazor_fft, ones_fft), dim=1).narrow(1,0,len) # (batch, len, num_head)
        
        h_cross_chunk = torch.einsum("blh,bhi->blhi", torch.exp(ln_phazor_conv), self.last_conv)

        h = h_inner_chunk + h_cross_chunk

        if self.is_refresh:
            self.last_conv = h[:,-1,:,:]

        h_norm = self.norm(h.real.view(batch*len, num_head, inner_dim)).view(batch, len, dim)
        y = self.fc_y(h_norm) * self.act(self.fc_y_act(x))
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
    def __init__(self, dim: int, num_head: int, dim_ff_hidden: int, dropout: float, dtype):
        super().__init__()
        self.dtype = dtype 
        self.sconv = SConvLayer(dim, num_head, dtype)
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
        self.block_list = nn.ModuleList([SConvBlock(dim, num_head, dim_ff_hidden, dropout, dtype) for _ in range(depth)])
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
        
    