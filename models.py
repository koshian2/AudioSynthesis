import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, Conv2d, BatchNorm1d
from torch.nn.utils import weight_norm, spectral_norm
import numpy as np

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def trainable(m):
    for param in m.parameters():
        param.requires_grad = True

def not_trainable(m):
    for param in m.parameters():
        param.requires_grad = False 

class ConditionalBatchNorm(nn.Module):
    def __init__(self, channels, n_moras):
        super().__init__()
        self.norm = BatchNorm1d(channels)
        self.gamma = Conv1d(n_moras, channels, 7, 1, 3)
        self.beta = Conv1d(n_moras, channels, 7, 1, 3)
        self.gamma.apply(init_weights)
        self.beta.apply(init_weights)        

    def forward(self, x, c):
        x = self.norm(x)
        B, _, L = x.shape
        pc = F.pad(c, (0, L-c.shape[-1], 0, 0, 0, 0)) # 右側だけPad
        # pc = torch.cat([c, torch.zeros(L-c.shape[-1])], dim=-1)
        gamma = self.gamma(pc)
        beta = self.beta(pc)
        return gamma * x + beta

class ResBlock(torch.nn.Module):
    def __init__(self, channels, use_cbn, n_moras, kernel_size=3, dilation=(1, 3)):
        super(ResBlock, self).__init__()
        self.use_cbn = use_cbn
        self.convs = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0])),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))
        ])
        self.norms = nn.ModuleList([
            ConditionalBatchNorm(channels, n_moras) if use_cbn else
            BatchNorm1d(channels) for i in range(2)
        ])
        self.convs.apply(init_weights)

    def forward(self, x, z):
        xt = x
        for c, b in zip(self.convs, self.norms):
            if self.use_cbn:
                xt = b(xt, z)
            else:
                xt = b(xt)
            xt = F.relu(xt)
            xt = c(xt)
        return xt + x

class GeneratorUnit(torch.nn.Module):
    def __init__(self, h, n_phoneme_types):
        super(GeneratorUnit, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1d(h.in_channels//(2**i), h.in_channels//(2**(i+1)),
                                k, u, padding=(k-u)//2))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.in_channels//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(
                    ResBlock(ch, h.conditional_norms[i], n_phoneme_types, k, d))

        self.ups.apply(init_weights)

    def forward(self, x, moras):
        for i in range(self.num_upsamples):
            if i > 0:
                x = F.relu(x)
            x = self.ups[i](x)
            xs = None
            c = moras[i] if i < len(moras) else None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, c)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, c)
            x = xs / self.num_kernels
        return x
    
class Generator(nn.Module):
    def __init__(self, h, args):
        super().__init__()
        self.h = h
        self.args = args
        self.coarse_indices = len(h.coarse_model.upsample_rates)
        self.stem = Conv1d(h.n_phoneme_types, h.coarse_model.in_channels, 7, 1, padding=3)
        self.coarse_model = GeneratorUnit(h.coarse_model, h.n_phoneme_types)
        self.fine_model = GeneratorUnit(h.fine_model, h.n_phoneme_types)
        self.conv_post = Conv1d(h.conv_post_channels, 1, 7, 1, padding=3)

        self.stem.apply(init_weights)
        self.conv_post.apply(init_weights)

        if self.args.stage == 1:
            self.fine_model.apply(not_trainable)
            self.conv_post.apply(not_trainable)
        elif self.args.stage == 2:
            self.stem.apply(not_trainable)
            self.coarse_model.apply(not_trainable)

    def forward(self, x, moras):
        mora_coarse, mora_fine = moras[:self.coarse_indices], moras[self.coarse_indices:]
        # stem block
        x = self.stem(x)
        x = F.relu(x)
        # to mel-spectrogram : [0, 1]
        x = self.coarse_model(x, mora_coarse)
        mel = torch.tanh(x) * 0.5 + 0.5 # mel-spectogram [0-1] of shape(128, 18*in_ch)

        if self.args.stage >= 2:
            # to raw-wave
            x = self.fine_model(mel, mora_fine)
            x = F.relu(x)
            x = self.conv_post(x)
            x = torch.tanh(x)
            
            return mel, x
        else:
            return mel, None

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, lrelu_slope=0.1, norm="spectral"):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.lrelu_slope = 0.1
        norm_layer = spectral_norm if norm == "spectral" else weight_norm
        self.convs = nn.ModuleList([
            norm_layer(Conv2d(1, 32, (kernel_size*3, 1), (stride**2, 1), padding=(get_padding(15, 1), 0))),
            norm_layer(Conv2d(32, 64, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_layer(Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_layer(Conv2d(128, 256, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_layer(Conv2d(256, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_layer(Conv2d(512, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_layer(Conv2d(512, 512, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_layer(Conv2d(512, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, norm_layer="spectral"):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2, norm=norm_layer),
            DiscriminatorP(3, norm=norm_layer)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs += fmap_r
            y_d_gs.append(y_d_g)
            fmap_gs += fmap_g

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(torch.nn.Module):
    def __init__(self, lrelu_slope=0.1, norm="spectral"):
        super(DiscriminatorS, self).__init__()
        self.lrelu_slope = lrelu_slope
        norm_layer = spectral_norm if norm == "spectral" else weight_norm
        self.convs = nn.ModuleList([
            norm_layer(Conv1d(1, 128, 41, 8, padding=20)),
            norm_layer(Conv1d(128, 128, 41, 4, groups=4, padding=20)),
            norm_layer(Conv1d(128, 256, 41, 4, groups=16, padding=20)),
            norm_layer(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_layer(Conv1d(512, 512, 41, 4, groups=16, padding=20)),
            norm_layer(Conv1d(512, 512, 41, 4, groups=16, padding=20)),
            norm_layer(Conv1d(512, 512, 41, 2, groups=16, padding=20)),
            norm_layer(Conv1d(512, 512, 5, 2, padding=2)),
        ])
        self.conv_post = norm_layer(Conv1d(512, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

# 容量の節約のためにアンサンブルしない
class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, norm_layer="spectral"):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(norm=norm_layer),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs += fmap_r
            y_d_gs.append(y_d_g)
            fmap_gs += fmap_g

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class SpectrogramDiscriminator(nn.Module):
    def __init__(self, lrelu_slope=0.1, norm="spectral"):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        norm_layer = spectral_norm if norm == "spectral" else weight_norm
        self.convs = nn.ModuleList([
            norm_layer(Conv2d(1, 32, (5, 37), (2, 18), padding=(2, 18))),
            norm_layer(Conv2d(32, 64, 3, 2, padding=1)),
            norm_layer(Conv2d(64, 128, 3, 2, padding=1)),
            norm_layer(Conv2d(128, 256, 3, 2, padding=1)),
            norm_layer(Conv2d(256, 512, 3, 2, padding=1)),
        ])
        self.conv_post = norm_layer(Conv2d(512, 1, 3, 1, padding=1))

    def forward_single(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        y_d_r, fmap_r = self.forward_single(y)
        y_d_g, fmap_g = self.forward_single(y_hat)
        y_d_rs.append(y_d_r)
        fmap_rs += fmap_r
        y_d_gs.append(y_d_g)
        fmap_gs += fmap_g

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

if __name__ == "__main__":
    from test_code.model_test import check_generator
    check_generator(4)