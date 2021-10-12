import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Any, Optional
from torch.nn.utils import weight_norm
from style_melgan.style_melgan_layers import TADEResBlock, init_weights
from style_melgan.pqmf import PQMF

class StyleMelGANGenerator(torch.nn.Module):
    """Style MelGAN generator module."""

    def __init__(self, h):
        super().__init__()
        self.h = h

        self.stem = weight_norm(torch.nn.Conv1d(
            h.gen.aux_channels, h.gen.n_channels, h.gen.kernel_size, 1, padding=(h.gen.kernel_size-1)//2)
        )

        self.blocks = torch.nn.ModuleList()
        for i, upsample_scale in enumerate(h.gen.upsample_scales):
            self.blocks += [
                TADEResBlock(
                    in_channels=h.gen.n_channels,
                    aux_channels=h.gen.aux_channels,
                    kernel_size=h.gen.kernel_size,
                    dilation=h.gen.dilation,
                    bias=h.gen.bias,
                    upsample_factor=upsample_scale,
                    upsample_mode=h.gen.upsample_mode,
                ),
            ]

        self.output_conv = weight_norm(torch.nn.Conv1d(
                h.gen.n_channels,
                h.gen.out_channels,
                h.gen.kernel_size,
                1,
                bias=h.gen.bias,
                padding=(h.gen.kernel_size - 1) // 2,
        ))

    def forward(
        self, m
    ) -> torch.Tensor:
        """Calculate forward propagation.
        Args:
            m (Tensor):  (B, n_mora, T)                           
        Returns:
            Tensor: generated sound (B, out_channels, T).
        """
        # encode all moras
        encoded_mora = [m]

        for i in range(1, len(self.h.gen.upsample_scales)):
            rev_i = len(self.h.gen.upsample_scales) - i - 1
            m = F.avg_pool1d(m, self.h.gen.upsample_scales[rev_i])
            encoded_mora.append(m)

        encoded_mora = encoded_mora[::-1]

        x = self.stem(encoded_mora[0])
        for i in range(len(self.blocks)):
            if i < len(self.blocks) - 1:
                x_low_res, x_high_res = encoded_mora[i], encoded_mora[i+1]
            else:
                x_low_res, x_high_res = encoded_mora[i], encoded_mora[i]
            ups = self.h.gen.upsample_scales[i]
            
            L = min(x.size(2), x_low_res.size(2), x_high_res.size(2)//ups)
            x = self.blocks[i](x[...,:L], x_low_res[...,:L], x_high_res[...,:ups*L])            

        x = self.output_conv(x)
        x = torch.tanh(x)
        return x

class BaseDiscriminator(torch.nn.Module):
    """MelGAN discriminator module."""

    def __init__(
        self,
        h,
        in_channels,
    ):
        """Initilize MelGANDiscriminator module.
        Args:
            h (AttrDict): Setting file
            in_channels (int): Number of input channels.
        """
        super().__init__()
        self.h = h
        self.layers = torch.nn.ModuleList()

        # add first layer
        self.layers += [
            torch.nn.ReflectionPad1d((np.prod(h.dis.kernel_sizes) - 1) // 2),
            weight_norm(torch.nn.Conv1d(
                in_channels, h.dis.channels, np.prod(h.dis.kernel_sizes), bias=h.dis.bias
            )),
            torch.nn.LeakyReLU(negative_slope=0.2),
        ]

        # add downsample layers
        in_chs = h.dis.channels
        for downsample_scale in h.dis.downsample_scales:
            out_chs = min(in_chs * downsample_scale, h.dis.max_downsample_channels)
            self.layers += [
                weight_norm(torch.nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_size=downsample_scale * 10 + 1,
                    stride=downsample_scale,
                    padding=downsample_scale * 5,
                    groups=in_chs // 4,
                    bias=h.dis.bias,
                )),
                torch.nn.LeakyReLU(negative_slope=0.2),
            ]
            in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, h.dis.max_downsample_channels)
        self.layers += [
            weight_norm(torch.nn.Conv1d(
                in_chs,
                out_chs,
                h.dis.kernel_sizes[0],
                padding=(h.dis.kernel_sizes[0] - 1) // 2,
                bias=h.dis.bias,
            )),
            torch.nn.LeakyReLU(negative_slope=0.2),
        ]
        self.layers += [
            weight_norm(torch.nn.Conv1d(
                out_chs,
                h.dis.out_channels,
                h.dis.kernel_sizes[1],
                padding=(h.dis.kernel_sizes[1] - 1) // 2,
                bias=h.dis.bias,
            )),
        ]

        self.layers.apply(init_weights)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            y[Tensor]: Output tensor of final layer.
        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]

        return outs

class StyleMelGANDiscriminator(torch.nn.Module):
    """Style MelGAN disciminator module."""

    def __init__(
        self,
        h
    ):
        """Initilize StyleMelGANDiscriminator module.
        Args:
            h (AttrDict): Setting of StyleMelGAN
        """
        super().__init__()

        self.h = h
        self.pqmfs = torch.nn.ModuleList()
        self.discriminators = torch.nn.ModuleList()

        for pqmf_param in h.dis.pqmf_params:
            in_channels = pqmf_param[0]
            if pqmf_param[0] == 1:
                self.pqmfs += [torch.nn.Identity()]
            else:
                self.pqmfs += [PQMF(*pqmf_param)]
            self.discriminators += [BaseDiscriminator(h, in_channels)]

    def forward(self, 
        x_real: torch.Tensor, x_gen: torch.Tensor):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, 1, T).
        Returns:
            List: List of discriminator outputs, #items in the list will be
                equal to h.repeats * #discriminators.
        """

        sample_size = min(x_real.size(-1), x_gen.size(-1))
        phase_max = min(self.h.dis.phase_max, sample_size//self.h.dis.window_sizes[-1])

        outs_real, outs_gen = [], []
        for _ in range(self.h.dis.repeats):
            phase = random.randint(self.h.dis.phase_min, phase_max)
            n = sample_size // phase

            xr = x_real[...,:phase*n].view(1, phase, n).permute(1, 0, 2) # [phase, 1, T'(=T/phase)]
            xg = x_gen[...,:phase*n].view(1, phase, n).permute(1, 0, 2)

            r, g = self._forward(xr, xg)
            outs_real += r
            outs_gen += g

        return outs_real, outs_gen

    def _forward(self, x_real: torch.Tensor, x_gen: torch.Tensor):
        outs_real = []
        outs_gen = []
        for idx, (ws, pqmf, disc) in enumerate(
            zip(self.h.dis.window_sizes, self.pqmfs, self.discriminators)
        ):
            # NOTE(kan-bayashi): Is it ok to apply different window for real and fake
            #   samples?
            start_idx = np.random.randint(min(x_real.size(-1), x_gen.size(-1)) - ws)
            x_real_ = x_real[:, :, start_idx : start_idx + ws]
            x_gen_ = x_gen[:, :, start_idx : start_idx + ws]
            if idx == 0:
                x_real_ = pqmf(x_real_) # Identity
                x_gen_ = pqmf(x_gen_) # Identity
            else:
                x_real_ = pqmf.analysis(x_real_)
                x_gen_ = pqmf.analysis(x_gen_)
            outs_real += [disc(x_real_)]
            outs_gen += [disc(x_gen_)]
        return outs_real, outs_gen # [repeats*n_dis, n_hidden]

class SpectrogramDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            weight_norm(torch.nn.Conv2d(1, 32, (5, 37), (2, 18), padding=(2, 18))),
            weight_norm(torch.nn.Conv2d(32, 64, 3, 2, padding=1)),
            weight_norm(torch.nn.Conv2d(64, 128, 3, 2, padding=1)),
            weight_norm(torch.nn.Conv2d(128, 256, 3, 2, padding=1)),
            weight_norm(torch.nn.Conv2d(256, 512, 3, 2, padding=1)),
        ])
        self.conv_post = weight_norm(torch.nn.Conv2d(512, 1, 3, 1, padding=1))

        self.convs.apply(init_weights)
        self.conv_post.apply(init_weights)

    def _forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        y_d_r, fmap_r = self._forward(y)
        y_d_g, fmap_g = self._forward(y_hat)
        y_d_rs.append(y_d_r)
        fmap_rs += fmap_r
        y_d_gs.append(y_d_g)
        fmap_gs += fmap_g

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs