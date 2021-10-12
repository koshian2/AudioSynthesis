
import torch
from torch.nn.utils import weight_norm
import torchlibrosa

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

class TADELayer(torch.nn.Module):
    """TADE Layer module."""

    def __init__(
        self,
        in_channels: int,
        aux_channels: int,
        kernel_size: int = 9,
        bias: bool = True,
        upsample_factor: int = 2,
        upsample_mode: str = "nearest",
    ):
        """Initilize TADELayer module.
        Args:
            in_channels (int): Number of input channles.
            aux_channels (int): Number of auxirialy channles.
            kernel_size (int): Kernel size.
            bias (bool): Whether to use bias parameter in conv.
            upsample_factor (int): Upsample factor.
            upsample_mode (str): Upsample mode.
        """
        super().__init__()
        self.norm = torch.nn.InstanceNorm1d(in_channels)
        self.gated_conv = weight_norm(torch.nn.Conv1d(
                aux_channels,
                in_channels * 2,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,                
        ))
        self.upsample = torch.nn.Upsample(
            scale_factor=upsample_factor, mode=upsample_mode
        )

        self.gated_conv.apply(init_weights)


    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Auxiliary input tensor (B, aux_channels, T').
        Returns:
            Tensor: Output tensor (B, in_channels, T * in_upsample_factor).
            Tensor: Upsampled aux tensor (B, in_channels, T * aux_upsample_factor).
        """
        x = self.norm(x)
        cg = self.gated_conv(c)
        cg1, cg2 = cg.split(cg.size(1) // 2, dim=1)
        y = cg1 * self.upsample(x) + cg2
        return y, c

class TADEResBlock(torch.nn.Module):
    """TADEResBlock module."""

    def __init__(
        self,
        in_channels,
        aux_channels: int = 45,
        kernel_size: int = 9,
        dilation: int = 2,
        bias: bool = True,
        upsample_factor: int = 2,
        upsample_mode: str = "nearest"
    ):
        """Initialize TADEResBlock module.
        Args:
            in_channels (int): Number of input channles.
            aux_channels (int): Number of auxirialy channles.
            kernel_size (int): Kernel size.
            bias (bool): Whether to use bias parameter in conv.
            upsample_factor (int): Upsample factor.
            upsample_mode (str): Upsample mode.
            gated_function (str): Gated function type (softmax of sigmoid).
        """
        super().__init__()
        self.tade1 = TADELayer(
            in_channels=in_channels,
            aux_channels=aux_channels,
            kernel_size=kernel_size,
            bias=bias,
            upsample_factor=1,
            upsample_mode=upsample_mode,
        )
        self.gated_conv1 = weight_norm(torch.nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        ))
        self.tade2 = TADELayer(
            in_channels=in_channels,
            aux_channels=aux_channels,
            kernel_size=kernel_size,
            bias=bias,
            upsample_factor=upsample_factor,
            upsample_mode=upsample_mode,
        )
        self.gated_conv2 = weight_norm(torch.nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias=bias,
            dilation=dilation,
            padding=(kernel_size - 1) // 2 * dilation,
        ))
        self.upsample = torch.nn.Upsample(
            scale_factor=upsample_factor, mode=upsample_mode
        )

        self.gated_conv1.apply(init_weights)
        self.gated_conv2.apply(init_weights)

    def forward(self, x: torch.Tensor, 
                c_low_res: torch.Tensor, c_high_res : torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c_low_res (Tensor): Auxiliary input tensor (B, aux_channels, T').
            c_high_res (Tensor): Auxiliary input tensor (B, aux_channels, upsample*T').
        Returns:
            Tensor: Output tensor (B, in_channels, T * in_upsample_factor).
        """
        residual = x

        x, c = self.tade1(x, c_low_res)
        x = self.gated_conv1(x)
        xa, xb = x.split(x.size(1) // 2, dim=1)
        x = torch.softmax(xa, dim=1) * torch.tanh(xb)

        x, c = self.tade2(x, c_high_res)
        x = self.gated_conv2(x)
        xa, xb = x.split(x.size(1) // 2, dim=1)
        x = torch.softmax(xa, dim=1) * torch.tanh(xb)

        return self.upsample(residual) + x

class MoraEncoder(torch.nn.Module):
    def __init__(self, 
                h,
                device_x, device_y):
        super().__init__()
        self.h = h
        self.device_x = device_x
        self.device_y = device_y 

    def forward(self, use_sentences, pad_pre=None, pad_post=None):
        ## input(List<Sentence>)

        moras, sound = [], []
        # 無音区間のPad
        if pad_pre is not None:
            x = torch.zeros((self.h.n_moras, pad_pre), device=self.device_x)
            y = torch.zeros((pad_pre), device=self.device_y)
            moras.append(x)
            sound.append(y)

        # センテンスの列
        for s in use_sentences:
            x = torch.zeros(s.sparse_size, device=self.device_x)
            for mora_idx, time_idx, duration, pitch in zip(
                s.sparse_mora_indices, s.sparse_time_indices, s.sparse_duration, s.sparse_values):
                x[mora_idx, time_idx:(time_idx+duration)] = pitch
            moras.append(x)
            sound.append(torch.FloatTensor(s.sound_data).to(self.device_y))

        # 無音区間
        if pad_post is not None:
            x = torch.zeros((self.h.n_moras, pad_post), device=self.device_x)
            y = torch.zeros((pad_post), device=self.device_y)
            moras.append(x)
            sound.append(y)

        moras = torch.cat(moras, dim=1).unsqueeze(0) # (1, n_moras, T)
        sound = torch.cat(sound).unsqueeze(0).unsqueeze(0) # (1, 1, T)

        T = min(moras.size(-1), sound.size(-1))
        return moras[...,:T], sound[...,:T]

class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.extractor1 = torchlibrosa.Spectrogram(
            n_fft=h.n_fft, hop_length=h.hop_size, win_length=h.win_size)
        self.extractor2 = torchlibrosa.LogmelFilterBank(
            sr=h.sampling_rate, n_fft=h.n_fft, n_mels=h.num_mels, is_log=True)

    def forward(self, x):
        # input (batch, 1, frames)
        x = self.extractor1(x.squeeze(1))
        x = self.extractor2(x)
        # log_spec : (batch, 1, time_steps, freq_bins)：完全な[-80, 0]のスケールではない。プラスにブレる
        log_spec = x.squeeze(1).swapaxes(1,2)
        return log_spec # (batch,  num_mels, )
