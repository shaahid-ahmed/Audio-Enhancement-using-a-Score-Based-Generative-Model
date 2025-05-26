import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = x.view(batch, channels, height * width).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(batch, channels, height, width)
    
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        # Upsampling layer
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Convolution layers (DoubleConv blocks)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        # Embedding layer for time conditioning
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        # Apply upsampling to the input tensor
        x = self.up(x)

        # Get spatial dimensions of both tensors
        diff_y = skip_x.size(2) - x.size(2)  # Difference in height
        diff_x = skip_x.size(3) - x.size(3)  # Difference in width

        # Apply padding if necessary (pad right and bottom)
        if diff_y > 0 or diff_x > 0:
            x = F.pad(x, (0, diff_x, 0, diff_y))  # Pad (left, right, top, bottom)

        # Concatenate the skip connection with the upsampled tensor
        x = torch.cat([skip_x, x], dim=1)

        # Apply the convolutions
        x = self.conv(x)

        # Apply time embedding and return the final output
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        # Encoder
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, time_dim)
        self.down2 = Down(128, 256, time_dim)
        self.down3 = Down(256, 512, time_dim)
        self.down4 = Down(512, 1024, time_dim)  
        self.down5 = Down(1024, 2048, time_dim)  
        
        # Bottleneck with more convolutions
        self.bot1 = DoubleConv(2048, 2048)
        self.bot2 = DoubleConv(2048, 2048)
        self.bot3 = DoubleConv(2048, 1024)  
        
        # Decoder with expanded feature maps
        self.up1 = Up(2048, 1024, time_dim)
        self.up2 = Up(1024, 512, time_dim)
        self.up3 = Up(512, 256, time_dim)
        self.up4 = Up(256, 128, time_dim)
        self.up5 = Up(128, 64, time_dim)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        t = t.view(-1, 1)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x5 = self.down4(x4, t)
        x6 = self.down5(x5, t)

        # Bottleneck
        x6 = self.bot1(x6)
        x6 = self.bot2(x6)
        x6 = self.bot3(x6)

        # Decoder
        x = self.up1(x6, x5, t)
        x = self.up2(x, x4, t)
        x = self.up3(x, x3, t)
        x = self.up4(x, x2, t)
        x = self.up5(x, x1, t)
        output = self.outc(x)
        return output


# Example Usage
# batch_size = 1
# image_height = 72
# image_width = 48
# c_in = 2
# time_dim = 256

# sample_input = torch.randn(batch_size, c_in, image_height, image_width)
# sample_time = torch.randn(batch_size, 1)
# device = "cuda:1"
# unet_model = UNet(c_in=c_in, c_out=1, time_dim=time_dim).to(device)
# output = unet_model(sample_input, sample_time)
# summary(unet_model, input_size=[(batch_size, c_in, image_height, image_width), (batch_size, time_dim)])
# print("Output shape:", output.shape)
