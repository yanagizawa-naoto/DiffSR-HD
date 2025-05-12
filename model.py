# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ——— Positional (sin/cos) time embedding ———
def sinusoidal_embedding(t: torch.Tensor, dim: int):
    """
    t: (batch,)
    returns: (batch, dim)
    """
    half = dim // 2
    device = t.device
    exponents = torch.arange(half, device=device, dtype=torch.float32) / half
    freqs = torch.exp(-math.log(10000) * exponents)            # (half,)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)         # (batch, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, 2*half)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb  # (batch, dim)

# ——— Residual block with time embedding ———
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, in_ch)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.nin_shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # add time embedding
        t = self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)  # (batch, in_ch, 1, 1)
        h = h + t
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.nin_shortcut(x)

# ——— Self-attention at lowest resolution ———
class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.norm = nn.GroupNorm(32, channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        h_ = self.norm(x)
        qkv = self.qkv(h_)  # (b, 3c, h, w)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        # reshape for attention
        hd = c // self.num_heads
        N  = h * w
        q = q.view(b, self.num_heads, hd, N).permute(0,1,3,2)  # (b,heads,N,hd)
        k = k.view(b, self.num_heads, hd, N)                 # (b,heads,hd,N)
        v = v.view(b, self.num_heads, hd, N).permute(0,1,3,2)  # (b,heads,N,hd)
        attn = torch.softmax(q @ k / math.sqrt(hd), dim=-1)   # (b,heads,N,N)
        out  = (attn @ v).permute(0,1,3,2).reshape(b, c, h, w) # (b,c,h,w)
        out  = self.proj(out)
        return x + out

# ——— Conditional UNet with skip connections ———
class ConditionalUNet(nn.Module):
    def __init__(self, base_ch=64, time_emb_dim=128):
        super().__init__()
        # time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim*4),
            nn.SiLU(),
            nn.Linear(time_emb_dim*4, time_emb_dim)
        )
        # initial conv: concat(LR up, y_t) => 6ch → base_ch
        self.init_conv = nn.Conv2d(6, base_ch, kernel_size=1)

        # down path
        self.res0 = ResBlock(base_ch, base_ch, time_emb_dim)  
        self.res1 = ResBlock(base_ch, base_ch, time_emb_dim)
        self.ds1  = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=2, padding=1)
        self.res2 = ResBlock(base_ch, base_ch, time_emb_dim)
        self.ds2  = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=2, padding=1)
        self.res3 = ResBlock(base_ch, base_ch, time_emb_dim)
        self.ds3  = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=2, padding=1)
        self.res4 = ResBlock(base_ch, base_ch, time_emb_dim)
        self.ds4  = nn.Conv2d(base_ch, base_ch, kernel_size=5, stride=5, padding=0)  # 240→48 / 135→27
        self.res5 = ResBlock(base_ch, base_ch, time_emb_dim)
        self.ds5  = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=3, padding=0)  # 48→16 / 27→9

        # bottom attention
        self.attn = SelfAttention(base_ch, num_heads=8)

        # up path (with skip concat)
        self.us5   = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=3, stride=3)
        self.resu5 = ResBlock(base_ch*2, base_ch, time_emb_dim)
        self.us4   = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=5, stride=5)
        self.resu4 = ResBlock(base_ch*2, base_ch, time_emb_dim)
        self.us3   = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resu3 = ResBlock(base_ch*2, base_ch, time_emb_dim)
        self.us2   = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resu2 = ResBlock(base_ch*2, base_ch, time_emb_dim)
        self.us1   = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resu1 = ResBlock(base_ch*2, base_ch, time_emb_dim)

        # final output
        self.final_norm = nn.GroupNorm(32, base_ch)
        self.final_act  = nn.SiLU()
        self.final_conv = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)

    def forward(self, lr_up: torch.Tensor, y_t: torch.Tensor, t: torch.Tensor):
        """
        lr_up, y_t: (b,3,1920,1080)
        t: (b,)
        """
        # time embedding
        te = sinusoidal_embedding(t, self.time_mlp[0].in_features)
        te = self.time_mlp(te)

        # initial conv
        x = torch.cat([lr_up, y_t], dim=1)  # (b,6,1920,1080)
        h = self.init_conv(x)               # (b,base_ch,1920,1080)

        # encoder
        h0 = self.res0(h, te)               # (b,base_ch,1920,1080)
        h1 = self.res1(h0, te)
        d1 = self.ds1(h1)                   # 960×540
        h2 = self.res2(d1, te)
        d2 = self.ds2(h2)                   # 480×270
        h3 = self.res3(d2, te)
        d3 = self.ds3(h3)                   # 240×135
        h4 = self.res4(d3, te)
        d4 = self.ds4(h4)                   # 48×27
        h5 = self.res5(d4, te)
        d5 = self.ds5(h5)                   # 16×9

        # bottleneck attention
        btm = self.attn(d5)                 # 16×9

        # decoder with skip connections
        u5 = self.us5(btm)                  # →48×27
        u5 = torch.cat([u5, h5], dim=1)     # (b,2*base_ch,48,27)
        r5 = self.resu5(u5, te)

        u4 = self.us4(r5)                   # →240×135
        u4 = torch.cat([u4, h4], dim=1)
        r4 = self.resu4(u4, te)

        u3 = self.us3(r4)                   # →480×270
        u3 = torch.cat([u3, h3], dim=1)
        r3 = self.resu3(u3, te)

        u2 = self.us2(r3)                   # →960×540
        u2 = torch.cat([u2, h2], dim=1)
        r2 = self.resu2(u2, te)

        u1 = self.us1(r2)                   # →1920×1080
        u1 = torch.cat([u1, h1], dim=1)
        r1 = self.resu1(u1, te)

        # final
        out = self.final_norm(r1)
        out = self.final_act(out)
        out = self.final_conv(out)          # (b,3,1920,1080)
        return out
