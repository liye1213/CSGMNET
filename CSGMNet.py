import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from lib.pvtv2 import pvt_v2_b2

def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                        and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1)):
                    pass
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class LaplaceConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        laplace_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)
        laplace_kernel = laplace_kernel.unsqueeze(0).unsqueeze(0).repeat((out_channels, in_channels, 1, 1))
        self.conv.weight = nn.Parameter(laplace_kernel)
        self.conv.bias.data.fill_(0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.relu(self.bn(x1))
        return x1

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class Boundary_guided_module(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channel1, out_channel, 1)
        self.conv2 = nn.Conv2d(in_channel2, out_channel, 1)

    def forward(self, edge, semantic):
        x = self.conv1(edge)
        x, _ = torch.max(x, dim=1, keepdim=True)
        x = self.sigmoid(x)
        x = x * self.conv2(semantic)
        x = x + self.conv2(semantic)
        return x

class Long_distance(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = x + self.gamma * out
        return out

class Residual(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True, padding=1, relu=True, bias=True):
        super().__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.bn = nn.BatchNorm2d(out_dim) if bn else None

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, f"{x.size()[1]} {self.inp_dim}"
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class near_and_long(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.long = Long_distance(in_channel)
        self.near = Residual(in_channel, out_channel)
        self.conv1 = nn.Conv2d(in_channel + out_channel, out_channel, 1)

    def forward(self, x):
        x1 = self.long(x)
        x2 = self.near(x)
        fuse = torch.cat([x1, x2], 1)
        fuse = self.conv1(fuse)
        return fuse

class multi_scale_fuseion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.c1 = Conv(in_channel, out_channel, kernel_size=1, padding=0)
        self.c2 = Conv(in_channel, out_channel, kernel_size=3, padding=1)
        self.c3 = Conv(in_channel, out_channel, kernel_size=7, padding=3)
        self.c4 = Conv(in_channel, out_channel, kernel_size=11, padding=5)
        self.s1 = Conv(out_channel * 4, out_channel, kernel_size=1, padding=0)
        self.attention = CBAM(out_channel)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x5 = torch.cat([x1, x2, x3, x4], 1)
        x5 = self.s1(x5)
        x6 = self.attention(x5)
        return x6

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, height, width, groups, channels_per_group)
    x = torch.transpose(x, 3, 4).contiguous()
    x = x.view(batch_size, height, width, -1)
    return x


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    selective_scan_fn = None
    selective_scan_ref = None

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    selective_scan_fn_v1 = None
    selective_scan_ref_v1 = None

class SS2D(nn.Module):
    def __init__(self,
                 d_model,
                 d_state=16,
                 d_conv=3,
                 expand=2,
                 dt_rank="auto",
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 dropout=0.,
                 conv_bias=True,
                 bias=False,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = (self.d_model + 15) // 16 if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
                                bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs)
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min)))
                       + torch.log(torch.tensor(dt_min))).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).repeat(d_inner, 1).contiguous()
        if copies > 1:
            A = A.repeat(copies, 1)
        A_log = nn.Parameter(torch.log(A))
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = D.repeat(copies)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        assert selective_scan_fn is not None, "selective_scan_fn not available"
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, 2, 3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = selective_scan_fn(xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False).view(B, K, -1, L)
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, 1, 2).contiguous().view(B, H, W, -1)
        y = nn.LayerNorm(self.d_inner)(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class GMCM(nn.Module):

    class _ChannelOps:
        @staticmethod
        def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
            return x.permute(0, 2, 3, 1).contiguous()

        @staticmethod
        def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
            return x.permute(0, 3, 1, 2).contiguous()

        @staticmethod
        def shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
            b, h, w, c = x.size()
            pg = c // groups
            x = x.view(b, h, w, groups, pg)
            x = torch.transpose(x, 3, 4).contiguous()
            return x.view(b, h, w, -1)

    class _LocalConvBranch(nn.Module):
  
        def __init__(self, c: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.BatchNorm2d(c),
                nn.Conv2d(c, c, 3, padding=1, bias=True),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=True),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 1, bias=True),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class _GlobalMixer(nn.Module):

        def __init__(self, c_half: int, attn_drop_rate: float, d_state: int, drop_path: float):
            super().__init__()
            self.norm = nn.LayerNorm(c_half, eps=1e-6)
            self.ss2d = SS2D(d_model=c_half, dropout=attn_drop_rate, d_state=d_state)
            self.drop_path = DropPath(drop_path)

        def forward(self, x_nhwc_right: torch.Tensor) -> torch.Tensor:
            x = self.norm(x_nhwc_right)
            x = self.ss2d(x)
            return self.drop_path(x)

    def __init__(self,
                 hidden_dim: int = 0,
                 out_channels: int = 0,
                 drop_path: float = 0.0,
                 norm_layer=lambda **kwargs: nn.LayerNorm(eps=1e-6),  
                 attn_drop_rate: float = 0.0,
                 d_state: int = 16,
                 **kwargs):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be even for GMCM."
        self._ops = self._ChannelOps()
        c_half = hidden_dim // 2

        self.global_mixer = self._GlobalMixer(
            c_half=c_half, attn_drop_rate=attn_drop_rate, d_state=d_state, drop_path=drop_path
        )
        self.local_branch = self._LocalConvBranch(c=c_half)
        self.proj_out = nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, stride=1)
        self.register_buffer("_alpha", torch.tensor(1.0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_nhwc = self._ops.nchw_to_nhwc(x)
        left, right = x_nhwc.chunk(2, dim=-1)

        g = self.global_mixer(right)
        l = self._ops.nhwc_to_nchw(left)
        l = self.local_branch(l)
        l = self._ops.nchw_to_nhwc(l)

        y = torch.cat([l, g], dim=-1)
        y = self._ops.shuffle(y, groups=2)
        y = y + self._alpha * x_nhwc

        y = self._ops.nhwc_to_nchw(y)
        y = self.proj_out(y)
        return y


class SpatialTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class SpatialTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([SpatialTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        src = self.norm(src)
        return src


class CFIM(nn.Module):

    @staticmethod
    def _flatten_hw(x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, 'b c h w -> b (h w) c')

    @staticmethod
    def _unflatten_hw(x_seq: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        h, w = hw
        return rearrange(x_seq, 'b (h w) c -> b c h w', h=h, w=w)

    def __init__(self, channel_sizes, target_dim=256, nhead=8, depth=6):
        super().__init__()
        self.branches = [f"s{i+1}" for i in range(len(channel_sizes))]
        self.proj = nn.ModuleDict({
            name: nn.Conv2d(cin, target_dim, kernel_size=1, bias=True)
            for name, cin in zip(self.branches, channel_sizes)
        })
        self.encoder = SpatialTransformerEncoder(d_model=target_dim, nhead=nhead, num_layers=depth)
        self.reproj = nn.ModuleDict({
            name: nn.Conv2d(target_dim, cout, kernel_size=1, bias=True)
            for name, cout in zip(self.branches, channel_sizes)
        })
        self.target_dim = target_dim

    def forward(self, inputs):
        assert len(inputs) == len(self.branches), "Number of inputs must match CFIM branches."
        hw_list, seq_list = [], []
        for name, x in zip(self.branches, inputs):
            b, c, h, w = x.shape
            hw_list.append((h, w))
            x_proj = self.proj[name](x)
            x_seq = self._flatten_hw(x_proj)
            seq_list.append(x_seq)

        combined = torch.cat(seq_list, dim=1)
        encoded = self.encoder(combined)

        split_sizes = [h * w for (h, w) in hw_list]
        encoded_splits = torch.split(encoded, split_sizes, dim=1)

        outputs = []
        for (name, (h, w), x_encoded) in zip(self.branches, hw_list, encoded_splits):
            x_feat = self._unflatten_hw(x_encoded, (h, w))
            x_out = self.reproj[name](x_feat)
            outputs.append(x_out)
        return outputs

class Field(nn.Module):
    def __init__(self, channel=32, num_classes=2, drop_rate=0.4):
        super().__init__()
        self.drop = nn.Dropout2d(drop_rate)
        self.backbone = pvt_v2_b2()
        path = r"/home/lyaya/HBGNet-main/pvt_v2_b2.pth"
        save_model = torch.load(path, map_location="cpu")
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.edge_lap = LaplaceConv2d(in_channels=3, out_channels=1)
        self.conv1 = Residual(1, 32)
        self.conv2 = nn.Conv2d(64, 16, 1)
        self.attention2 = CBAM(64)

        self.fuse1 = near_and_long(512, 256)
        self.fuse2 = near_and_long(320, 128)
        self.fuse3 = near_and_long(128, 64)
        self.fuse4 = near_and_long(64, 32)

    
        self.gmcm1 = GMCM(512, 256)
        self.gmcm2 = GMCM(320, 128)
        self.gmcm3 = GMCM(128, 64)
        self.gmcm4 = GMCM(64, 32)

        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.boundary2 = Boundary_guided_module(64, 64, 16)
        self.boundary3 = Boundary_guided_module(64, 128, 16)
        self.boundary4 = Boundary_guided_module(64, 256, 16)

        self.multi_fusion = multi_scale_fuseion(64, 64)
        self.out_feature = nn.Conv2d(64, 1, 1)
        self.edge_feature = nn.Conv2d(64, num_classes, 1)
        self.dis_feature = nn.Conv2d(64, 1, 1)

   
        self.cfim = CFIM([64, 128, 320, 512])

    def forward(self, x):
        edge = self.edge_lap(x)
        edge = self.conv1(edge)
        pvt = self.backbone(x)
        x1 = self.drop(pvt[0])
        x2 = self.drop(pvt[1])
        x3 = self.drop(pvt[2])
        x4 = self.drop(pvt[3])

        fused_feats = self.cfim([x1, x2, x3, x4])
        x1, x2, x3, x4 = fused_feats

        x1 = self.gmcm4(x1)
        x2 = self.gmcm3(x2)
        x3 = self.gmcm2(x3)
        x4 = self.gmcm1(x4)

        x1 = self.fuse4(x1)
        x2 = self.fuse3(x2)
        x3 = self.fuse2(x3)
        x4 = self.fuse1(x4)

        x1 = self.up1(x1)
        edge = torch.cat([edge, x1], 1)
        edge = self.attention2(edge)
        edge1 = self.conv2(edge)

        x2 = self.up2(x2)
        bs2 = self.boundary2(edge, x2)

        x3 = self.up3(x3)
        bs3 = self.boundary3(edge, x3)

        x4 = self.up4(x4)
        bs4 = self.boundary4(edge, x4)

        ms = torch.cat([edge1, bs2, bs3, bs4], 1)
        out = self.multi_fusion(ms)

        edge_out = self.edge_feature(edge)
        edge_out = F.log_softmax(edge_out, dim=1)
        mask_out = self.out_feature(out)
        dis_out = self.dis_feature(out)
        return [mask_out, edge_out, dis_out]

if __name__ == "__main__":
    tensor = torch.randn((2, 3, 512, 512))
    net = Field()
    outputs = net(tensor)
    for o in outputs:
        print(o.shape)
