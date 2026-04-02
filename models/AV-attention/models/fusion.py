# Attentional Feature Fusion (Dai et al.); adapted from open-aff.
import torch
import torch.nn as nn

def _check_same_shape(f_aud, f_vis, module_name="fusion"):
    if f_aud.dim() != 4 or f_vis.dim() != 4:
        raise ValueError(
            f"{module_name} expects 4D tensors [B, C, H, W], "
            f"got {f_aud.shape} and {f_vis.shape}"
        )
    if f_aud.shape != f_vis.shape:
        raise ValueError(
            f"{module_name} expects same shape for audio/visual features, "
            f"got {f_aud.shape} and {f_vis.shape}"
        )

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, base_width=64,
                 fuse_type='DAF', stride=1, groups=1, dilation=1):
        super(Bottleneck, self).__init__()

        norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.conv4 = conv3x3(planes * self.expansion, width, stride, groups, dilation)
        self.bn4 = norm_layer(width)
        self.conv5 = conv1x1(width, inplanes)
        self.bn5 = norm_layer(inplanes)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.stride = stride

        if fuse_type == 'AFF':
            self.fuse_mode = AFF(channels=planes * self.expansion)
        elif fuse_type == 'iAFF':
            self.fuse_mode = iAFF(channels=planes * self.expansion)
        elif fuse_type == 'DAF':
            self.fuse_mode = DAF()
        elif fuse_type == 'Concat':
            self.fuse_mode = ConcatFusion(channels=planes * self.expansion)
        elif fuse_type == 'SUM':
            self.fuse_mode = SUM()
        elif fuse_type == 'MEAN':
            self.fuse_mode = MEAN()
        elif fuse_type == 'MAX':
            self.fuse_mode = MAX()
        elif fuse_type == 'PRODUCT':
            self.fuse_mode = PRODUCT()
        elif fuse_type == 'GMU':
            self.fuse_mode = GMU(channels=planes * self.expansion)
        elif fuse_type == 'CrossAttention':
            self.fuse_mode = CrossAttentionFusion(channels=planes * self.expansion)
        elif fuse_type == 'CAAFF':
            self.fuse_mode = CAAFF(channels=planes * self.expansion)
        else:
            self.fuse_mode = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.fuse_mode(out, identity)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        return out


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class DAF(nn.Module):
    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual

class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class iAFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class ConcatFusion(nn.Module):
    def __init__(self, channels=64):
        super(ConcatFusion, self).__init__()
        self.proj = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, residual):
        xo = torch.cat([x, residual], dim=1)
        xo = self.proj(xo)
        return xo


class SUM(nn.Module):
    def __init__(self):
        super(SUM, self).__init__()

    def forward(self, x, residual):
        return x + residual


class MEAN(nn.Module):
    def __init__(self):
        super(MEAN, self).__init__()

    def forward(self, x, residual):
        return 0.5 * (x + residual)


class MAX(nn.Module):
    def __init__(self):
        super(MAX, self).__init__()

    def forward(self, x, residual):
        return torch.max(x, residual)


class PRODUCT(nn.Module):
    def __init__(self):
        super(PRODUCT, self).__init__()

    def forward(self, x, residual):
        return x * residual


class GMU(nn.Module):
    def __init__(self, channels=64):
        super(GMU, self).__init__()
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        x_cat = torch.cat([x, residual], dim=1)
        z = self.sigmoid(self.gate(x_cat))
        xo = z * x + (1 - z) * residual
        return xo


class CrossAttentionBlock(nn.Module):
    def __init__(self, channels=64, num_heads=4):
        super(CrossAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, query_feat, key_value_feat):
        b, c, h, w = query_feat.size()

        query_tokens = query_feat.view(b, c, -1).permute(0, 2, 1)
        kv_tokens = key_value_feat.view(b, c, -1).permute(0, 2, 1)

        attn_out, _ = self.attn(query_tokens, kv_tokens, kv_tokens)
        attn_out = attn_out.permute(0, 2, 1).contiguous().view(b, c, h, w)

        return attn_out


class CrossAttentionFusion(nn.Module):
    def __init__(self, channels=64, num_heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.attn_x = CrossAttentionBlock(channels=channels, num_heads=num_heads)
        self.attn_r = CrossAttentionBlock(channels=channels, num_heads=num_heads)
        self.proj = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, residual):
        x_att = self.attn_x(x, residual) + x
        r_att = self.attn_r(residual, x) + residual

        xo = torch.cat([x_att, r_att], dim=1)
        xo = self.proj(xo)
        return xo


class CAAFF(nn.Module):
    def __init__(self, channels=64, r=4, num_heads=4):
        super(CAAFF, self).__init__()
        self.attn_x = CrossAttentionBlock(channels=channels, num_heads=num_heads)
        self.attn_r = CrossAttentionBlock(channels=channels, num_heads=num_heads)
        self.fuse = iAFF(channels=channels, r=r)

    def forward(self, x, residual):
        x_att = self.attn_x(x, residual) + x
        r_att = self.attn_r(residual, x) + residual
        xo = self.fuse(x_att, r_att)
        return xo


class IAFF_Paper(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        inter = max(1, channels // r)
        self.conv_y = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn_y = nn.BatchNorm2d(channels)

        def global_branch():
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )

        def local_branch():
            return nn.Sequential(
                nn.Conv2d(channels, inter, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )

        self.global_att = global_branch()
        self.local_att = local_branch()
        self.global_att2 = global_branch()
        self.local_att2 = local_branch()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, Y):
        conv_y = self.bn_y(self.conv_y(Y))
        X = conv_y + Y
        w = self.sigmoid(self.global_att(X) + self.local_att(X))
        x_prime = conv_y * w + Y * (1 - w)
        w2 = self.sigmoid(self.global_att2(x_prime) + self.local_att2(x_prime))
        y_prime = conv_y * w2 + Y * (1 - w2)
        return y_prime


class PDAFFusion(nn.Module):
    def __init__(self, dim=256, num_heads=4, iaff_r=4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.cross_vis_q = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_aud_q = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.iaff = IAFF_Paper(channels=dim * 2, r=iaff_r)
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
        )
        for m in self.out_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, f_vis, f_aud):
        fv = f_vis.unsqueeze(1)
        fa = f_aud.unsqueeze(1)
        vis_out, _ = self.cross_vis_q(fv, fa, fa)
        vis_hat = vis_out.squeeze(1) + f_vis
        aud_out, _ = self.cross_aud_q(fa, fv, fv)
        aud_hat = aud_out.squeeze(1) + f_aud
        cat = torch.cat([vis_hat, aud_hat], dim=1)
        Y = cat.view(cat.size(0), self.dim * 2, 1, 1)
        yp = self.iaff(Y)
        flat = yp.view(yp.size(0), -1)
        return self.out_proj(flat)


class SimpleConcatFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
        )
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, f_vis, f_aud):
        return self.proj(torch.cat([f_vis, f_aud], dim=1))