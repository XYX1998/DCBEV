import torch
import torch.nn as nn
from abc import ABCMeta
from mmcv.runner import BaseModule
from ..builder import NECKS
import torch.nn.functional as F
import pdb
from operator import mul
from functools import reduce
from torchvision.models.resnet import Bottleneck
import math
from einops import rearrange, repeat
from typing import List
import math
from einops import rearrange, repeat
from typing import List


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up

        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, dim, blocks, residual=True, factor=2):
        super().__init__()

        layers = list()
        channels = dim

        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        y = x

        for layer in self.layers:
            y = layer(y, x)

        return y




def _make_grid(resolution, extents):
    # Create a grid of cooridinates in the birds-eye-view
    x1, z1, x2, z2 = extents
    zz, xx = torch.meshgrid(torch.arange(z1, z2, resolution), torch.arange(x1, x2, resolution))

    return torch.stack([xx, zz], dim=-1)

def ResNetBottleNeck(c): return Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid(
        (xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1),
                    value=1)                   # 3 h w
    # 1 3 h w
    indices = indices[None]

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(
            mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(
            std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class TransformModule(nn.Module):
    def __init__(self, dim=8):
        super(TransformModule, self).__init__()
        self.dim = dim
        self.mat_list = nn.ModuleList()
        self.fc_transform = nn.Sequential(
            nn.Linear(dim * dim, dim * dim),
            nn.ReLU(),
            nn.Linear(dim * dim, dim * dim),
            nn.ReLU()
        )
    def forward(self, x):
        # shape x: B, C, H, W
        # x = self.bn(x)
        x = x.view(list(x.size()[:2]) + [self.dim * self.dim]).contiguous()
        view_comb = self.fc_transform(x)
        view_comb = view_comb.view(list(view_comb.size()[:2]).contiguous() + [self.dim, self.dim])
        return view_comb
# generate grids in BEV
class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        decoder_blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width,
                            h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()  # 3 3
        # 3 (h w)
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')
        grid = rearrange(grid, 'd (h w) -> d h w', h=h,
                         w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer(
            'grid', grid, persistent=False)                    # 3 h w
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, h, w))    # d h w

    def get_prior(self):
        return self.learned_features

@torch.no_grad()
def bev2image_sampling(points, I, E, height, width):
    """
    bev points to images: each bev point -> image points
    Args:
        points: (k, 3), (x,y,z)
        I: (b, n, 3, 3)
        E: (b, n, 4, 4)
    Return:
        sampled points: (k, 6, 2)
    """
    # (k, 3) -> (k, 4)
    k = points.shape[0]
    b, n = I.shape[:2]
    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    intrin_mat = F.pad(I, (0, 1, 0, 1), value=0)
    intrin_mat[..., -1, -1] = 1.0
    # (k, 3) -> (b, n, k, 4, 1)
    points = points.view(1, 1, k, 4).repeat(b, n, 1, 1).unsqueeze(-1).contiguous()
    # (b, n, 4, 4) * (k, 4)^T
    point2image = (intrin_mat @ E).view(b, n, 1, 4, 4).repeat(1, 1, k, 1, 1).contiguous()
    sample_points = (point2image @ points).squeeze(-1)  # (b, n, k, 4)

    # filter points
    eps = 1e-5
    # mask: (b, n, k, 4)
    mask = (sample_points[..., 2:3] > eps)
    sample_points = sample_points[..., 0:2] / \
        sample_points[..., 2:3].clamp(min=eps)

    sample_points[..., 0] /= width
    sample_points[..., 1] /= height

    # sample points in the image
    mask = (mask & (sample_points[..., 0:1] > 0.0)
            & (sample_points[..., 0:1] < 1.0)
            & (sample_points[..., 1:2] > 0.0)
            & (sample_points[..., 1:2] < 1.0))
    mask = torch.nan_to_num(mask)

    return sample_points, mask


class IndexBEVProjector(nn.Module):
    """GridBEVProjector, based on Grid Sampling (nearest)
    """

    def __init__(self, image_size, grid_size=(3, 3), height=-1.0):
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        grid_h, grid_w = grid_size
        y = torch.arange(grid_h) - grid_h // 2
        x = torch.arange(grid_w) - grid_w // 2
        offsets = torch.stack(torch.meshgrid(
            x, y, indexing="xy")).permute(1, 2, 0)
        self.register_buffer("grid_offsets", offsets, persistent=False)
        self.bev_height = height

    def forward(self, bev_grids, images, I, E):
        """
        bev_grids: (3, H, W)
        images: (b, n, c, h, w), features
        I: (b, n, 3, 3)
        E: (b, n, 4, 4)
        im_size: (height, width)
        """
        b, n = I.shape[:2]
        # unfold feature maps
        bn, c, h, w = images.shape

        # bev_grids -> image_coords
        # (3, H, W) -> (H*W, 3), k=H*W
        bev_points = bev_grids.reshape(3, -1).transpose(0, 1)
        bev_points[:, -1] = self.bev_height

        # (b, n, k, 2), (b, n, k, 1)
        sample_points, sample_mask = bev2image_sampling(
            bev_points, I, E, self.image_size[0], self.image_size[1])
        num_grid_points = self.grid_size[0] * self.grid_size[1]
        sample_points[..., 0] *= w
        sample_points[..., 1] *= h
        sample_points = sample_points.round().long()
        grid_offsets = self.grid_offsets.view(1, 1, 1, num_grid_points, 2).contiguous()

        # [b, n, k, 9, 2]
        sample_points = sample_points.unsqueeze(-2) + grid_offsets
        # restrict sample_points between 0~H-1
        sample_points[..., 0].clamp_(min=0, max=w-1)
        sample_points[..., 1].clamp_(min=0, max=h-1)
        # [b, n, k, 9]
        k = sample_points.shape[2]
        sample_points_inds = sample_points[..., 0] + sample_points[..., 1] * w
        # [b*n, k*9]
        sample_points_inds = sample_points_inds.view(
            b * n, k * num_grid_points).contiguous()
        # [b*n*h*w, c]
        images = rearrange(images, "b c h w -> (b h w) c")
        ind_offsets = (torch.arange(b * n, device=images.device)
                       * (h * w)).view(b * n, 1).contiguous()
        # b*n*k*9, 1
        sample_points_inds = (sample_points_inds + ind_offsets).view(-1).contiguous()
        # [b*n*k*9, c]
        sample_feats = images[sample_points_inds].reshape(
            b, n, k, num_grid_points, c)
        # embed()
        return sample_feats, sample_mask.detach()

class KernelAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(
            dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(
            dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(
            dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None, mask=None):
        """
        q: (b n d H W)
        k: (b n k g d)
        v: (b n k g d)
        mask: (b n k 1)
        """
        _, _, _, H, W = q.shape
        num_points = k.shape[-2]
        # Move feature dim to last for multi-head proj
        # (b, n, k, d)
        q = rearrange(q, 'b n d H W -> b n (H W) d')

        # Project with multiple heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # Group the head dim with batch dim
        q = rearrange(q, 'b n q (m d) -> (b m) n q 1 d',
                      m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b n q g (m d) -> (b m) n q g d',
                      m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b n q g (m d) -> (b m) q (n g) d',
                      m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * \
            torch.einsum('b n Q c d, b n Q K d -> b n Q c K', q, k)
        dot = rearrange(dot, 'b n Q c K -> b Q (n c K)')
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1, num_points)
            mask = rearrange(mask, 'b h n Q g -> (b h) Q (n g)')
            dot[~mask] = -10**9
        att = dot.to(q).softmax(dim=-1)
        a = torch.einsum('b Q K, b Q K d -> b Q d', att, v)

        a = rearrange(a, '(b m) Q d -> b Q (m d)',
                      m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z

class GeometryKernelAttention(nn.Module):

    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        bev_z: int,
        kernel_h: int,
        kernel_w: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
        sampling_type: str = "index",
        use_kernel_conv: bool = True,
        kernel_conv_h: int = 1,
        kernel_conv_w: int = 7
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        if sampling_type == "unfold":
            self.sampling = UnfoldBEVProjector(
                (image_height, image_width), grid_size=(kernel_h, kernel_w), height=bev_z)
        elif sampling_type == "index":
            self.sampling = IndexBEVProjector(
                (image_height, image_width), grid_size=(kernel_h, kernel_w), height=bev_z)
        else:
            raise NotImplementedError()

        self.feature_linear = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, dim, bias=False)
        )

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, dim, bias=False)
            )
        if use_kernel_conv:
            self.conv = nn.Conv2d(
                feat_dim, feat_dim, (kernel_conv_h, kernel_conv_w),
                padding=(kernel_conv_h // 2, kernel_conv_w // 2))
        else:
            self.conv = lambda x: x

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Linear(4, dim, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attn = KernelAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        I_: torch.FloatTensor,
        E_: torch.FloatTensor
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        # """
        # print(E_inv.size())
        # print(I_inv.size())
        # 使用 unsqueeze 在第 1 维度（索引为 1）插入一个大小为 1 的维度
        # for f in feature:
        #     print(feature.size())
 
        I_ = I_.unsqueeze(1)  # 形状变为 [12, 1, 4, 4]
        I_inv = I_inv.unsqueeze(1)  # 形状变为 [12, 1, 3, 3]
        E_ = E_.unsqueeze(1)  # 形状变为 [12, 1, 4, 4]
        E_inv = E_inv.unsqueeze(1)  # 形状变为 [12, 1, 3, 3]

        n = 1
        b = feature.size()[0]

        # b n 3 h w
        pixel = self.image_plane
        _, _, _, h, w = pixel.shape

        # b n 4 1
        c = E_inv[..., -1:]
        # (b n) 4 1 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]
        # (b n) d 1 1
        c_embed = self.cam_embed(c_flat)

        # 1 1 3 (h w)
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        # b n 3 (h w)
        cam = I_inv @ pixel_flat
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)
        # b n 4 (h w)
        d = E_inv @ cam
        # (b n) 4 h w
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)

        # 2 H W
        world = bev.grid[:2]
        # 1 d H W
        w_embed = self.bev_embed(world[None])
        # (b n) d H W
        bev_embed = w_embed - c_embed
        # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
        # b n d H W
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)

        # (b n) d h w

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        # print(feature_flat.size())
        feature_flat = self.conv(feature_flat)
        # project local patches using sampling
        # concat feature and embeddings for sampling
        d_feature = feature_flat.shape[1]
        feature_embed = torch.cat([feature_flat, d_flat], dim=1)
        feature_embed, mask = self.sampling(
            bev.grid.detach().clone(), feature_embed, I_, E_)

        # b, n, q, num_points, c
        feature_flat = feature_embed[..., :d_feature]
        d_flat = feature_embed[..., d_feature:]

        # (b n) q, num_points, 4
        d_embed = self.img_embed(d_flat)

        # d_embed: b, n, q, num_points, d
        # c_embed: (b, n), d, 1, 1
        img_embed = d_embed - c_embed.view(b, n, 1, 1, d_embed.shape[-1]).contiguous()
        img_embed = img_embed / (img_embed.norm(dim=-1, keepdim=True) + 1e-7)

        # g: num_grid_points
        # b, n, q, g, c
        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)
        else:
            # (b, n) d, h, w
            key_flat = img_embed

        # (b, n) d, h, w
        val_flat = self.feature_linear(feature_flat)

        # Expand + refine the BEV embedding
        # b, n, d, H, W
        query = query_pos + x[:, None]

        return self.cross_attn(query, key_flat, val_flat, mask=mask, skip=x if self.skip else None)

class GeometryKernelEncoder(nn.Module):

    def __init__(
            self,
            dim: int = 64,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        cross_views = list()
        layers = list()
        cross_view = {
                    'heads': 4,
                    'dim_head': 32,
                    'qkv_bias': True,
                    'skip': True,
                    'no_image_features': False,
                    'bev_z': 1.0,
                    'image_height': 1024,
                    'image_width': 1024,
                    'kernel_h': 7,
                    'kernel_w': 1,
                    'sampling_type': "index",
                    'use_kernel_conv': True,
                    'kernel_conv_h': 1,
                    'kernel_conv_w': 7
                }
        if scale < 1.0:
            self.down = lambda x: F.interpolate(
                x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x
        bev_embedding = {
                    'dim': 64,
                    'sigma': 0.02,
                    'bev_height': 98,
                    'bev_width': 100,
                    'h_meters': 50,
                    'w_meters': 50,
                    'offset': 0,
                    'decoder_blocks': [2, 2, 2]
                }

        self.output_shape =[[12, 96, 256, 256], [12, 192, 128, 128], [12, 384, 64, 64], [12, 768, 32, 32]]
        for feat_shape, num_layers in zip(self.output_shape, middle):
            _, feat_dim, feat_height, feat_width = self.down(
                torch.zeros(feat_shape)).shape

            cva = GeometryKernelAttention(
                feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim)
                                  for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(**bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, feature_maps,intrinsics,extrinsics):
        b, _, _, _,  = feature_maps[0].size()
        # bev_feats = list()
        scale = 8,16,32,64,128
        # calib.shape = [bs,3,3]

        # b n c h w
        # image = batch['image'].flatten(0, 1)
        # b n 3 3
        I_inv = intrinsics.inverse()
        # b n 4 4
        E_inv = extrinsics.inverse()     

        # features = [self.down(y) for y in self.backbone(self.norm(image))]

        # d H W
        x = self.bev_embedding.get_prior()
        # b d H W
        x = repeat(x, '... -> b ...', b=b)
        # print('bbb,nnn',b,n)
        for cross_view, feature, layer in zip(self.cross_views, feature_maps, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=1)
            x = cross_view(x, self.bev_embedding, feature, I_inv,
                           E_inv, intrinsics, extrinsics)
            x = layer(x)

        return x


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up

        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, dim, blocks, residual=True, factor=2):
        super().__init__()

        layers = list()
        channels = dim

        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        y = x

        for layer in self.layers:
            y = layer(y, x)

        return y




def _make_grid(resolution, extents):
    # Create a grid of cooridinates in the birds-eye-view
    x1, z1, x2, z2 = extents
    zz, xx = torch.meshgrid(torch.arange(z1, z2, resolution), torch.arange(x1, x2, resolution))

    return torch.stack([xx, zz], dim=-1)

def ResNetBottleNeck(c): return Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid(
        (xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1),
                    value=1)                   # 3 h w
    # 1 3 h w
    indices = indices[None]

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(
            mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(
            std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class TransformModule(nn.Module):
    def __init__(self, dim=8):
        super(TransformModule, self).__init__()
        self.dim = dim
        self.mat_list = nn.ModuleList()
        self.fc_transform = nn.Sequential(
            nn.Linear(dim * dim, dim * dim),
            nn.ReLU(),
            nn.Linear(dim * dim, dim * dim),
            nn.ReLU()
        )
    def forward(self, x):
        # shape x: B, C, H, W
        # x = self.bn(x)
        x = x.view(list(x.size()[:2]) + [self.dim * self.dim]).contiguous()
        view_comb = self.fc_transform(x)
        view_comb = view_comb.view(list(view_comb.size()[:2]).contiguous() + [self.dim, self.dim])
        return view_comb
# generate grids in BEV
class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        decoder_blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width,
                            h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()  # 3 3
        # 3 (h w)
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')
        grid = rearrange(grid, 'd (h w) -> d h w', h=h,
                         w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer(
            'grid', grid, persistent=False)                    # 3 h w
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, h, w))    # d h w

    def get_prior(self):
        return self.learned_features

@torch.no_grad()
def bev2image_sampling(points, I, E, height, width):
    """
    bev points to images: each bev point -> image points
    Args:
        points: (k, 3), (x,y,z)
        I: (b, n, 3, 3)
        E: (b, n, 4, 4)
    Return:
        sampled points: (k, 6, 2)
    """
    # (k, 3) -> (k, 4)
    k = points.shape[0]
    b, n = I.shape[:2]
    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    intrin_mat = F.pad(I, (0, 1, 0, 1), value=0)
    intrin_mat[..., -1, -1] = 1.0
    # (k, 3) -> (b, n, k, 4, 1)
    points = points.view(1, 1, k, 4).repeat(b, n, 1, 1).unsqueeze(-1).contiguous()
    # (b, n, 4, 4) * (k, 4)^T
    point2image = (intrin_mat @ E).view(b, n, 1, 4, 4).repeat(1, 1, k, 1, 1).contiguous()
    sample_points = (point2image @ points).squeeze(-1).contiguous()  # (b, n, k, 4)

    # filter points
    eps = 1e-5
    # mask: (b, n, k, 4)
    mask = (sample_points[..., 2:3] > eps)
    sample_points = sample_points[..., 0:2] / \
        sample_points[..., 2:3].clamp(min=eps)

    sample_points[..., 0] /= width
    sample_points[..., 1] /= height

    # sample points in the image
    mask = (mask & (sample_points[..., 0:1] > 0.0)
            & (sample_points[..., 0:1] < 1.0)
            & (sample_points[..., 1:2] > 0.0)
            & (sample_points[..., 1:2] < 1.0))
    mask = torch.nan_to_num(mask)

    return sample_points, mask


class IndexBEVProjector(nn.Module):
    """GridBEVProjector, based on Grid Sampling (nearest)
    """

    def __init__(self, image_size, grid_size=(3, 3), height=-1.0):
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        grid_h, grid_w = grid_size
        y = torch.arange(grid_h) - grid_h // 2
        x = torch.arange(grid_w) - grid_w // 2
        offsets = torch.stack(torch.meshgrid(
            x, y, indexing="xy")).permute(1, 2, 0)
        self.register_buffer("grid_offsets", offsets, persistent=False)
        self.bev_height = height

    def forward(self, bev_grids, images, I, E):
        """
        bev_grids: (3, H, W)
        images: (b, n, c, h, w), features
        I: (b, n, 3, 3)
        E: (b, n, 4, 4)
        im_size: (height, width)
        """
        b, n = I.shape[:2]
        # unfold feature maps
        bn, c, h, w = images.shape

        # bev_grids -> image_coords
        # (3, H, W) -> (H*W, 3), k=H*W
        bev_points = bev_grids.reshape(3, -1).transpose(0, 1)
        bev_points[:, -1] = self.bev_height

        # (b, n, k, 2), (b, n, k, 1)
        sample_points, sample_mask = bev2image_sampling(
            bev_points, I, E, self.image_size[0], self.image_size[1])
        num_grid_points = self.grid_size[0] * self.grid_size[1]
        sample_points[..., 0] *= w
        sample_points[..., 1] *= h
        sample_points = sample_points.round().long()
        grid_offsets = self.grid_offsets.view(1, 1, 1, num_grid_points, 2).contiguous()

        # [b, n, k, 9, 2]
        sample_points = sample_points.unsqueeze(-2) + grid_offsets
        # restrict sample_points between 0~H-1
        sample_points[..., 0].clamp_(min=0, max=w-1)
        sample_points[..., 1].clamp_(min=0, max=h-1)
        # [b, n, k, 9]
        k = sample_points.shape[2]
        sample_points_inds = sample_points[..., 0] + sample_points[..., 1] * w
        # [b*n, k*9]
        sample_points_inds = sample_points_inds.view(
            b * n, k * num_grid_points).contiguous()
        # [b*n*h*w, c]
        images = rearrange(images, "b c h w -> (b h w) c")
        ind_offsets = (torch.arange(b * n, device=images.device)
                       * (h * w)).view(b * n, 1).contiguous()
        # b*n*k*9, 1
        sample_points_inds = (sample_points_inds + ind_offsets).view(-1).contiguous()
        # [b*n*k*9, c]
        sample_feats = images[sample_points_inds].reshape(
            b, n, k, num_grid_points, c)
        # embed()
        return sample_feats, sample_mask.detach()

class KernelAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(
            dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(
            dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(
            dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None, mask=None):
        """
        q: (b n d H W)
        k: (b n k g d)
        v: (b n k g d)
        mask: (b n k 1)
        """
        _, _, _, H, W = q.shape
        num_points = k.shape[-2]
        # Move feature dim to last for multi-head proj
        # (b, n, k, d)
        q = rearrange(q, 'b n d H W -> b n (H W) d')

        # Project with multiple heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # Group the head dim with batch dim
        q = rearrange(q, 'b n q (m d) -> (b m) n q 1 d',
                      m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b n q g (m d) -> (b m) n q g d',
                      m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b n q g (m d) -> (b m) q (n g) d',
                      m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * \
            torch.einsum('b n Q c d, b n Q K d -> b n Q c K', q, k)
        dot = rearrange(dot, 'b n Q c K -> b Q (n c K)')
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1, num_points)
            mask = rearrange(mask, 'b h n Q g -> (b h) Q (n g)')
            dot[~mask] = -10**9
        att = dot.to(q).softmax(dim=-1)
        a = torch.einsum('b Q K, b Q K d -> b Q d', att, v)

        a = rearrange(a, '(b m) Q d -> b Q (m d)',
                      m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z

class GeometryKernelAttention(nn.Module):

    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        bev_z: int,
        kernel_h: int,
        kernel_w: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
        sampling_type: str = "index",
        use_kernel_conv: bool = True,
        kernel_conv_h: int = 1,
        kernel_conv_w: int = 7
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        if sampling_type == "unfold":
            self.sampling = UnfoldBEVProjector(
                (image_height, image_width), grid_size=(kernel_h, kernel_w), height=bev_z)
        elif sampling_type == "index":
            self.sampling = IndexBEVProjector(
                (image_height, image_width), grid_size=(kernel_h, kernel_w), height=bev_z)
        else:
            raise NotImplementedError()

        self.feature_linear = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, dim, bias=False)
        )

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, dim, bias=False)
            )
        if use_kernel_conv:
            self.conv = nn.Conv2d(
                feat_dim, feat_dim, (kernel_conv_h, kernel_conv_w),
                padding=(kernel_conv_h // 2, kernel_conv_w // 2))
        else:
            self.conv = lambda x: x

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Linear(4, dim, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attn = KernelAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        I_: torch.FloatTensor,
        E_: torch.FloatTensor
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        # """
        # print(E_inv.size())
        # print(I_inv.size())
        # 使用 unsqueeze 在第 1 维度（索引为 1）插入一个大小为 1 的维度
        # for f in feature:
        #     print(feature.size())
 
        I_ = I_.unsqueeze(1)  # 形状变为 [12, 1, 4, 4]
        I_inv = I_inv.unsqueeze(1)  # 形状变为 [12, 1, 3, 3]
        E_ = E_.unsqueeze(1)  # 形状变为 [12, 1, 4, 4]
        E_inv = E_inv.unsqueeze(1)  # 形状变为 [12, 1, 3, 3]

        n = 1
        b = feature.size()[0]

        # b n 3 h w
        pixel = self.image_plane
        _, _, _, h, w = pixel.shape

        # b n 4 1
        c = E_inv[..., -1:]
        # (b n) 4 1 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]
        # (b n) d 1 1
        c_embed = self.cam_embed(c_flat)

        # 1 1 3 (h w)
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        # b n 3 (h w)
        cam = I_inv @ pixel_flat
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)
        # b n 4 (h w)
        d = E_inv @ cam
        # (b n) 4 h w
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)

        # 2 H W
        world = bev.grid[:2]
        # 1 d H W
        w_embed = self.bev_embed(world[None])
        # (b n) d H W
        bev_embed = w_embed - c_embed
        # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
        # b n d H W
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)

        # (b n) d h w

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        # print(feature_flat.size())
        feature_flat = self.conv(feature_flat)
        # project local patches using sampling
        # concat feature and embeddings for sampling
        d_feature = feature_flat.shape[1]
        feature_embed = torch.cat([feature_flat, d_flat], dim=1)
        feature_embed, mask = self.sampling(
            bev.grid.detach().clone(), feature_embed, I_, E_)

        # b, n, q, num_points, c
        feature_flat = feature_embed[..., :d_feature]
        d_flat = feature_embed[..., d_feature:]

        # (b n) q, num_points, 4
        d_embed = self.img_embed(d_flat)

        # d_embed: b, n, q, num_points, d
        # c_embed: (b, n), d, 1, 1
        img_embed = d_embed - c_embed.view(b, n, 1, 1, d_embed.shape[-1]).contiguous()
        img_embed = img_embed / (img_embed.norm(dim=-1, keepdim=True) + 1e-7)

        # g: num_grid_points
        # b, n, q, g, c
        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)
        else:
            # (b, n) d, h, w
            key_flat = img_embed

        # (b, n) d, h, w
        val_flat = self.feature_linear(feature_flat)

        # Expand + refine the BEV embedding
        # b, n, d, H, W
        query = query_pos + x[:, None]

        return self.cross_attn(query, key_flat, val_flat, mask=mask, skip=x if self.skip else None)

class GeometryKernelEncoder(nn.Module):

    def __init__(
            self,
            dim: int = 64,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        cross_views = list()
        layers = list()
        cross_view = {
                    'heads': 4,
                    'dim_head': 32,
                    'qkv_bias': True,
                    'skip': True,
                    'no_image_features': False,
                    'bev_z': 1.0,
                    'image_height': 1024,
                    'image_width': 1024,
                    'kernel_h': 7,
                    'kernel_w': 1,
                    'sampling_type': "index",
                    'use_kernel_conv': True,
                    'kernel_conv_h': 1,
                    'kernel_conv_w': 7
                }
        if scale < 1.0:
            self.down = lambda x: F.interpolate(
                x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x
        bev_embedding = {
                    'dim': 64,
                    'sigma': 0.02,
                    'bev_height': 98,
                    'bev_width': 100,
                    'h_meters': 49,
                    'w_meters': 50,
                    'offset': 0,
                    'decoder_blocks': [2, 2, 2]
                }

        self.output_shape =[[12, 209, 256, 256], [12, 305, 128, 128], [12, 497, 64, 64], [12, 881, 32, 32]]
        for feat_shape, num_layers in zip(self.output_shape, middle):
            
            _, feat_dim, feat_height, feat_width = self.down(
                torch.zeros(feat_shape)).shape

            cva = GeometryKernelAttention(
                feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim)
                                  for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(**bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, feature_maps,intrinsics,extrinsics):
        b, _, _, _,  = feature_maps[0].size()
        # bev_feats = list()
        scale = 8,16,32,64,128
        # calib.shape = [bs,3,3]

        # b n c h w
        # image = batch['image'].flatten(0, 1)
        # b n 3 3
        I_inv = intrinsics.inverse()
        # b n 4 4
        E_inv = extrinsics.inverse()     

        # features = [self.down(y) for y in self.backbone(self.norm(image))]

        # d H W
        x = self.bev_embedding.get_prior()
        # b d H W
        x = repeat(x, '... -> b ...', b=b)
        # print('bbb,nnn',b,n)
        for cross_view, feature, layer in zip(self.cross_views, feature_maps, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=1)
            x = cross_view(x, self.bev_embedding, feature, I_inv,
                           E_inv, intrinsics, extrinsics)
            x = layer(x)

        return x

# lss part
def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats

class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])
        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)
        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)
        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        val = gradx[back]
        return val, None, None

# pyva part
# generate grids in BEV
def _make_grid(resolution, extents):
    # Create a grid of cooridinates in the birds-eye-view
    x1, z1, x2, z2 = extents
    zz, xx = torch.meshgrid(torch.arange(z1, z2, resolution), torch.arange(x1, x2, resolution))
    return torch.stack([xx, zz], dim=-1)

class Resampler(nn.Module):
    def __init__(self, resolution, extents):
        super().__init__()

        # Store z positions of the near and far planes
        # extents[1]:zmin,extents[3]:zmax
        self.near = extents[1]
        self.far = extents[3]

        # Make a grid in the x-z plane
        self.grid = _make_grid(resolution, extents)

    def forward(self, features, calib):
        # Copy grid to the correct device
        self.grid = self.grid.to(features)

        # We ignore the image v-coordinate, and assume the world Y-coordinate
        # is zero, so we only need a 2x2 submatrix of the original 3x3 matrix
        # calib shape:[bs,3,3]-->[bs,2,3]-->[bs,2,2]-->[bs,1,1,2,2]
        calib = calib[:, [0, 2]][..., [0, 2]].view(-1, 1, 1, 2, 2).contiguous()

        # Transform grid center locations into image u-coordinates
        cam_coords = torch.matmul(calib, self.grid.unsqueeze(-1)).squeeze(-1)

        # Apply perspective projection and normalize
        ucoords = cam_coords[..., 0] / cam_coords[..., 1]
        ucoords = ucoords / features.size(-1) * 2 - 1

        # Normalize z coordinates
        zcoords = (cam_coords[..., 1] - self.near) / (self.far - self.near) * 2 - 1

        # Resample 3D feature map
        grid_coords = torch.stack([ucoords, zcoords], -1).clamp(-1.1, 1.1)
        return F.grid_sample(features, grid_coords)


class DenseTransformer(nn.Module):
    def __init__(self, in_channels, channels, resolution, grid_extents,
                 ymin, ymax, focal_length, groups=1):
        super().__init__()

        # Initial convolution to reduce feature dimensions
        self.conv = nn.Conv2d(in_channels, channels, 1)
        self.bn = nn.GroupNorm(16, channels)

        # Resampler transforms perspective features to BEV
        self.resampler = Resampler(resolution, grid_extents)

        # Compute input height based on region of image covered by grid
        self.zmin, zmax = grid_extents[1], grid_extents[3]
        self.in_height = math.ceil(focal_length * (ymax - ymin) / self.zmin)
        # self.ymid = 1
        self.ymid = (ymin + ymax) / 2

        # Compute number of output cells required
        self.out_depth = math.ceil((zmax - self.zmin) / resolution)

        # Dense layer which maps UV features to UZ
        self.fc = nn.Conv1d(
            channels * self.in_height, channels * self.out_depth, 1, groups=groups
        )
        self.out_channels = channels

    def forward(self, features, calib, *args):
        # Crop feature maps to a fixed input height
        # print('features,',features)
        features = torch.stack([self._crop_feature_map(fmap, cal)
                                for fmap, cal in zip(features, calib)])

        # Reduce feature dimension to minimize memory usage
        features = F.relu(self.bn(self.conv(features)))

        # Flatten height and channel dimensions
        B, C, _, W = features.shape
        flat_feats = features.flatten(1, 2)
        # H is not fixed every time
        bev_feats = self.fc(flat_feats).view(B, C, -1, W).contiguous()
        # Resample to orthographic grid
        return self.resampler(bev_feats, calib)

    def _crop_feature_map(self, fmap, calib):
        # Compute upper and lower bounds of visible region
        focal_length, img_offset = calib[1, 1:]
        vmid = self.ymid * focal_length / self.zmin + img_offset
        vmin = math.floor(vmid - self.in_height / 2)
        vmax = math.floor(vmid + self.in_height / 2)

        # Pad or crop input tensor to match dimensions
        return F.pad(fmap, [0, 0, -vmin, vmax - fmap.shape[-2]])


@NECKS.register_module()
# class LSS_PYVA_neck(BaseModule):
    # def __init__(self, in_channels=768, channels=64, resolution= 0.25 * reduce(mul, [1, 2]),
    #              extents=[-25.0, 1.0, 25.0, 50.0], ymin=-2, ymax=4, focal_length=630.0,
    #              downsample=32, bev_feature_channels=64, ogfH=600, ogfW=800,
    #              grid_conf=dict(dbound=[1, 50, 1], xbound=[-25,25,0.5], zbound=[1, 50, 0.5], ybound=[-10,10,20])):
class DGKT_withoutGKT(BaseModule, metaclass=ABCMeta):
    def __init__(self, in_channels=768, channels=64, resolution= 0.25 * reduce(mul, [1, 2]),
                extents=[-25.0, 1.0, 25.0, 50.0], ymin=-2, ymax=4, focal_length=630.0,
                downsample=32, bev_feature_channels=64, ogfH=1024, ogfW=1024,
                grid_conf=dict(dbound=[1, 50, 1], xbound=[-25,25,0.5], zbound=[1, 50, 0.5], ybound=[-10,10,20])):
        super(DGKT_withoutGKT, self).__init__()
 
    # resolution = 0.25*(1*2)=0.5
    # focal = 78.75,39.375,19.6875,9.84375,4.921875
    # subset_extents = [-25,39,25,50],[-25,19.5,25,39],\
    # [-25,9.5,25,19.5],[-25,4.5,25,9.5],[-25,1,25,4.5]
    # ymin=-2,ymax=4
        # super(LSS_PYVA_neck,self).__init__()

        # pyva part
        # self.transformers = nn.ModuleList()
        
        # for i in range(5):
        #     # Scaled focal length for each transformer
        #     focal = focal_length / pow(2, i + 3)

        #     # Compute grid bounds for each transformer
        #     zmax = min(math.floor(focal * 2) * resolution, extents[3])
        #     zmin = math.floor(focal) * resolution if i < 4 else extents[1]
        #     subset_extents = [extents[0], zmin, extents[2], zmax]
        #     # Build transformers
        #     tfm = DenseTransformer(in_channels, channels, resolution,
        #                            subset_extents, ymin, ymax, focal)
        #     self.transformers.append(tfm)

        # lss part
        self.grid_conf = grid_conf
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.downsample = downsample
        self.ogfH = ogfH
        self.ogfW = ogfW
        self.frustum = self.create_frustum()
        self.D ,_,_,_ = self.frustum.shape
        self.C = bev_feature_channels
        # by default, self.C = 64, self.D = 49
        # self.output_shape =[[12, 96, 256, 256], [12, 192, 128, 128], [12, 384, 64, 64], [12, 768, 32, 32]]
        self.depthnet4 = nn.Conv2d(in_channels, self.D + self.C, kernel_size=1, padding=0)
        self.depthnet3 = nn.Conv2d(384, self.D + self.C,kernel_size=1, padding=0)
        self.depthnet2 = nn.Conv2d(192, self.D + self.C, kernel_size=1, padding=0)
        self.depthnet1 = nn.Conv2d(96, self.D + self.C, kernel_size=1, padding=0)
        self.use_quickcumsum = True

    def create_frustum(self):
        fH, fW = self.ogfH//self.downsample, self.ogfW//self.downsample
        depth_samples = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        num_depth, _, _ = depth_samples.shape
        x_samples = torch.linspace(0, self.ogfW - 1, fW, dtype=torch.float).view(1,1,fW).expand(num_depth,fH,fW)
        y_samples = torch.linspace(0, self.ogfH - 1, fH, dtype=torch.float).view(1,fH,1).expand(num_depth,fH,fW)

        # D x H x W x 3
        frustum = torch.stack((x_samples,y_samples,depth_samples), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, intrinstics):
        B = intrinstics.shape[0]
        D,H,W,C = self.frustum.shape
        points = self.frustum.view(1,D,H,W,-1).expand(B,D,H,W,C)
        points = torch.cat([points[:,:,:,:,:2]* points[:,:,:,:,2:3],
                            points[:,:,:,:,2:3]],4)
        combine = torch.inverse(intrinstics)
        points = combine.view(B,1,1,1,3,3).matmul(points.unsqueeze(-1)).squeeze(-1).view(B,1,D,H,W,-1)
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)
        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (nx[1] * nx[2] * B) \
                + geom_feats[:, 1] * (nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[1], nx[2], nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 1], geom_feats[:, 2], geom_feats[:, 0]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def forward(self, feature_maps, IE):
        I, E = IE
        geom = self.get_geometry(I)  
        b, _, _, h, w, _ = geom.shape
        
        DGKT_features = []
        # print(len(feature_maps))
        feature_ls_list = []
        for i, feature_map in enumerate(feature_maps):
            # print(i,feature_map.size())
            # feature = feature_map[:, :, :h, :w]
            # print(f'特征图的featuresize {i}:', feature_map.size())

            # Apply the corresponding depthnet
            if i == 3:
                feature = self.depthnet4(feature_map)
            elif i == 2:
                feature = self.depthnet3(feature_map)
            elif i == 1:
                feature = self.depthnet2(feature_map)
            elif i == 0:
                feature = self.depthnet1(feature_map)

            # print(f'深度图的featuresize {i}:', feature.size())
            feature = F.interpolate(feature, size=(32,32), mode='bilinear', align_corners=False)

            # print(f'合并的featuresize {i}:', combined_feature.size())
            depth = self.get_depth_dist(feature[:, :self.D])
            new_feature = depth.unsqueeze(1) * feature[:, self.D:(self.D + self.C)].unsqueeze(2)
            feature = new_feature.view(b, 1, self.C, self.D, h, w)
            feature = feature.permute(0, 1, 3, 4, 5, 2)
            feature_lss =  self.voxel_pooling(geom, feature)
            # print(f'feature_lss {i}:', feature_lss.size())
            feature_ls_list.append(feature_lss)


        feature_final  = feature_ls_list[0] + feature_ls_list[1] + feature_ls_list[2] + feature_ls_list[3]

        return feature_final,feature_final,feature_lss