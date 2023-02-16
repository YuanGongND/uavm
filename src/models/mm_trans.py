# -*- coding: utf-8 -*-
# @Time    : 11/27/22 00:18 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : mm_trans.py

# Transformer models in paper "UAVM: Towards Unifying Audio and Visual Models"

import math
import warnings
import torch
import torch.nn as nn
import numpy as np
import torchvision
from functools import partial

# code from the t2t-vit paper
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        #print(C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Single Modality Transformer (Building Block of UAVM)
class SingleTrans(nn.Module):
    """ The Single Modality Transformer Model (the building block of the unified Transformer model) """
    def __init__(self, embed_dim, num_heads, depth, label_dim, seq_len, input_dim):
        """
        :param embed_dim: the Transformer embedding dimension
        :param num_heads: the number of attention heads
        :param label_dim: the number of classification labels
        :param seq_len: the input audio/visual feature sequence length
        :param input_dim: the input audio/visual feature dimension
        """
        super().__init__()
        self.input_dim, self.embed_dim = input_dim, embed_dim
        # Transformer encode blocks
        self.blocks = nn.ModuleList([Block(dim=self.embed_dim, num_heads=num_heads) for i in range(depth)])
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        self.in_layernorm = nn.LayerNorm(self.input_dim, eps=1e-06)
        self.in_proj = nn.Linear(self.input_dim, self.embed_dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.embed_dim, eps=1e-06), nn.Linear(self.embed_dim, label_dim))
        print('sequence length of the model is {:d}'.format(seq_len))

    def forward(self, x, feat=False):
        """ The main forward function
        :param x: input audio or video features, require shape in [batch_size, sequence_len, feat_dim]
        :param feat: if True, return last Transformer layer output, if False, return the classification logits
        :return: x: if feat==True, output [batch_size, sequence_len, embed_dim], if feat==False, output [batch_size, label_dim]
        """
        x = self.in_layernorm(x)
        # only apply the projection if the input
        if self.embed_dim != self.input_dim:
            x = self.in_proj(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        if feat == False:
            x = torch.mean(x, dim=1)
            x = self.mlp_head(x)
        return x

    def extract_feat(self, x, out_layer=0):
        """ Get the feature of a specific Transformer layer (used for probing test)
        :param x: input audio or video features, require shape in [batch_size, sequence_len, feat_dim]
        :param out_layer: feature of which layer to output
        :return: x: output [batch_size, sequence_len, embed_dim]
        """
        x = self.in_layernorm(x)
        if self.embed_dim != self.input_dim:
            x = self.in_proj(x)
        x = x + self.pos_embed
        layer = 0
        for blk in self.blocks:
            if layer >= out_layer:
                break
            x = blk(x)
            layer += 1
        return x

    def get_att_map(self, block, x):
        """ Get the attention map of a specific input x of a specific block (used for probing test)
        :param block: the attention block of interest
        :param x: input audio or video features, require shape in [batch_size, sequence_len, feat_dim]
        :return: attn: output attention map
        """
        qkv = block.attn.qkv
        num_heads = block.attn.num_heads
        scale = block.attn.scale
        B, N, C = x.shape
        qkv = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn

    def extract_att_map(self, x, feat=False):
        """ Get the attention map of a specific input x of all attention blocks (used for probing test)
        :param x: input audio or video features, require shape in [batch_size, sequence_len, feat_dim]
        :return: x: output of transformer
        :return: att_list: a list attention maps, of all attention blocks
        """
        x = self.in_layernorm(x)
        # only apply the projection if the input
        if self.embed_dim != self.input_dim:
            x = self.in_proj(x)
        x = x + self.pos_embed
        att_list = []
        for blk in self.blocks:
            cur_att = self.get_att_map(blk, x)
            att_list.append(cur_att)
            x = blk(x)
        if feat == False:
            x = torch.mean(x, dim=1)
            x = self.mlp_head(x)
        return x, att_list

# The UAVM Model (the main model proposed in the paper)
class UnifiedTrans(nn.Module):
    """ The UAVM Model (the main model proposed in the paper) """
    def __init__(self, embed_dim, num_heads, depth_i, depth_s, label_dim, a_seq_len, v_seq_len, a_input_dim, v_input_dim, s_embed_dim=0):
        """
        :param embed_dim: the embedding dimension of the modality-specific layers
        :param num_heads: the number of attention heads
        :param depth_i: the number of modality-independent layers
        :param depth_s: the number of modality-sharing layers
        :param label_dim: the number of classification labels
        :param a_seq_len: the length of audio feature sequence length
        :param v_seq_len: the length of video feature sequence length
        :param a_input_dim: the input audio feature dimension
        :param v_input_dim: the input video feature dimension
        :param s_embed_dim: the embedding dimension of the modality-sharing layers, when set to 0, equal to embed_dim
        :return: returns nothing
        """
        super().__init__()

        self.a_seq_len, self.v_seq_len = a_seq_len, v_seq_len
        self.a_input_dim, self.v_input_dim = a_input_dim, v_input_dim
        self.depth_i, self.depth_s = depth_i, depth_s

        self.embed_dim = embed_dim
        self.s_embed_dim = self.embed_dim if s_embed_dim == 0 else s_embed_dim

        assert self.a_seq_len == self.v_seq_len
        self.seq_len = self.a_seq_len

        # if no modality-specific layer, add a projection layer before inputting to modality-sharing layer
        if self.depth_i == 0:
            self.a_in_proj = nn.Sequential(nn.LayerNorm(a_input_dim, eps=1e-06), nn.Linear(self.a_input_dim, embed_dim))
            self.v_in_proj = nn.Sequential(nn.LayerNorm(v_input_dim, eps=1e-06), nn.Linear(self.v_input_dim, embed_dim))
        else:
            self.a_trans = SingleTrans(self.embed_dim, num_heads, self.depth_i, label_dim, self.a_seq_len, self.a_input_dim)
            self.v_trans = SingleTrans(self.embed_dim, num_heads, self.depth_i, label_dim, self.v_seq_len, self.v_input_dim)

        # if no modality-sharing layer, add mlp layers for audio and video, making it a completely modality-independent models
        if self.depth_s != 0:
            self.s_trans = SingleTrans(self.s_embed_dim, num_heads, self.depth_s, label_dim, self.seq_len, self.embed_dim)
        else:
            # note, a_mlp and v_mlp are only used when there is no modality-sharing layer, otherwise a shared mlp layer is used in s_trans
            self.a_mlp = nn.Sequential(nn.LayerNorm(self.embed_dim, eps=1e-06), nn.Linear(self.embed_dim, label_dim))
            self.v_mlp = nn.Sequential(nn.LayerNorm(self.embed_dim, eps=1e-06), nn.Linear(self.embed_dim, label_dim))

        print('current model embed {:d} s_embed {:d} seq_len {:d} with {:d} modal-specific layers and {:d} model-agnostic layers'
              .format(self.embed_dim, self.s_embed_dim, self.seq_len, self.depth_i, self.depth_s))

    def forward(self, x, modality):
        """ The main forward function
        :param x: input audio or video features, require shape in [batch_size, sequence_len, feat_dim]
        :param modality: the modality indicator, must be either in 'a' and 'v'
        :return: x: output label in shape [batch_size, label_dim]
        """

        # if there's any modality-specific transformer layer.
        if self.depth_i != 0:
            if modality == 'a':
                x = self.a_trans(x, feat=True)
            elif modality == 'v':
                x = self.v_trans(x, feat=True)
            else:
                raise ValueError('input should be either a or v')
        # otherwise if there is no modality-specific layer, go to the input projection linear layer.
        else:
            if modality == 'a':
                x = self.a_in_proj(x)
            elif modality == 'v':
                x = self.v_in_proj(x)
            else:
                raise ValueError('input should be either a or v')

        # if there's any modality-sharing layer
        if self.depth_s != 0:
            x = self.s_trans(x, feat=False)
        # if there's no modality-sharing layer, go to audio- and visual- specific mlp layer to get the prediction logits.
        else:
            # pool the time dimension
            x = torch.mean(x, dim=1)
            if modality == 'a':
                x = self.a_mlp(x)
            elif modality == 'v':
                x = self.v_mlp(x)
            else:
                raise ValueError('input should be either a or v')
        return x

    # x shape in [batch_size, sequence_len, feat_dim]
    def extract_feat(self, x, modality, out_layer=0):
        """ Get the feature of a specific Transformer layer (used for probing test)
        :param x: input audio or video features, require shape in [batch_size, sequence_len, feat_dim]
        :param modality: the modality indicator, must be either in 'a' and 'v'
        :param out_layer: feature of which layer to output
        :return: x: output [batch_size, sequence_len, embed_dim]
        """
        # if out_layer larger than total # layers, then just do a normal forward pass
        if out_layer > self.depth_i + self.depth_s:
            return self.forward(x, modality)
        else:
            # if there is at least one modality-specific layer
            if self.depth_i != 0:
                # if out_layer is larger than the # modality-specific layers, need to first go over modality-specific layers and than modality-sharing layers
                if out_layer > self.depth_i:
                    if modality == 'a':
                        x = self.a_trans(x, feat=True)
                    elif modality == 'v':
                        x = self.v_trans(x, feat=True)
                    else:
                        raise ValueError('input should be either a or v')
                    out = self.s_trans.extract_feat(x, out_layer-self.depth_i)
                    return out
                # if out_layer is smaller or equal to the # modality-specific layers, only need to go over modality-specific layers
                elif out_layer <= self.depth_i:
                    if modality == 'a':
                        x = self.a_trans.extract_feat(x, out_layer)
                    elif modality == 'v':
                        x = self.v_trans.extract_feat(x, out_layer)
                    return x
            # if there is no modality-specific layer
            elif self.depth_i == 0:
                if modality == 'a':
                    x = self.a_in_proj(x)
                elif modality == 'v':
                    x = self.v_in_proj(x)
                else:
                    raise ValueError('input should be either a or v')
                out = self.s_trans.extract_feat(x, out_layer)
                return out

class UnifiedTransVisual(UnifiedTrans):
    """ For visualizing the attention map of UnifiedTrans """
    def forward(self, x, modality):
        att_list = []
        # go to modal-specific transformer if there's any modal-specific layer
        if self.depth_i != 0:
            if modality == 'a':
                x, cur_att_list = self.a_trans.extract_att_map(x, feat=True)
                att_list = att_list + cur_att_list
            elif modality == 'v':
                x, cur_att_list = self.v_trans.extract_att_map(x, feat=True)
                att_list = att_list + cur_att_list
            else:
                raise ValueError('input should be either a or v')
        else:
            if modality == 'a':
                x = self.a_in_proj(x)
            elif modality == 'v':
                x = self.v_in_proj(x)
            else:
                raise ValueError('input should be either a or v')

        if self.depth_s != 0:
            x, cur_att_list = self.s_trans.extract_att_map(x, feat=False)
            att_list = att_list + cur_att_list
        else:
            # pool the time dimension
            x = torch.mean(x, dim=1)
            if modality == 'a':
                x = self.a_mlp(x)
            elif modality == 'v':
                x = self.v_mlp(x)
            else:
                raise ValueError('input should be either a or v')
        return x, att_list

class FullAttTrans(nn.Module):
    """The cross-modal attention model (baseline model in Table 1)"""
    def __init__(self, embed_dim, num_heads, depth_i, depth_s, label_dim, a_seq_len, v_seq_len, a_input_dim, v_input_dim, s_embed_dim=0):
        super().__init__()

        self.a_seq_len, self.v_seq_len = a_seq_len, v_seq_len
        self.a_input_dim, self.v_input_dim = a_input_dim, v_input_dim
        self.depth_i, self.depth_s = depth_i, depth_s

        self.embed_dim = embed_dim
        self.s_embed_dim = self.embed_dim if s_embed_dim == 0 else s_embed_dim

        assert self.a_seq_len == self.v_seq_len
        self.seq_len = self.a_seq_len

        # if no modal-specific layer, still add a projection layer at beginning
        if self.depth_i == 0:
            self.a_in_proj = nn.Sequential(nn.LayerNorm(a_input_dim, eps=1e-06), nn.Linear(self.a_input_dim, embed_dim))
            self.v_in_proj = nn.Sequential(nn.LayerNorm(v_input_dim, eps=1e-06), nn.Linear(self.v_input_dim, embed_dim))
        else:
            self.a_trans = SingleTrans(self.embed_dim, num_heads, self.depth_i, label_dim, self.a_seq_len, self.a_input_dim)
            self.v_trans = SingleTrans(self.embed_dim, num_heads, self.depth_i, label_dim, self.v_seq_len, self.v_input_dim)

        if self.depth_s != 0:
            self.s_trans = SingleTrans(self.s_embed_dim, num_heads, self.depth_s, label_dim, self.a_seq_len + self.v_seq_len, self.embed_dim)
        else:
            self.a_mlp = nn.Sequential(nn.LayerNorm(self.embed_dim, eps=1e-06), nn.Linear(self.embed_dim, label_dim))
            self.v_mlp = nn.Sequential(nn.LayerNorm(self.embed_dim, eps=1e-06), nn.Linear(self.embed_dim, label_dim))

        print('current model embed {:d} s_embed {:d} seq_len {:d} with {:d} modal-specific layers and {:d} full attention layers'
              .format(self.embed_dim, self.s_embed_dim, self.seq_len, self.depth_i, self.depth_s))

    # ax, vx shape in [batch_size, sequence_len, feat_dim]
    def forward(self, ax, vx):
        # go to modal-specific transformer if there's any modal-specific layer
        if self.depth_i != 0:
            ax = self.a_trans(ax, feat=True)
            vx = self.v_trans(vx, feat=True)
        else:
            ax = self.a_in_proj(ax)
            vx = self.v_in_proj(vx)

        # concatenate the two modalities
        x = torch.cat((ax, vx), dim=1)

        if self.depth_s != 0:
            x = self.s_trans(x, feat=False)
        return x

    # x shape in [batch_size, sequence_len, feat_dim]
    def extract_feat(self, x, modality, out_layer=0):
        if out_layer > self.depth_i + self.depth_s:
            return self.forward(x, modality)
        else:
            if self.depth_i != 0:
                if out_layer > self.depth_i:
                    if modality == 'a':
                        x = self.a_trans(x, feat=True)
                    elif modality == 'v':
                        x = self.v_trans(x, feat=True)
                    else:
                        raise ValueError('input should be either a or v')
                    out = self.s_trans.extract_feat(x, out_layer-self.depth_i)
                    return out
                elif out_layer <= self.depth_i:
                    if modality == 'a':
                        x = self.a_trans.extract_feat(x, out_layer)
                    elif modality == 'v':
                        x = self.v_trans.extract_feat(x, out_layer)
                    return x
            elif self.depth_i == 0:
                if modality == 'a':
                    x = self.a_in_proj(x)
                elif modality == 'v':
                    x = self.v_in_proj(x)
                else:
                    raise ValueError('input should be either a or v')
                out = self.s_trans.extract_feat(x, out_layer)
                return out

class FullAttTransVisual(FullAttTrans):
    """For visualizing the attention map of the cross-modal attention model"""
    def forward(self, ax, vx):
        if self.depth_i != 0:
            ax, cur_a_att_list = self.a_trans.extract_att_map(ax, feat=True)
            vx, cur_v_att_list = self.v_trans.extract_att_map(vx, feat=True)
        else:
            ax = self.a_in_proj(ax)
            vx = self.v_in_proj(vx)

        # concatenate the two modalities
        x = torch.cat((ax, vx), dim=1)

        if self.depth_s != 0:
            x, cur_av_att_list = self.s_trans.extract_att_map(x, feat=False)
        if self.depth_i != 0:
            return x, cur_a_att_list, cur_v_att_list, cur_av_att_list
        else:
            return x, None, None, cur_av_att_list

class SeperateTrans(nn.Module):
    """The modal-independent model (baseline model in Table 1)"""
    def __init__(self, embed_dim, num_heads, depth_i, depth_s, label_dim, a_seq_len, v_seq_len, a_input_dim, v_input_dim, s_embed_dim=0):
        super().__init__()

        self.a_seq_len, self.v_seq_len = a_seq_len, v_seq_len
        self.a_input_dim, self.v_input_dim = a_input_dim, v_input_dim
        self.depth_i, self.depth_s = depth_i, depth_s

        self.embed_dim = embed_dim
        self.s_embed_dim = self.embed_dim if s_embed_dim == 0 else s_embed_dim

        assert self.a_seq_len == self.v_seq_len
        self.seq_len = self.a_seq_len

        # if no modal-specific layer, still add a projection layer at beginning
        if self.depth_i == 0:
            self.a_in_proj = nn.Sequential(nn.LayerNorm(a_input_dim, eps=1e-06), nn.Linear(self.a_input_dim, embed_dim))
            self.v_in_proj = nn.Sequential(nn.LayerNorm(v_input_dim, eps=1e-06), nn.Linear(self.v_input_dim, embed_dim))
        else:
            self.a_trans = SingleTrans(self.embed_dim, num_heads, self.depth_i, label_dim, self.a_seq_len, self.a_input_dim)
            self.v_trans = SingleTrans(self.embed_dim, num_heads, self.depth_i, label_dim, self.v_seq_len, self.v_input_dim)

        if self.depth_s != 0:
            # a_s_trans and v_s_trans are also modality-specific
            self.a_s_trans = SingleTrans(self.s_embed_dim, num_heads, self.depth_s, label_dim, self.seq_len, self.embed_dim)
            self.v_s_trans = SingleTrans(self.s_embed_dim, num_heads, self.depth_s, label_dim, self.seq_len, self.embed_dim)
        else:
            self.a_mlp = nn.Sequential(nn.LayerNorm(self.embed_dim, eps=1e-06), nn.Linear(self.embed_dim, label_dim))
            self.v_mlp = nn.Sequential(nn.LayerNorm(self.embed_dim, eps=1e-06), nn.Linear(self.embed_dim, label_dim))

        print('current model embed {:d} s_embed {:d} seq_len {:d} with {:d} modal-specific layers and {:d} model-agnostic layers'
              .format(self.embed_dim, self.s_embed_dim, self.seq_len, self.depth_i, self.depth_s))

    # x shape in [batch_size, sequence_len, feat_dim]
    def forward(self, x, modality):
        # go to modal-specific transformer if there's any modal-specific layer
        if self.depth_i != 0:
            if modality == 'a':
                x = self.a_trans(x, feat=True)
            elif modality == 'v':
                x = self.v_trans(x, feat=True)
            else:
                raise ValueError('input should be either a or v')
        else:
            if modality == 'a':
                x = self.a_in_proj(x)
            elif modality == 'v':
                x = self.v_in_proj(x)
            else:
                raise ValueError('input should be either a or v')

        if self.depth_s != 0:
            if modality == 'a':
                x = self.a_s_trans(x, feat=False)
            elif modality == 'v':
                x = self.v_s_trans(x, feat=False)
            else:
                raise ValueError('input should be either a or v')
        else:
            # pool the time dimension
            x = torch.mean(x, dim=1)
            if modality == 'a':
                x = self.a_mlp(x)
            elif modality == 'v':
                x = self.v_mlp(x)
            else:
                raise ValueError('input should be either a or v')
        return x