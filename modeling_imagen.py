import math
from typing import Any, Dict, Tuple
import jax
import flax
from flax import linen as nn
import jax.numpy as jnp

from tqdm import tqdm

import optax

from sampler import GaussianDiffusionContinuousTimes, extract
from einops import rearrange, repeat, reduce, pack, unpack
from utils import exists, default
from layers import ResnetBlock, SinusoidalPositionEmbeddings, CrossEmbedLayer, TextConditioning, TransformerBlock, Downsample, Upsample

class UnetDBlock(nn.Module):
    """UnetD block with a projection shortcut and batch normalization."""
    dim: int # dim
    cond_dim : int # text dim
    time_cond_dim : int # text dim
    
    
    strides: Tuple[int, int]
    num_resnet_blocks: int = 3
    text_cross_attention: bool = False
    num_attention_heads: int = 0

    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, time_emb, conditioning=None):
        x_proj = nn.Conv(features=self.dim, kernel_size=(1, 1),
                strides=self.strides, dtype=self.dtype)(x)
        x = Downsample(dim=self.dim)(x)
        x = ResnetBlock(dim=self.dim, dtype=self.dtype)(x, time_emb, conditioning) # and cond
        for _ in range(self.num_resnet_blocks):
            x = ResnetBlock(dim=self.dim, dtype=self.dtype)(x)
        
        if self.num_attention_heads > 0:
            x = TransformerBlock(dim=self.dim, heads=self.num_attention_heads, dim_head=64, dtype=self.dtype)(x)
        return x + x_proj


class UnetUBlock(nn.Module):
    """UnetU block with a projection shortcut and batch normalization."""
    dim: int # dim
    cond_dim : int # text dim
    time_cond_dim : int # text dim
    
    strides: Tuple[int, int]
    num_resnet_blocks: int = 3
    text_cross_attention: bool = False
    num_attention_heads: int = 0

    dtype: jnp.dtype = jnp.float32
    @nn.compact
    def __call__(self, x, time_emb, conditioning=None):
        x_proj = nn.Conv(features=self.dim, kernel_size=(1, 1),
            strides=1, dtype=self.dtype)(x)

        x = ResnetBlock(dim=self.dim, dtype=self.dtype)(x, time_emb, conditioning) # and cond
        for _ in range(self.num_resnet_blocks):
            x = ResnetBlock(dim=self.dim, time_cond_time=self.time_cond_dim, dtype=self.dtype)(x, time_emb)
            
        if self.num_attention_heads > 0:
            x = TransformerBlock(dim=self.dim, heads=self.num_attention_heads, dim_head=64, dtype=self.dtype)(x)
        x = Upsample(dim=self.dim, dtype=self.dtype)(x + x_proj)
        return x 


class EfficentUNet(nn.Module):
    # config: Dict[str, Any]
    dim: int = 128
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
    num_time_tokens: int = 2
    cond_dim: int = None  # default to dim
    lowres_conditioning: bool = False
    max_token_len: int = 256

    strides: Tuple[int, int] = (2, 2)

    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, time, texts=None, attention_masks=None, condition_drop_prob=0.1, rng=None):
        time_conditioning_dim = self.dim * 4 * \
            (2 if self.lowres_conditioning else 1)
        cond_dim = default(self.cond_dim, self.dim)

        time_hidden = SinusoidalPositionEmbeddings(dim=self.dim)(time)
        time_hidden = nn.Dense(
            features=time_conditioning_dim, dtype=self.dtype)(time_hidden)
        time_hidden = nn.silu(time_hidden)

        t = nn.Dense(features=time_conditioning_dim,
                     dtype=self.dtype)(time_hidden)

        time_tokens = nn.Dense(cond_dim * self.num_time_tokens, dtype=self.dtype)(t)
        time_tokens = rearrange(
            time_tokens, 'b (r d) -> b r d', r=self.num_time_tokens)

        t, c = TextConditioning(cond_dim=cond_dim, time_cond_dim=time_conditioning_dim, max_token_length=self.max_token_len, cond_drop_prob=condition_drop_prob)(texts, attention_masks, t, time_tokens, rng)
        # TODO: add lowres conditioning
        
        x = CrossEmbedLayer(dim=self.dim,
                            kernel_sizes=(3, 7, 15), stride=1, dtype=self.dtype)(x)
        hiddens = [None]

        for dim_mult in self.dim_mults:
            x = UnetDBlock(dim=self.dim * dim_mult, cond_dim=cond_dim, time_cond_dim=time_conditioning_dim,
                           strides=self.strides, dtype=self.dtype)(x, t, c)
            hiddens.append(x)
        hiddens.pop()

        for dim_mult, hidden in zip(reversed(self.dim_mults), reversed(hiddens)):
            x = UnetUBlock(dim=self.dim * dim_mult, cond_dim=cond_dim,  time_cond_dim=time_conditioning_dim,
                           strides=self.strides, dtype=self.dtype)(x, t, c)
            if hidden is not None:
                x = jnp.concatenate([x, hidden], axis=-1)

        x = nn.Dense(features=3, dtype=self.dtype)(x)

        return x    