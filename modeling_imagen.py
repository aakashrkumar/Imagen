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


class Block(nn.Module):
    num_channels: int
    @nn.compact
    def __call__(self, x, shift_scale=None):
        x = nn.GroupNorm(group_size=8)(x)
        if exists(shift_scale):
            shift, scale = shift_scale
            x = x * (scale + 1) + shift
        x = nn.swish(x)
        return nn.Conv(features=self.num_channels, kernel_size=(3, 3), padding=1)(x)


class ResnetBlock(nn.Module):
    """ResNet block with a projection shortcut and batch normalization."""
    num_channels: int
    cond_dim : int
    time_cond_time: int=None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, time_emb=None, cond=None):
        scale_shift = None
        if exists(time_emb):
            time_emb = nn.silu(time_emb)
            time_emb = nn.Dense(features=self.num_channels * 2)(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = jnp.split(time_emb, 2, axis=1)
        h = Block(self.num_channels)(x)
        h = CrossAttention(self.num_channels, self.cond_dim, time_cond_time=self.time_cond_time, dtype=self.dtype)(h, cond) + h
        h = Block(self.num_channels)(h, shift_scale=scale_shift)
            
        return h + nn.Conv(features=self.num_channels, kernel_size=(1, 1))(x)


class AlternateCrossAttentionBlock(nn.Module):
    num_channels: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, s, a):

        text_embeds = s
        # repeat mask accross latent dimension
        attention_mask = repeat(a, 'b s -> b s d', d=s.shape[-1])
        # multiply attention mask and text sequence
        text_embeds = text_embeds * attention_mask

        q = nn.Dense(features=self.num_channels, dtype=self.dtype)(x)
        k_x = nn.Dense(features=self.num_channels, dtype=self.dtype)(x)
        k_x = rearrange(k_x, 'b w h c -> b w h 1 c')
        v_x = nn.Dense(features=self.num_channels, dtype=self.dtype)(x)
        v_x = rearrange(v_x, 'b w h c -> b w h 1 c')

        k_s = nn.Dense(features=self.num_channels,
                       dtype=self.dtype)(text_embeds)
        k_s = rearrange(k_s, 'b s d -> b 1 1 s d')
        v_s = nn.Dense(features=self.num_channels,
                       dtype=self.dtype)(text_embeds)
        v_s = rearrange(v_s, 'b s d -> b 1 1 s d')

        k = k_x + k_s
        v = v_x + v_s
        # take the transpose of the v vector
        v = rearrange(v, 'b w h s c -> b w h c s')

        # dot product between v transpose and k
        attention_matrix = jnp.einsum('...ij, ...jk -> ...ik', v, k)
        attention_matrix = attention_matrix / \
            jnp.sqrt(self.num_channels)  # scale the attention matrix
        attention_matrix = nn.softmax(attention_matrix, axis=-1)
        # dot product between queries and attention matrix
        output = jnp.einsum('...ij, ...jk -> ...ik', q, attention_matrix)
        output = reduce(output, 'b w h s c -> b w h c', 'max')
        # reshape channels
        output = nn.Dense(features=x.shape[-1], dtype=self.num_channels)

        x = x + output  # add original information
        x = nn.LayerNorm(dtype=self.dtype)(x)  # normalize

        return x


class CrossAttention(nn.Module):
    # attempted to implement cross attention based on scaled dot product attention
    num_channels: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, s, a):

        text_embeds = s
        # repeat mask accross latent dimension
        attention_mask = repeat(a, 'b s -> b s d', d=s.shape[-1])
        # multiply attention mask and text sequence
        text_embeds = text_embeds * attention_mask

        q = nn.Dense(features=self.num_channels, dtype=self.dtype)(x)
        k_x = nn.Dense(features=self.num_channels, dtype=self.dtype)(x)
        k_x = rearrange(k_x, 'b w h c -> b w h 1 c')
        v_x = nn.Dense(features=x.shape[-1], dtype=self.dtype)(x)
        v_x = rearrange(v_x, 'b w h c -> b w h 1 c')

        k_s = nn.Dense(features=self.num_channels,
                       dtype=self.dtype)(text_embeds)
        k_s = rearrange(k_s, 'b s d -> b 1 1 s d')
        v_s = nn.Dense(features=x.shape[-1], dtype=self.dtype)(text_embeds)
        v_s = rearrange(v_s, 'b s d -> b 1 1 s d')

        k = k_x + k_s
        v = v_x + v_s
        # take the transpose of the k matrix
        k = rearrange(k, 'b w h s c -> b w h c s')

        # dot product between q  and k transpose
        attention_matrix = jnp.einsum('...ij, ...jk -> ...ik', q, k)
        attention_matrix = attention_matrix / \
            jnp.sqrt(self.num_channels)  # scale the attention matrix
        attention_matrix = nn.softmax(attention_matrix, axis=-1)
        # dot product between attention matrix and values
        output = jnp.einsum('...ij, ...jk -> ...ik', attention_matrix, v)
        output = reduce(output, 'b w h s c -> b w h c', 'max')

        x = x + output  # add original information
        x = nn.LayerNorm(dtype=self.dtype)(x)  # normalize

        return x


class SinusoidalPositionEmbeddings(nn.Module):
    dim: int = 512

    @nn.compact
    def __call__(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = jnp.concatenate(
            (jnp.sin(embeddings), jnp.cos(embeddings)), axis=-1)
        return embeddings


class CombineEmbs(nn.Module):
    """Combine positional encoding with text/image encoding."""

    d: int = 32  # should be the dimensions of x
    n: int = 10000  # user defined scalor
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, t, s=None, a=None):
        # timestep encoding, Note t is a tensor of dimension (batch_size,)

        # dimension is nummber of channels
        d = x.shape[-1]
        # create a tensor of dimensions: batch_size x channels
        pe = jnp.zeros((t.shape[0], d))
        # go from t: (batch_size,) to (batch_size,1)
        position = jnp.array([t]).reshape(-1, 1)
        # use the formula n ^ (2*i/d) for i∈2Z (even numbers)
        div_term = jnp.power(self.n, jnp.arange(0, d, 2) / d)
        # set all even indices to sin
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        # set all odd indices to cos
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        # add the height and width channels
        pe = pe[:, jnp.newaxis, jnp.newaxis, :]
        # project accross height and width (spatial dimensions)
        pe = jnp.repeat(pe, x.shape[1], axis=1)
        pe = jnp.repeat(pe, x.shape[2], axis=2)
        # concatinate timestep embeds
        x = x + pe

        # add text/image encoding to x, Note for text, s is a tensor of dimension (batch_size, sequence_length, hidden_latent_size)
        if s is not None:
            text_embeds = s
            # repeat mask accross latent dimension
            attention_mask = repeat(a, 'b s -> b s d', d=s.shape[-1])
            # multiply attention mask and text sequence
            text_embeds = text_embeds * attention_mask
            # mean pooling of sequence with attention mask
            text_embeds_pooled = jnp.sum(text_embeds, axis=1)
            attention_mask_sum = jnp.clip(
                jnp.sum(attention_mask), a_min=1e-9, a_max=None)
            text_embeds_pooled = text_embeds / attention_mask_sum
            # project to correct number of channels
            text_embed_proj = nn.Dense(
                features=self.d, dtype=self.dtype)(text_embeds_pooled)
            # add axis for height and width
            text_embed_proj = text_embed_proj[:, jnp.newaxis, jnp.newaxis, :]
            # project across height and width
            text_embed_proj = jnp.repeat(text_embed_proj, x.shape[1], axis=1)
            text_embed_proj = jnp.repeat(text_embed_proj, x.shape[2], axis=2)
            # concatinate text_embeds
            x = x + text_embed_proj

        # use layer norm as suggested by the paper
        x = nn.LayerNorm(dtype=self.dtype)(x)
        return x


class UnetDBlock(nn.Module):
    """UnetD block with a projection shortcut and batch normalization."""
    num_channels: int
    strides: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32
    num_resnet_blocks: int = 3
    text_cross_attention: bool = False
    num_attention_heads: int = 0

    @nn.compact
    def __call__(self, x, time_emb, conditioning=None):
        # predownsample the input -- EfficientUNet maybe make optional
        x = nn.Conv(features=self.num_channels, kernel_size=(3, 3),
                    strides=self.strides, dtype=self.dtype, padding=1)(x)

        x = ResnetBlock(num_channels=self.num_channels, dtype=self.dtype)(x, time_emb, conditioning) # and cond
        for _ in range(self.num_resnet_blocks):
            x = ResnetBlock(num_channels=self.num_channels, dtype=self.dtype)(x)
        
        if self.num_attention_heads > 0:
            x = nn.SelfAttention(num_heads=self.num_attention_heads, qkv_features=2 *
                                 self.num_channels, out_features=self.num_channels)(x)
        return x


class UnetUBlock(nn.Module):
    """UnetU block with a projection shortcut and batch normalization."""
    num_channels: int
    strides: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32
    num_resnet_blocks: int = 3
    text_cross_attention: bool = False
    num_attention_heads: int = 0

    @nn.compact
    def __call__(self, x, time_emb, conditioning=None):
        x = ResnetBlock(num_channels=self.num_channels, dtype=self.dtype)(x, time_emb, conditioning) # and cond
        for _ in range(self.num_resnet_blocks):
            x = ResnetBlock(num_channels=self.num_channels, dtype=self.dtype)(x, time_emb)
            
        if self.num_attention_heads > 0:
            x = nn.SelfAttention(num_heads=self.num_attention_heads, qkv_features=2 *
                                 self.num_channels, out_features=self.num_channels)(x)

        x = jax.image.resize(
            x,
            shape=(x.shape[0], x.shape[1] * 2, x. shape[2] * 2, x.shape[3]),
            method="nearest",
        )
        x = nn.Conv(features=self.num_channels, kernel_size=(
            3, 3), dtype=self.dtype, padding=1)(x)
        return x


class CrossEmbedLayer(nn.Module):
    dim_out: int = 128
    kernel_sizes: tuple[int, ...] = (3, 7, 15)
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        dim_scales = [int(self.dim_out / (2 ** i))
                      for i in range(1, len(self.kernel_sizes))]
        dim_scales = [*dim_scales, self.dim_out - sum(dim_scales)]
        return jnp.concatenate([nn.Conv(features=dim_scale, kernel_size=(kernel_size, kernel_size), strides=self.stride, padding=(kernel_size - self.stride) // 2)(x) for dim_scale, kernel_size in zip(dim_scales, self.kernel_sizes)], axis=-1)

class TextConditioning(nn.Module):
    cond_drop_prob: float = 0.0   
    cond_dim: int = 128
    time_cond_dim: int = 128
    max_token_length: int = 256
    @nn.compact
    def __call__(self, text_embeds, text_mask, time_cond, time_tokens, rng):
        if exists(text_embeds):
            text_tokens = nn.Dense(features=self.cond_dim)(text_embeds)
            text_tokens = text_tokens[:, :self.max_token_length]
            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_token_length - text_tokens_len
            if remainder > 0:
                text_tokens = jnp.pad(text_tokens, ((0, 0), (0, remainder)))
            rng, key = jax.random.split(rng)
            text_keep_mask = jax.random.uniform(key, (text_tokens.shape[0],)) > self.cond_drop_prob
            text_keep_mask_embed = rearrange(text_keep_mask, 'b -> b 1 1')
            if remainder > 0:
                text_mask = jnp.pad(text_mask, (0, remainder), value=False)
                text_mask = rearrange(text_mask, 'b n -> b n 1')
                text_keep_mask_embed = text_mask & text_keep_mask_embed
            null_text_embed = jax.random.normal(jax.random.PRNGKey(0), (1, self.max_token_length, self.cond_dim))
            text_tokens = jnp.where(text_keep_mask_embed, text_tokens, null_text_embed)
            
            mean_pooled_text_tokens = jnp.mean(text_tokens, axis=-2)
            text_hiddens = nn.LayerNorm()(mean_pooled_text_tokens)
            text_hiddens = nn.Dense(features=self.time_cond_dim)(text_hiddens)
            text_hiddens = nn.silu(text_hiddens)
            text_hiddens = nn.Dense(features=self.time_cond_dim)(text_hiddens)
            
            text_keep_mask_hidden = rearrange(text_keep_mask, 'b -> b 1')
            null_text_hidden = jax.random.normal(jax.random.PRNGKey(1), (1, self.time_cond_dim))
            text_hiddens = jnp.where(text_keep_mask_hidden, text_hiddens, null_text_hidden)
            
            time_cond = time_cond + text_hiddens
        c = time_tokens if not exists(text_embeds) else jnp.concatenate([time_tokens, text_tokens], axis=-2)
        c = nn.LayerNorm()(c)
        return time_cond, c

class EfficentUNet(nn.Module):
    # config: Dict[str, Any]
    dim: int = 128
    dim_mults: tuple = (1, 2, 4, 8),
    num_time_tokens: int = 2
    cond_dim: int = None  # default to dim
    lowres_conditioning: bool = False
    max_token_len: int = 256

    strides: Tuple[int, int] = (2, 2)

    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, time, texts=None, attention_masks=None, rng=None):
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

        t, c = TextConditioning(cond_dim=cond_dim, time_cond_dim=time_conditioning_dim, max_token_length=self.max_token_len)(texts, attention_masks, t, time_tokens, rng)
        # TODO: add lowres conditioning
        

        x = CrossEmbedLayer(dim_out=self.dim,
                            kernel_sizes=(3, 7, 15), stride=1)(x)

        hiddens = []

        for dim_mult in self.dim_mults:
            x = UnetDBlock(num_channels=self.dim * dim_mult,
                           strides=self.strides, dtype=self.dtype)(x, time_tokens, c)
            hiddens.append(x)

        for dim_mult, hidden in zip(reversed(self.dim_mults), reversed(hiddens)):
            x = UnetUBlock(num_channels=self.dim * dim_mult,
                           strides=self.strides, dtype=self.dtype)(x, time_tokens, c)
            x = jnp.concatenate([x, hidden], axis=-1)

        x = nn.Dense(features=3, dtype=self.dtype)(x)

        return x    