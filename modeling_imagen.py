from typing import Any, Dict, Tuple
import jax
import flax
from flax import linen as nn
import jax.numpy as jnp

from tqdm import tqdm

import optax

from sampler import GaussianDiffusionContinuousTimes, extract
from einops import rearrange, repeat, reduce, pack, unpack


class ResNetBlock(nn.Module):
    """ResNet block with a projection shortcut and batch normalization."""
    num_layers: int
    num_channels: int
    strides: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32
    training: bool = True

    @nn.compact
    def __call__(self, x):
        # Iterate over the number of layers.
        for _ in range(self.num_layers):
            # Save the input for the residual connection.
            residual = x

            # Normalization, swish, and convolution.
            x = nn.GroupNorm(dtype=self.dtype)(x)
            x = nn.swish(x)
            x = nn.Conv(self.num_channels, kernel_size=(3, 3),
                        dtype=self.dtype, padding="same")(x)

            # Normalization, swish, and convolution.
            x = nn.GroupNorm(dtype=self.dtype)(x)
            x = nn.swish(x)
            x = nn.Conv(self.num_channels, kernel_size=(3, 3),
                        dtype=self.dtype, padding="same")(x)

            # Projection shortcut.
            residual = nn.Conv(features=self.num_channels,
                               kernel_size=(1, 1), dtype=self.dtype)(residual)

            # Add the residual connection.
            x = x + residual
        return x


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

        k_s = nn.Dense(features=self.num_channels, dtype=self.dtype)(text_embeds)
        k_s = rearrange(k_s, 'b s d -> b 1 1 s d')
        v_s = nn.Dense(features=self.num_channels, dtype=self.dtype)(text_embeds)
        v_s = rearrange(v_s, 'b s d -> b 1 1 s d')

        
        k = k_x + k_s
        v = v_x + v_s
        v = rearrange(v, 'b w h s c -> b w h c s') # take the transpose of the v vector

        attention_matrix = jnp.einsum('...ij, ...jk -> ...ik', v, k) # dot product between v transpose and k
        attention_matrix = attention_matrix / jnp.sqrt(self.num_channels) # scale the attention matrix
        attention_matrix = nn.softmax(attention_matrix, axis=-1)
        output = jnp.einsum('...ij, ...jk -> ...ik', q, attention_matrix) # dot product between queries and attention matrix
        output = reduce(output, 'b w h s c -> b w h c', 'max')
        output = nn.Dense(features=x.shape[-1], dtype=self.num_channels) # reshape channels

        x = x + output # add original information
        x = nn.LayerNorm(dtype=self.dtype)(x) # normalize

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

        k_s = nn.Dense(features=self.num_channels, dtype=self.dtype)(text_embeds)
        k_s = rearrange(k_s, 'b s d -> b 1 1 s d')
        v_s = nn.Dense(features=x.shape[-1], dtype=self.dtype)(text_embeds)
        v_s = rearrange(v_s, 'b s d -> b 1 1 s d')

        
        k = k_x + k_s
        v = v_x + v_s
        k = rearrange(k, 'b w h s c -> b w h c s') # take the transpose of the k vector

        attention_matrix = jnp.einsum('...ij, ...jk -> ...ik', q, k) # dot product between v transpose and k
        attention_matrix = attention_matrix / jnp.sqrt(self.num_channels) # scale the attention matrix
        attention_matrix = nn.softmax(attention_matrix, axis=-1)
        output = jnp.einsum('...ij, ...jk -> ...ik', attention_matrix, v) # dot product between queries and attention matrix
        output = reduce(output, 'b w h s c -> b w h c', 'max')

        x = x + output # add original information
        x = nn.LayerNorm(dtype=self.dtype)(x) # normalize

        return x


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
            attention_mask_sum = jnp.clip(jnp.sum(attention_mask), a_min=1e-9, a_max=None)
            text_embeds_pooled = text_embeds / attention_mask_sum
            # project to correct number of channels
            text_embed_proj = nn.Dense(features=self.d, dtype=self.dtype)(text_embeds_pooled)
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
    def __call__(self, x, time, texts=None, attention_masks=None):
        x = nn.Conv(features=self.num_channels, kernel_size=(3, 3),
                    strides=self.strides, dtype=self.dtype, padding=1)(x)        
        x = CombineEmbs()(x, time)
        if self.text_cross_attention and texts is not None:
            x = CrossAttention(num_channels=self.num_channels, dtype=self.dtype)(x, texts, attention_masks)

        x = ResNetBlock(num_layers=self.num_resnet_blocks,
                        num_channels=self.num_channels, strides=self.strides, dtype=self.dtype)(x)
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
    def __call__(self, x, time, texts = None, attention_masks=None):
        x = CombineEmbs()(x, time)
        x = ResNetBlock(num_layers=self.num_resnet_blocks,
                        num_channels=self.num_channels, strides=self.strides, dtype=self.dtype)(x)
        if self.text_cross_attention and texts is not None:
            x = CrossAttention(num_channels=self.num_channels, dtype=self.dtype)(x, texts, attention_masks)
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


class EfficentUNet(nn.Module):
    # config: Dict[str, Any]
    strides: Tuple[int, int] = (2, 2)
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, time, texts=None, attention_masks=None):
        x = nn.Conv(features=128, kernel_size=(3, 3),
                    dtype=self.dtype, padding="same")(x)

        uNet256D = UnetDBlock(num_channels=128, strides=self.strides,
                              num_resnet_blocks=3, dtype=self.dtype)(x, time)
        uNet128D = UnetDBlock(num_channels=256, strides=self.strides,
                              num_resnet_blocks=3, num_attention_heads=8, dtype=self.dtype)(uNet256D, time)
        uNet64D = UnetDBlock(num_channels=512,  strides=self.strides,
                             num_resnet_blocks=3, num_attention_heads=8, dtype=self.dtype)(uNet128D, time)
        uNet32D = UnetDBlock(num_channels=1024, strides=self.strides,
                             num_resnet_blocks=3, num_attention_heads=8, dtype=self.dtype)(uNet64D, time)

        uNet32U = UnetUBlock(num_channels=1024, strides=self.strides,
                             num_resnet_blocks=3, num_attention_heads=8, dtype=self.dtype)(uNet32D, time)
        uNet64U = UnetUBlock(num_channels=512, strides=self.strides,
                             num_resnet_blocks=3, num_attention_heads=8, dtype=self.dtype)(jnp.concatenate([uNet32U, uNet64D], axis=-1), time)
        uNet128U = UnetUBlock(num_channels=256, strides=self.strides,
                              num_resnet_blocks=3, num_attention_heads=8, dtype=self.dtype)(jnp.concatenate([uNet64U, uNet128D], axis=-1), time)
        uNet256U = UnetUBlock(num_channels=128, strides=self.strides,
                              num_resnet_blocks=3, dtype=self.dtype)(jnp.concatenate([uNet128U, uNet256D], axis=-1), time)

        x = nn.Dense(features=3, dtype=self.dtype)(uNet256U)

        return x