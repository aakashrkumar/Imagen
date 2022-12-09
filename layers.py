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
from utils import exists, default, jax_unstack
from einops_exts import rearrange_many, repeat_many
import partitioning as nnp

from flax.linen import partitioning as nn_partitioning

from config import UnetConfig, ImagenConfig

with_sharding_constraint = nn_partitioning.with_sharding_constraint


class EinopsToAndFrom(nn.Module):
    fn: Any
    from_einops: str
    to_einops: str

    @nn.compact
    def __call__(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f"{self.from_einops} -> {self.to_einops}")
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f"{self.to_einops} -> {self.from_einops}", **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    config: UnetConfig
    dim: int

    @nn.compact
    def __call__(self, x, context=None, mask=None, attn_bias=None):
        b, n = x.shape[:2]
        scale = self.config.dim_heads ** -0.5
        inner_dim = self.config.dim_heads * self.config.num_heads

        x = nn.LayerNorm()(x)

        q = nnp.Dense(features=inner_dim, use_bias=False, shard_axes={
                      "kernel": ("embed_kernel", "mlp")})(x)
        k, v = nnp.Dense(features=self.config.dim_heads * 2, use_bias=False,
                         shard_axes={"kernel": ("embed_kernel", "mlp")})(x).split(2, axis=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.config.num_heads)
        q = q * scale

        q = with_sharding_constraint(q, ("batch", "seq", "heads", "embed"))
        k = with_sharding_constraint(k, ("batch", "heads", "embed"))
        v = with_sharding_constraint(v, ("batch", "heads", "embed"))

        null_kv = self.param(
            'null_kv', nn.initializers.lecun_normal(), (2, self.config.dim_heads))
        # null kv for classifier free guidance
        nk, nv = repeat_many(jax_unstack(null_kv, axis=-2), 'd -> b 1 d', b=b)
        nk = with_sharding_constraint(nk, ("batch", "heads", "embed"))
        nv = with_sharding_constraint(nv, ("batch", "heads", "embed"))

        k = jnp.concatenate((k, nk), axis=-2)
        v = jnp.concatenate((v, nv), axis=-2)

        k = with_sharding_constraint(k, ("batch", "heads", "embed"))
        v = with_sharding_constraint(v, ("batch", "heads", "embed"))

        if exists(context):
            context_hidden = nn.LayerNorm()(context)
            context_hidden = nnp.Dense(
                features=self.config.dim_heads*2, shard_axes={"kernel": ("embed_kernel", "mlp")})(context_hidden)
            ck, cv = context_hidden.split(2, axis=-1)

            k = jnp.concatenate((k, ck), axis=-2)
            v = jnp.concatenate((v, cv), axis=-2)

        sim = jnp.einsum('b h i d, b j d -> b h i j', q, k)

        if exists(attn_bias):
            sim = sim + attn_bias

        max_neg_value = -jnp.finfo(sim.dtype).max
        if exists(mask):
            mask = jnp.pad(mask, (1, 0), constant_values=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = jnp.where(mask, sim, max_neg_value)
        attn = nn.softmax(sim, axis=-1)
        out = jnp.einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        out = nnp.Dense(features=self.dim, use_bias=False, shard_axes={
                        "kernel": ("embed_kernel", "mlp")})(out)
        out = nn.LayerNorm()(out)
        return out


class CrossAttention(nn.Module):
    config: UnetConfig

    norm_context: bool = False

    @nn.compact
    def __call__(self, x, context, mask=None):
        scale = self.config.num_heads ** -0.5
        inner_dim = self.config.dim_heads * self.config.num_heads

        b, n = x.shape[:2]
        x = nn.LayerNorm()(x)
        context = nn.LayerNorm()(context)

        q = nnp.Dense(features=inner_dim, use_bias=False, shard_axes={
                      "kernel": ("embed_kernel", "mlp")})(x)
        k, v = nnp.Dense(features=inner_dim * 2, use_bias=False, shard_axes={
                         "kernel": ("embed_kernel", "mlp")})(context).split(2, axis=-1)

        q, k, v = rearrange_many(
            (q, k, v), 'b n (h d) -> b h n d', h=self.config.num_heads)

        q = with_sharding_constraint(q, ("batch", "seq", "heads", "embed"))
        k = with_sharding_constraint(k, ("batch", "seq", "heads", "embed"))
        v = with_sharding_constraint(v, ("batch", "seq", "heads", "embed"))

        null_kv = self.param('null_kv', nn.initializers.lecun_normal(),
                             (2, self.config.dim_heads))

        nk, nv = repeat_many(jax_unstack(null_kv, axis=-2),
                             'd -> b h 1 d', h=self.config.num_heads, b=b)

        nk = with_sharding_constraint(nk, ("batch", "seq", "heads", "embed"))
        nv = with_sharding_constraint(nv, ("batch", "seq", "heads", "embed"))

        k = jnp.concatenate((nk, k), axis=-2)
        v = jnp.concatenate((nv, v), axis=-2)

        k = with_sharding_constraint(k, ("batch", "seq", "heads", "embed"))
        v = with_sharding_constraint(v, ("batch", "seq", "heads", "embed"))

        q = q * scale

        sim = jnp.einsum('b h i d, b h j d -> b h i j', q, k)
        max_neg_value = -jnp.finfo(sim.dtype).max

        if exists(mask):
            mask = jnp.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            # to do, check if mask should be inverted
            sim = jnp.where(mask, sim, max_neg_value)

        attn = nn.softmax(sim, axis=-1)

        out = jnp.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = nnp.Dense(features=self.dim, use_bias=False,
                        shard_axes={"kernel": ("embed_kernel", "mlp")})(out)
        out = nn.LayerNorm()(out)

        return out


class SinusoidalPositionEmbeddings(nn.Module):
    config: UnetConfig

    @nn.compact
    def __call__(self, time):
        half_dim = self.config.dim//2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=self.config.dtype) * -emb)
        emb = rearrange(time, "i -> i 1") * rearrange(emb, "j -> 1 j")
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1, dtype=self.config.dtype)


class CrossEmbedLayer(nn.Module):
    dim: int = 128
    kernel_sizes: Tuple[int, ...] = (3, 7, 15)
    stride: int = 2
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        kernel_sizes = sorted(self.kernel_sizes)
        num_scales = len(self.kernel_sizes)

        dim_scales = [int(self.dim / (2 ** i))
                      for i in range(1, num_scales)]
        dim_scales = dim_scales + [self.dim - sum(dim_scales)]
        convs = []
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            convs.append(nnp.Conv(features=dim_scale, kernel_size=(
                kernel, kernel), strides=self.stride, padding=(kernel - self.stride) // 2, dtype=self.dtype)(x))

        return jnp.concatenate(convs, axis=-1)


class TextConditioning(nn.Module):
    cond_drop_prob: float = 0.1
    cond_dim: int = 128
    time_cond_dim: int = 128
    max_token_length: int = 256

    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, text_embeds, text_mask, time_cond, time_tokens, rng):
        if exists(text_embeds):
            text_tokens = nnp.Dense(features=self.cond_dim, shard_axes={
                                    "kernel": ("embed_kernel", "mlp")})(text_embeds)
            text_tokens = text_tokens[:, :self.max_token_length]
            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_token_length - text_tokens_len
            if remainder > 0:
                text_tokens = jnp.pad(text_tokens, ((0, 0), (0, remainder)))
            rng, key = jax.random.split(rng)
            text_keep_mask = jax.random.uniform(
                key, (text_tokens.shape[0],)) > self.cond_drop_prob
            text_keep_mask_embed = rearrange(text_keep_mask, 'b -> b 1 1')
            if remainder > 0:
                text_mask = jnp.pad(text_mask, (0, remainder), value=False)
                text_mask = rearrange(text_mask, 'b n -> b n 1')
                text_keep_mask_embed = text_mask & text_keep_mask_embed

            null_text_embed = self.param('null_text_embed', nn.initializers.lecun_normal(
            ), (1, self.max_token_length, self.cond_dim))
            # TODO: should this be inverted?
            text_tokens = jnp.where(
                text_keep_mask_embed, text_tokens, null_text_embed)

            mean_pooled_text_tokens = jnp.mean(text_tokens, axis=-2)
            text_hiddens = nn.LayerNorm()(mean_pooled_text_tokens)
            text_hiddens = nnp.Dense(features=self.time_cond_dim, shard_axes={
                                     "kernel": ("embed_kernel", "mlp")}, dtype=self.dtype)(text_hiddens)
            text_hiddens = nn.silu(text_hiddens)
            text_hiddens = nnp.Dense(features=self.time_cond_dim, shard_axes={
                                     "kernel": ("embed_kernel", "mlp")}, dtype=self.dtype)(text_hiddens)

            text_keep_mask_hidden = rearrange(text_keep_mask, 'b -> b 1')

            null_text_hidden = self.param(
                'null_text_hidden', nn.initializers.lecun_normal(), (1, self.time_cond_dim))
            text_hiddens = jnp.where(
                text_keep_mask_hidden, text_hiddens, null_text_hidden)  # same question

            time_cond = time_cond + text_hiddens
        c = time_tokens if not exists(text_embeds) else jnp.concatenate([
            time_tokens, text_tokens], axis=-2)
        c = nn.LayerNorm()(c)
        return time_cond, c


class TransformerBlock(nn.Module):
    config: UnetConfig
    dim: int

    @nn.compact
    def __call__(self, x, context=None):
        x = EinopsToAndFrom(Attention(config=self.config, dim=self.dim),
                            'b h w c', 'b (h w) c')(x, context=context) + x
        x = ChannelFeedForward(dim=self.dim, mult=self.config.ff_mult)(x) + x
        return x


class ChannelFeedForward(nn.Module):
    dim: int
    mult: int = 2

    @nn.compact
    def __call__(self, x):
        x = ChannelLayerNorm(dim=self.dim)(x)
        x = nn.Conv(features=self.dim * self.mult, kernel_size=(1, 1))(x)
        x = nn.gelu(x)
        x = ChannelLayerNorm(dim=self.dim * self.mult)(x)
        x = nn.Conv(features=self.dim, kernel_size=(1, 1))(x)
        return x


class ChannelLayerNorm(nn.Module):
    """
    LayerNorm for :class:`.ChanFeedForward`.
    """
    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        var = jnp.var(x, axis=-1, keepdims=True)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + self.eps) * jnp.ones((1, 1, 1, self.dim))


class Block(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x, shift_scale=None):
        x = nn.GroupNorm(num_groups=8)(x)
        if exists(shift_scale):
            shift, scale = shift_scale
            x = x * (scale + 1) + shift
            x = with_sharding_constraint(
                x, ("batch", "width", "height", "dim"))
        x = nn.silu(x)
        return nnp.Conv(features=self.dim, kernel_size=(3, 3), padding=1)(x)


class ResnetBlock(nn.Module):
    """ResNet block with a projection shortcut and batch normalization."""
    config: UnetConfig
    dim: int

    @nn.compact
    def __call__(self, x, time_emb=None, cond=None):
        scale_shift = None
        if exists(time_emb):
            time_emb = nn.silu(time_emb)
            time_emb = nnp.Dense(features=self.dim * 2, dtype=self.config.dtype,
                                 shard_axes={"kernel": ("embed_kernel", "mlp")})(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b 1 1 c')
            scale_shift = jnp.split(time_emb, 2, axis=-1)
        h = Block(self.dim)(x)
        if exists(cond):
            h = EinopsToAndFrom(CrossAttention(dim=self.dim),
                                'b h w c', ' b (h w) c')(h, context=cond) + h

        h = Block(self.dim)(h, shift_scale=scale_shift)
        # padding was not same
        return h + nn.Conv(features=self.dim, kernel_size=(1, 1), padding="same")(x)


class Downsample(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Conv(features=self.dim, kernel_size=(5, 5), strides=(2, 2), padding=2)(x)


class Upsample(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        x = jax.image.resize(
            x,
            shape=(x.shape[0], x.shape[1] * 2, x. shape[2] * 2, x.shape[3]),
            method="nearest",
        )
        x = nn.Conv(features=self.dim, kernel_size=(5, 5), padding=2)(x)
        return x


class CombineEmbs(nn.Module):
    """Combine positional encoding with text/image encoding."""

    d: int = 32  # should be the dimensions of x
    n: int = 10000  # user defined scalor
    dtype: jnp.dtype = jnp.bfloat16

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
            text_embed_proj = nnp.Dense(
                features=self.d, dtype=self.dtype, shard_axes={"kernel": ("embed_kernel", "mlp")})(text_embeds_pooled)
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


class AlternateCrossAttentionBlock(nn.Module):
    num_channels: int
    dtype: jnp.dtype = jnp.bfloat16

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


class CrossAttentionBlock(nn.Module):
    # attempted to implement cross attention based on scaled dot product attention
    num_channels: int
    dtype: jnp.dtype = jnp.bfloat16

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
