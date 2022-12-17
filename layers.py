from functools import partial
import math
from typing import Any, Dict, Tuple
import jax
import flax
from flax import linen as nn
import jax.numpy as jnp
from numpy import block

from tqdm import tqdm

import optax

from sampler import GaussianDiffusionContinuousTimes
from einops import rearrange, repeat, reduce, pack, unpack
from utils import exists, default, jax_unstack, prob_mask_like
from einops_exts import rearrange_many, repeat_many
import partitioning as nnp

from flax.linen import partitioning as nn_partitioning

from config import BlockConfig, UnetConfig, ImagenConfig
from jax.experimental import checkify

with_sharding_constraint = nn_partitioning.with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes

class CheckNan(nn.Module):
    layer: str
    @nn.compact
    def __call__(self, x):
        checkify.check(jnp.isfinite(x).all() >= 0, f"Infinite (infinite) at {self.layer}")
        checkify.check(jnp.max(x) < 100, f"Infinite (max < 1oo) at {self.layer}")
        # checkify.check(jnp.max(x) < 100, f"Infinite (max < 1oo)")
        
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
    block_config: BlockConfig

    @nn.compact
    def __call__(self, x, context=None, mask=None, attn_bias=None):
        b, n = x.shape[:2]

        scale = self.config.dim_heads ** -0.5 # TODO: Implement cosine sim attention
        inner_dim = self.config.dim_heads * self.block_config.num_heads
        x = LayerNorm()(x)
        x = with_sharding_constraint(x, ("batch", "length", "embed"))

        q = nnp.Dense(features=inner_dim, use_bias=False, shard_axes={
                      "kernel": ("heads", "kv")}, dtype=self.config.dtype)(x)
        k, v = nnp.Dense(features=self.config.dim_heads * 2, use_bias=False,
                         shard_axes={"kernel": ("heads", "kv")}, dtype=self.config.dtype)(x).split(2, axis=-1)  # TODO: Check if it should be 2 or 3 kernel shards

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.block_config.num_heads)
        q = q * scale

        q = with_sharding_constraint(q, ("batch", "length", "heads", "kv"))
        k = with_sharding_constraint(k, ("batch", "heads", "kv"))
        v = with_sharding_constraint(v, ("batch", "heads", "kv"))

        null_kv = param_with_axes(
            'null_kv', nn.initializers.lecun_normal(), (2, self.config.dim_heads), axes=("kv", "heads"))
        null_kv = null_kv.astype(self.config.dtype)
        # null kv for classifier free guidance
        nk, nv = repeat_many(jax_unstack(null_kv, axis=-2), 'd -> b 1 d', b=b)
        nk = with_sharding_constraint(nk, ("batch", "heads", "kv"))
        nv = with_sharding_constraint(nv, ("batch", "heads", "kv"))

        k = jnp.concatenate((k, nk), axis=-2)
        v = jnp.concatenate((v, nv), axis=-2)

        k = with_sharding_constraint(k, ("batch", "heads", "kv"))
        v = with_sharding_constraint(v, ("batch", "heads", "kv"))

        if exists(context):
            context_hidden = nnp.LayerNorm()(context)
            context_hidden = nnp.Dense(
                features=self.config.dim_heads*2, shard_axes={"kernel": ("heads", "kv")}, dtype=self.config.dtype)(context_hidden)
            ck, cv = context_hidden.split(2, axis=-1)

            k = jnp.concatenate((k, ck), axis=-2)
            v = jnp.concatenate((v, cv), axis=-2)

        sim = jnp.einsum('b h i d, b j d -> b h i j', q, k)
        if exists(attn_bias):
            sim = sim + attn_bias

        if exists(mask):
            mask = jnp.pad(mask, (1, 0), constant_values=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = jnp.where(mask, -jnp.inf, sim) # TODO: make sure the order of params is correct

        attn = nn.softmax(sim, axis=-1)
        attn.astype(self.config.dtype)
        attn = with_sharding_constraint(attn, ("batch", "length", "heads", "kv"))

        out = jnp.einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        out = nnp.Dense(features=self.block_config.dim, use_bias=False, shard_axes={"kernel": ("heads",  "kv")})(out)
        out = LayerNorm()(out)
        return out


class TransformerBlock(nn.Module):
    config: UnetConfig
    block_config: BlockConfig

    @nn.compact
    def __call__(self, x, context=None):
        # TODO: implement attention_depth
        # TODO: maybe implement pack/unpack
        x = EinopsToAndFrom(Attention(config=self.config, block_config=self.block_config), 'b h w c', 'b (h w) c')(x, context=context) + x
        x = with_sharding_constraint(x, ("batch", "length", "embed"))
        x = ChannelFeedForward(dim=self.block_config.dim, mult=self.config.ff_mult)(x) + x # TODO: Lucidrains uses FeedForward instead of ChannelFeedForward
        return x

class FeedForward(nn.Module):
    dim: int
    mult:int = 2
    
    @nn.compact
    def __call__(self, x):
        hidden_dim = int(self.dim * self.mult)
        x = LayerNorm(axis=-2)(x) # TODO: Figure out if layer norm axes are correct
        x = nn.Dense(hidden_dim, bias = False)(x)
        x = nn.gelu(x)
        x = LayerNorm(axis=-2)(x) # ibid
        x = nn.Dense(self.dim, bias = False)(x)
        return x
        

class ChannelFeedForward(nn.Module):
    dim: int
    mult: int = 2

    @nn.compact
    def __call__(self, x):
        x = ChannelLayerNorm()(x)
        x = nnp.Conv(features=self.dim * self.mult, kernel_size=(1, 1), shard_axes={"kernel": ("width", "height", "mlp")})(x)
        x = nn.gelu(x)
        x = ChannelLayerNorm()(x)
        x = nnp.Conv(features=self.dim, kernel_size=(1, 1), shard_axes={"kernel": ("width", "height", "mlp")})(x)
        return x


class LayerNorm(nn.Module):
    axis: int = -1

    @nn.compact
    def __call__(self, x):
        var = jnp.var(x, axis=self.axis, keepdims=True)
        mean = jnp.mean(x, axis=self.axis, keepdims=True)
        eps:float = 1e-5 if x.dtype == jnp.float32 else 1e-3

        g = param_with_axes('g', nn.initializers.ones, (x.shape[-1], *((1,) * (-self.axis - 1))), axes=("embed",))
        return (x - mean) / jnp.sqrt(var + eps) * g

ChannelLayerNorm = partial(LayerNorm, axis=(-1))
class ChannelLayerNorm2(nn.Module):
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


class CrossAttention(nn.Module):
    config: UnetConfig
    block_config: BlockConfig

    norm_context: bool = False

    @nn.compact
    def __call__(self, x, context, mask=None):
        # TODO: again, implement cosine sim attention
        assert self.block_config.num_heads > 0
        
        scale = self.config.dim_heads ** -0.5
        inner_dim = self.config.dim_heads * self.block_config.num_heads

        b, n = x.shape[:2]
        


        if self.norm_context:
            context = nnp.LayerNorm()(context)
        
        q = nnp.Dense(features=inner_dim, use_bias=False, shard_axes={
                      "kernel": ("heads", "kv")})(x)
        k, v = nnp.Dense(features=inner_dim * 2, use_bias=False, shard_axes={
                         "kernel": ("heads", "kv")})(context).split(2, axis=-1)

        q, k, v = rearrange_many(
            (q, k, v), 'b n (h d) -> b h n d', h=self.block_config.num_heads)

        q = with_sharding_constraint(q, ("batch", "length", "heads", "kv"))
        k = with_sharding_constraint(k, ("batch", "length", "heads", "kv"))
        v = with_sharding_constraint(v, ("batch", "length", "heads", "kv"))

        null_kv = param_with_axes('null_kv', nn.initializers.lecun_normal(),
                                  (2, self.config.dim_heads), axes=("kv", "heads"))

        nk, nv = repeat_many(jax_unstack(null_kv, axis=-2),
                             'd -> b h 1 d', h=self.block_config.num_heads, b=b)

        nk = with_sharding_constraint(nk, ("batch", "length", "heads", "kv"))
        nv = with_sharding_constraint(nv, ("batch", "length", "heads", "kv"))

        k = jnp.concatenate((nk, k), axis=-2)
        v = jnp.concatenate((nv, v), axis=-2)

        k = with_sharding_constraint(k, ("batch", "length", "heads", "kv"))
        v = with_sharding_constraint(v, ("batch", "length", "heads", "kv"))

        q = q * scale

        sim = jnp.einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(mask):
            mask = jnp.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            # TODO check if mask should be inverted and if params are correct
            sim = jnp.where(mask, -jnp.inf, sim)

        attn = nn.softmax(sim, axis=-1)
        attn = with_sharding_constraint(attn, ("batch", "length", "heads", "kv"))
        
        out = jnp.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = with_sharding_constraint(out, ("batch", "length", "heads", "kv"))
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        out = nnp.Dense(features=self.block_config.dim, use_bias=False,
                        shard_axes={"kernel": ("heads", "kv")})(out)
        out = nnp.LayerNorm()(out)
        return out


class SinusoidalPosEmb(nn.Module):
    config: UnetConfig

    @nn.compact
    def __call__(self, time):
        half_dim = self.config.dim//2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=self.config.dtype) * -emb)
        emb = rearrange(time, "i -> i 1") * rearrange(emb, "j -> 1 j")
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1, dtype=self.config.dtype)
    
class LearnedSinusoidalPosEmb(nn.Module):
    config: UnetConfig
    @nn.compact
    def __call__(self, time):
        weights = param_with_axes('pos_emb', nn.initializers.normal(), (self.config.dim,), axes=("embed",))
        time = rearrange(time, "b -> b 1")
        freqs = time * rearrange(weights, "d -> 1 d") * 2 * math.pi
        foruriered = jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)], axis=-1, dtype=self.config.dtype)
        foruriered = jnp.concatenate([time, foruriered], axis=-1, dtype=self.config.dtype)
        return foruriered

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
                kernel, kernel), strides=self.stride, padding=(kernel - self.stride) // 2, dtype=self.dtype, shard_axes={
                    "kernel": ("width", "height", "embed"),
            })(x))

        return jnp.concatenate(convs, axis=-1)


class TextConditioning(nn.Module):
    cond_drop_prob: float = 0.1
    cond_dim: int = 128
    time_cond_dim: int = 128
    max_token_length: int = 256

    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, text_embeds, text_mask, time_cond, time_tokens, rng):
        text_tokens = None
        if exists(text_embeds):
            batch_size = text_embeds.shape[0]
            rng, key = jax.random.split(rng)
            text_keep_mask = prob_mask_like((batch_size,), 1 - self.cond_drop_prob, key)
            text_keep_mask_embed = rearrange(text_keep_mask, 'b -> b 1 1')
            text_keep_mask_hidden = rearrange(text_keep_mask, 'b -> b 1')

            text_tokens = nnp.Dense(features=self.cond_dim, shard_axes={
                                    "kernel": ("embed", "mlp"),
                                    })(text_embeds)
            text_tokens = text_tokens[:, :self.max_token_length]
            
            if exists(text_mask):
                text_mask = text_mask[:, :self.max_token_length]
            
            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_token_length - text_tokens_len
            
            if remainder > 0:
                # the text_tokens shape is (batch_size, max_token_length, cond_dim)
                text_tokens = jnp.pad(text_tokens, ((0, 0), (0, remainder))) # TODO: check how to do padding here
                # text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))
            
            rng, key = jax.random.split(rng)
            if remainder > 0:
                text_mask = jnp.pad(text_mask, (0, remainder), value=False)
                text_mask = rearrange(text_mask, 'b n -> b n 1')
                text_keep_mask_embed = text_mask & text_keep_mask_embed

            null_text_embed = param_with_axes('null_text_embed', nn.initializers.lecun_normal(), (1, self.max_token_length, self.cond_dim), axes=("embed", "mlp"))
            # TODO: should this be inverted?
            text_tokens = jnp.where(
                text_keep_mask_embed, text_tokens, null_text_embed) # TODO: check this too

            # TODO: add attention pooling
            
            mean_pooled_text_tokens = jnp.mean(text_tokens, axis=-2)
            
            text_hiddens = nnp.LayerNorm()(mean_pooled_text_tokens)
            text_hiddens = nnp.Dense(features=self.time_cond_dim, shard_axes={"kernel": ("embed", "mlp")}, dtype=self.dtype)(text_hiddens)
            text_hiddens = nn.silu(text_hiddens)
            text_hiddens = nnp.Dense(features=self.time_cond_dim, shard_axes={"kernel": ("embed", "mlp")}, dtype=self.dtype)(text_hiddens)


            null_text_hidden = param_with_axes(
                'null_text_hidden', nn.initializers.lecun_normal(), (1, self.time_cond_dim), axes=("embed", "mlp"))
            text_hiddens = jnp.where(
                text_keep_mask_hidden, text_hiddens, null_text_hidden)  # same question

            time_cond = time_cond + text_hiddens
        c = time_tokens if not exists(text_embeds) else jnp.concatenate([
            time_tokens, text_tokens], axis=-2)
        c = nnp.LayerNorm()(c)
        return time_cond, c


class Block(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x, scale_shift=None):
        x = nnp.GroupNorm(num_groups=8, shard_axes={"scale": ("embed",), "bias":("embed",)})(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
            x = with_sharding_constraint(
                x, ("batch", "width", "height", "dim"))
        x = nn.silu(x) # TODO: Try swish
        return nnp.Conv(features=self.dim, kernel_size=(3, 3), padding=1, shard_axes=({"kernel": ("width", "height", "mlp")}))(x)


class ResnetBlock(nn.Module):
    """ResNet block with a projection shortcut and batch normalization."""
    config: UnetConfig
    block_config: BlockConfig

    @nn.compact
    def __call__(self, x, time_emb=None, cond=None):
        scale_shift = None
        if exists(time_emb):
            time_emb = nn.silu(time_emb)
            time_emb = nnp.Dense(features=self.block_config.dim * 2, dtype=self.config.dtype,
                                 shard_axes={"kernel": ("embed", "mlp")})(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b 1 1 c')
            scale_shift = jnp.split(time_emb, 2, axis=-1)
        h = Block(self.block_config.dim)(x)
        if exists(cond) and self.block_config.num_heads > 0:
            # TODO: maybe use pack like lucidrains, but maybe Einops is better, at least notationally
            h = EinopsToAndFrom(CrossAttention(config=self.config, block_config=self.block_config),
                                'b h w c', ' b (h w) c')(h, context=cond) + h

        h = Block(self.block_config.dim)(h, scale_shift=scale_shift)
        # TODO: Maybe implement global context like lucidrains
        return h + nnp.Conv(features=self.block_config.dim, kernel_size=(1, 1), padding="same", shard_axes={"kernel": ("width", "height", "mlp")})(x)


class Downsample(nn.Module):
    config: UnetConfig
    block_config: BlockConfig

    @nn.compact
    def __call__(self, x):
        # TODO: Implement the pixel shuffle from lucidrains
        x = rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        x = nnp.Conv(features=self.block_config.dim, kernel_size=(1, 1), shard_axes={"kernel": ("width", "height", "mlp")})(x) # TODO: Check kernel size/padding


class Upsample(nn.Module):
    config: UnetConfig
    block_config: BlockConfig

    @nn.compact
    def __call__(self, x):
        x = jax.image.resize(
            x,
            shape=(x.shape[0], x.shape[1] * 2, x. shape[2] * 2, x.shape[3]),
            method="nearest",
        )
        x = nnp.Conv(features=self.block_config.dim, kernel_size=(5, 5), padding=2, shard_axes={"kernel": ("width", "height", "mlp")})(x) # TODO: Check kernel size/padding
        return x
# TODO: Implement pixel shuffle resampling

class UpsampleCombiner(nn.Module):
    config: UnetConfig
    dim: Tuple[int]
    @nn.compact
    def __call__(self, x, fmaps=None) -> Any:
        blocks = [Block(self.dim) for _ in range(len(fmaps))]
        f_maps = [jax.image.resize(fmaps, shape=(x.shape), method="nearest") for fmap in fmaps]
        outs = [block(fmap) for block, fmap in zip(blocks, f_maps)]
        return jnp.concatenate([x, *outs], axis=-1)
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
                features=self.d, dtype=self.dtype, shard_axes={"kernel": ("embed", "mlp")})(text_embeds_pooled)
            # add axis for height and width
            text_embed_proj = text_embed_proj[:, jnp.newaxis, jnp.newaxis, :]
            # project across height and width
            text_embed_proj = jnp.repeat(text_embed_proj, x.shape[1], axis=1)
            text_embed_proj = jnp.repeat(text_embed_proj, x.shape[2], axis=2)
            # concatinate text_embeds
            x = x + text_embed_proj

        # use layer norm as suggested by the paper
        x = nnp.LayerNorm(dtype=self.dtype)(x)
        return x


class AlternateCrossAttentionBlock(nn.Module):
    num_channels: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, s, a):
        x = nnp.LayerNorm(dtype=self.dtype)(x)  # normalize
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

        return x


class CrossAttentionBlock(nn.Module):
    # attempted to implement cross attention based on scaled dot product attention
    num_channels: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, s, a):
        x = nnp.LayerNorm(dtype=self.dtype)(x)  # normalize
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
        return x
