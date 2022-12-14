from typing import Any, Dict, Tuple
from flax import linen as nn
import jax.numpy as jnp

from einops import rearrange, repeat, reduce, pack, unpack
from utils import exists, default
from layers import ResnetBlock, SinusoidalPositionEmbeddings, CrossEmbedLayer, TextConditioning, TransformerBlock, Downsample, Upsample, Attention, EinopsToAndFrom
from jax.experimental.pjit import PartitionSpec as P
import partitioning as nnp
from flax.linen import partitioning as nn_partitioning

from config import UnetConfig, ImagenConfig

with_sharding_constraint = nn_partitioning.with_sharding_constraint
scan_with_axes = nn_partitioning.scan_with_axes
ScanIn = nn_partitioning.ScanIn

class EfficentUNet(nn.Module):
    config: UnetConfig

    @nn.compact
    def __call__(self, x: jnp.array, time, texts=None, attention_masks=None, condition_drop_prob=0.0, lowres_cond_img=None, lowres_noise_times=None, rng=None):
        if self.config.lowres_conditioning:
            assert exists(lowres_cond_img) and exists(lowres_noise_times), "lowres_cond_img and lowres_noise_times must be not None if lowres_conditioning is True"
        else:
            assert not exists(lowres_cond_img) and not exists(lowres_noise_times), "lowres_cond_img and lowres_noise_times must be None if lowres_conditioning is False"

        x = x.astype(self.config.dtype)
        time = time.astype(self.config.dtype)
        if exists(texts):
            texts = texts.astype(self.config.dtype)
        if exists(attention_masks):
            attention_masks = attention_masks.astype(self.config.dtype)

        x = with_sharding_constraint(x, P("batch", "height", "width", "embed"))
        texts = with_sharding_constraint(texts, P("batch", "length", "embed"))

        if exists(lowres_cond_img):
            x = jnp.concatenate([x, lowres_cond_img], axis=-1)
            x = with_sharding_constraint(x, P("batch", "height", "width", "embed"))
            x = x.astype(self.config.dtype)

        cond_dim = default(self.config.cond_dim, self.config.dim)

        time_hidden = SinusoidalPositionEmbeddings(config=self.config)(time)  # (b, 1, d)
        time_hidden = nnp.Dense(features=self.config.time_conditiong_dim, shard_axes={"kernel": ("embed", "mlp")})(time_hidden)
        time_hidden = nn.silu(time_hidden)

        t = nnp.Dense(features=self.config.time_conditiong_dim,
                      dtype=self.config.dtype, shard_axes={"kernel": ("embed", "mlp")})(time_hidden)

        t = with_sharding_constraint(t, ("batch", "embed"))

        time_tokens = nnp.Dense(self.config.cond_dim * self.config.num_time_tokens, shard_axes={"kernel": ("embed", "mlp")})(t)
        time_tokens = rearrange(time_tokens, 'b (r d) -> b r d', r=self.config.num_time_tokens)

        time_tokens = with_sharding_constraint(time_tokens, P("batch", "seq", "embed"))
        if self.config.lowres_conditioning:
            lowres_time_hiddens = SinusoidalPositionEmbeddings(config=self.config)(lowres_noise_times)  # (b, 1, d)
            lowres_time_hiddens = nnp.Dense(features=self.config.time_conditiong_dim, shard_axes={"kernel": ("embed", "mlp")})(lowres_time_hiddens)
            lowres_time_hiddens = nn.silu(lowres_time_hiddens)
            lowres_time_tokens = nnp.Dense(self.config.cond_dim * self.config.num_time_tokens, shard_axes={"kernel": ("embed", "mlp")})(lowres_time_hiddens)
            lowres_time_tokens = rearrange(lowres_time_tokens, 'b (r d) -> b r d', r=self.config.num_time_tokens)
            lowres_t = nnp.Dense(features=self.config.time_conditiong_dim, dtype=self.config.dtype, shard_axes={"kernel": ("embed", "mlp")})(lowres_time_hiddens)

            t = t + lowres_t
            time_tokens = jnp.concatenate([time_tokens, lowres_time_tokens], axis=-2)

        t, c = TextConditioning(cond_dim=cond_dim, time_cond_dim=self.config.time_conditiong_dim, max_token_length=self.config.max_token_len, cond_drop_prob=condition_drop_prob)(texts, attention_masks, t, time_tokens, rng)
        # TODO: add lowres conditioning

        t = with_sharding_constraint(t, ("batch", "embed"))
        c = with_sharding_constraint(c, ("batch", "embed"))

        x = CrossEmbedLayer(dim=self.config.dim,
                            kernel_sizes=(3, 7, 15), stride=1)(x)
        x = with_sharding_constraint(x, ("batch", "height", "width", "embed"))

        init_conv_residual = x
        # downsample
        hiddens = []
        for block_config in self.config.block_configs:
            x = Downsample(config=self.config, block_config=block_config)(x)
            x = ResnetBlock(config=self.config, block_config=block_config)(x, t, c)
            for _ in range(block_config.num_resnet_blocks):
                x = ResnetBlock(config=self.config, block_config=block_config)(x)
                x = with_sharding_constraint(x, ("batch", "height", "width", "embed"))
                hiddens.append(x)
            x = TransformerBlock(config=self.config, block_config=block_config)(x)
            x = with_sharding_constraint(x, ("batch", "height", "width", "embed"))
            hiddens.append(x)
        
        # middle
        block_config = self.config.block_configs[-1]
        x = ResnetBlock(config=self.config, block_config=block_config)(x, t, c)
        x = EinopsToAndFrom(Attention(config=self.config, block_config=block_config), 'b h w c', 'b (h w) c')(x)
        x = ResnetBlock(config=self.config, block_config=block_config)(x, t, c)
        
        # Upsample
        add_skip_connection = lambda x: jnp.concatenate([x, hiddens.pop()], axis=-1)
        for block_config in reversed(self.config.block_configs):
            x = add_skip_connection(x)
            x = ResnetBlock(config=self.config, block_config=block_config)(x, t, c)
            for _ in range(block_config.num_resnet_blocks):
                x = add_skip_connection(x)
                x = with_sharding_constraint(x, P("batch", "height", "width", "embed"))
                x = ResnetBlock(config=self.config, block_config=block_config)(x)
                x = with_sharding_constraint(x, P("batch", "height", "width", "embed"))
            
            x = TransformerBlock(config=self.config, block_config=block_config)(x)
            x = Upsample(config=self.config, block_config=block_config)(x)
        
        x = jnp.concatenate([x, init_conv_residual], axis=-1)
        
        x = ResnetBlock(config=self.config, block_config=block_config)(x, t, c)
            
        # x = nn.Dense(features=3, dtype=self.dtype)(x)
        x = nnp.Conv(features=3, kernel_size=(3, 3), strides=1, dtype=self.config.dtype, padding=1, shard_axes={"kernel": ("width", "height", "embed")})(x)
        return x    