from enum import Enum
from typing import List, Iterable, Optional, Union, Tuple, Dict, Any
import jax.numpy as jnp
from pydantic import BaseModel
from flax import struct

def ListOrTuple(inner_type):
    return Union[List[inner_type], Tuple[inner_type]]


def SingleOrList(inner_type):
    return Union[inner_type, ListOrTuple(inner_type)]


class UnetConfig(struct.PyTreeNode):
    dim:                       int = 128
    dim_mults:                 Tuple[int] = (1, 2, 4, 8)
    cond_dim:                  int = 128

    time_conditiong_dim:       int = 512  # dim * 4 (* 2 if lowres_conditioning)

    num_time_tokens:           int = 2
    max_token_len:             int = 256
    token_embedding_dim:       int = 512

    channels:                  int = 3

    dim_heads:                 int = 32
    num_heads:                 int = 4
    ff_mult:                   int = 2

    num_resnet_blocks:         int = 6

    lowres_conditioning:       bool = False

    strides: Tuple[int, int] = (2, 2)
    
    scheduler:                 str = struct.field(pytree_node=False, default="cosine")

    dtype:                     Any = struct.field(pytree_node=False, default=jnp.bfloat16)


class ImagenConfig(struct.PyTreeNode):
    unets:                  Tuple[UnetConfig] = (UnetConfig(dim=128, dim_mults=(1, 2, 4, 8), num_resnet_blocks=8, scheduler="cosine"),)
    image_sizes:            Tuple[int] = (64,)
    timesteps:              int = 1000

    text_encoder_name:      str = struct.field(pytree_node=False, default="t5-small")

    channels:               int = 3
    loss_type:              str = struct.field(pytree_node=False, default="l2")
    cond_drop_prob:         float = 0.5
    
    batch_size:             int = 128
