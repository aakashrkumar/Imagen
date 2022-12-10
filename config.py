from enum import Enum
from typing import List, Iterable, Optional, Union, Tuple, Dict, Any
import jax.numpy as jnp


def ListOrTuple(inner_type):
    return Union[List[inner_type], Tuple[inner_type]]


def SingleOrList(inner_type):
    return Union[inner_type, ListOrTuple(inner_type)]


class UnetConfig:
    dim:                       int = 128
    dim_mults:                 ListOrTuple(int) = (1, 2, 4, 8)
    cond_dim:                  int = 128

    time_conditiong_dim:       int = 512  # dim * 4 (* 2 if lowres_conditioning)

    num_time_tokens:           int = 2
    max_token_len:             int = 256
    token_embedding_dim:       int = 512

    channels:                  int = 3

    dim_heads:                 int = 32
    num_heads:                 int = 4
    ff_mult:                   int = 2

    num_resnet_blocks:         int = 3

    lowres_conditioning:       bool = False

    strides: Tuple[int, int] = (2, 2)

    dtype:                     Any = jnp.bfloat16

    def __init__(self, dim: int=None, dim_mults: ListOrTuple(int)=None, cond_dim: int=None):
        if dim is not None:
            self.dim = dim
        if dim_mults is not None:
            self.dim_mults = dim_mults
        if cond_dim is not None:
            self.cond_dim = cond_dim


class ImagenConfig:
    unets:                  ListOrTuple(UnetConfig)
    image_sizes:            ListOrTuple(int) = (64, 256)
    timesteps:              int = 1000

    text_encoder_name:      str = "t5-small"

    channels:               int = 3
    loss_type:              str = 'l2'
    cond_drop_prob:         float = 0.5
    
    batch_size:             int = 128

    @classmethod
    def create(cls, image_sizes: ListOrTuple(int), dims: ListOrTuple(int), batch_size:int):
        return cls(
            unets=[UnetConfig(dim=dim) for dim in dims],
            image_sizes=image_sizes
        )
