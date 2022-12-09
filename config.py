from enum import Enum
from typing import List, Iterable, Optional, Union, Tuple, Dict, Any
import jax.numpy as jnp


def ListOrTuple(inner_type):
    return Union[List[inner_type], Tuple[inner_type]]


def SingleOrList(inner_type):
    return Union[inner_type, ListOrTuple(inner_type)]


class UnetConfig:
    dim:                       int = 128
    dim_mults:                 ListOrTuple(int)
    text_embed_dim:            int = 512
    cond_dim:                  int = 128
    channels:                  int = 3
    attn_dim_head:             int = 64
    attn_heads:                int = 16
    
    num_resnet_blocks:         int = 8

    num_time_tokens:           int = 2
    lowres_conditioning:       bool = False
    max_token_len:             int = 256
    

    strides: Tuple[int, int] = (2, 2)
    
    dtype: jnp.bfloat16


class ImagenConfig:
    unets:                  ListOrTuple(UnetConfig)
    image_sizes:            ListOrTuple(int) = (64, 256)
    timesteps:              int = 1024
    
    text_encoder_name:      str = "t5-small"
    
    channels:               int = 3
    loss_type:              str = 'l2'
    cond_drop_prob:         float = 0.5