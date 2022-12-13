from enum import Enum
from typing import List, Iterable, Optional, Union, Tuple, Dict, Any
import jax.numpy as jnp
from pydantic import BaseModel
from flax import struct

def ListOrTuple(inner_type):
    return Union[List[inner_type], Tuple[inner_type]]


def SingleOrTuple(inner_type):
    return Union[inner_type, Tuple[inner_type]]

class BlockConfig(struct.PyTreeNode):
    dim:                       int = 128
    num_heads:                 int = 4
    num_resnet_blocks:         int = 8
    self_attention:            bool = True
    cross_attention:           bool = True
        

class UnetConfig(struct.PyTreeNode):
    dim:                       int = 128
    dim_mults:                 Tuple[int] = (1, 2, 4, 8)
    cond_dim:                  int = 128

    time_conditiong_dim:       int = 512  # dim * 4 (* 2 if lowres_conditioning)

    num_time_tokens:           int = 2
    max_token_len:             int = 256
    token_embedding_dim:       int = 512

    channels:                  int = 3

    dim_heads:                 int = 64
    num_heads:                 SingleOrTuple(int) = 4
    ff_mult:                   int = 2

    # num_resnet_blocks:       int = 8

    lowres_conditioning:       bool = False

    strides:                   Tuple[int, int] = (2, 2)
    
    block_configs:             Tuple[BlockConfig] = None
    
    scheduler:                 str = struct.field(pytree_node=False, default="cosine")

    dtype:                     Any = struct.field(pytree_node=False, default=jnp.bfloat16)
    
    @classmethod
    def create(self, *,
            dim:                    int,
            dim_mults:              Tuple[int],
            
            cond_dim:               int=None,
            time_cond_dim:          int=None,
            
            num_heads:              SingleOrTuple(int)=4,
            num_resnet_blocks:      SingleOrTuple(int)=8,
            
            lowres_conditioning:    bool=False,
            scheduler:              str="cosine",
            dtype:                  Any=jnp.bfloat16,
        ):
        block_configs = []
        for i in range(len(dim_mults)):
            n_heads = num_heads if isinstance(num_heads, int) else num_heads[i]
            n_resnet_blocks = num_resnet_blocks if isinstance(num_resnet_blocks, int) else num_resnet_blocks[i]
            block_configs.append(BlockConfig(
                dim=dim * dim_mults[i],
                num_heads=n_heads,
                num_resnet_blocks=n_resnet_blocks,
            ))
        if cond_dim is None:
            cond_dim = dim
        if time_cond_dim is None:
            time_cond_dim = dim * 4
            if lowres_conditioning:
                time_cond_dim *= 2
            
        return UnetConfig(
            dim=dim,
            dim_mults=dim_mults,
            cond_dim=cond_dim,
            
            time_conditiong_dim=time_cond_dim,
            
            lowres_conditioning=lowres_conditioning,
            block_configs=tuple(block_configs),
            scheduler=scheduler,
            dtype=dtype,
        )
        
        
        

class ImagenConfig(struct.PyTreeNode):
    unets:                  Tuple[UnetConfig] = (
                                UnetConfig.create(dim=128, dim_mults=(1, 2, 4, 8), num_heads=(0, 2, 4, 8), num_resnet_blocks=3, 
                                                  scheduler="cosine", lowres_conditioning=False, dtype=jnp.bfloat16),
                    #            UnetConfig.create(dim=128, dim_mults=(1, 2, 4, 8), num_heads=(0, 0, 0, 8), num_resnet_blocks=(2, 4, 8, 8), 
                                       #           scheduler="cosine", lowres_conditioning=True, dtype=jnp.bfloat16),
                            )
    image_sizes:            Tuple[int] = (
                                64,
                               # 256,
                            )
    timesteps:              int = 1000

    text_encoder_name:      str = struct.field(pytree_node=False, default="t5-small")

    channels:               int = 3
    loss_type:              str = struct.field(pytree_node=False, default="l2")
    cond_drop_prob:         float = 0.5
    
    batch_size:             int = 128

