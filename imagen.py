from typing import Tuple
import jax
import flax
import optax
from flax import linen as nn


class Unet(nn.Module):
    channels:Tuple[int, ...] = (16, 4) 
    
    