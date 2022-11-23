import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange, reduce, repeat

b = 1
h = 10
w = 10
d = 15

# x = np.ones((b, h, w, d))
# s = np.ones((b, ))

x = np.ones((1, 15))
s = np.ones((21, 15))

q = x
v = s + x
k = s + x

y = np.dot(v.transpose(), k)
y = np.dot(q, y)
print(y.shape)

y = rearrange(x, 'b w h c -> b w h 1 c')
z = np.einsum('b w h s c, b s c -> b w h s c', y, s)


class CrossAttention(nn.Module):
    num_channels: int
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x, s):
        q = nn.Dense(features=self.num_channels, dtype=self.dtype)(x)
        k_x = nn.Dense(features=self.num_channels, dtype=self.dtype)(x)
        k_x = rearrange(k_x, 'b w h c -> b w h 1 c')
        v_x = nn.Dense(features=self.num_channels, dtype=self.dtype)(x)
        v_x = rearrange(v_x, 'b w h c -> b w h 1 c')

        k_s = nn.Dense(features=self.num_channels, dtype=self.dtype)(s)
        k_s = rearrange(k_s, 'b s d -> b 1 1 s d')
        v_s = nn.Dense(features=self.num_channels, dtype=self.dtype)(s)
        v_s = rearrange(v_s, 'b s d -> b 1 1 s d')

        
        k = k_x + k_s
        v = v_x + v_s
        v = rearrange(v, 'b w h s c -> b w h c s')

        attention_matrix = jnp.einsum('...ij, ...jk -> ...ik', v, k) # dot product between v transpose and k
        attention_matrix = attention_matrix / jnp.sqrt(self.num_channels)
        attention_matrix = nn.softmax(attention_matrix, axis=-1)
        output = jnp.einsum('...ij, ...jk -> ...ik', q, attention_matrix) # dot product between queries and attention matrix
        output = reduce(z1, 'b w h s c -> b w h c', 'max')
        output = nn.Dense(features=x.shape[-1], dtype=self.num_channels)

        x = x + output

        return x

# class CrossAttention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         *,
#         context_dim = None,
#         dim_head = 64,
#         heads = 8,
#         norm_context = False,
#         cosine_sim_attn = False
#     ):
#         super().__init__()
#         self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1.
#         self.cosine_sim_attn = cosine_sim_attn
#         self.cosine_sim_scale = 16 if cosine_sim_attn else 1

#         self.heads = heads
#         inner_dim = dim_head * heads

#         context_dim = default(context_dim, dim)

#         self.norm = LayerNorm(dim)
#         self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

#         self.null_kv = nn.Parameter(torch.randn(2, dim_head))
#         self.to_q = nn.Linear(dim, inner_dim, bias = False)
#         self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim, bias = False),
#             LayerNorm(dim)
#         )

#     def forward(self, x, context, mask = None):
#         b, n, device = *x.shape[:2], x.device

#         x = self.norm(x)
#         context = self.norm_context(context)

#         q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

#         q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = self.heads)

#         # add null key / value for classifier free guidance in prior net

#         nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b h 1 d', h = self.heads,  b = b)

#         k = torch.cat((nk, k), dim = -2)
#         v = torch.cat((nv, v), dim = -2)

#         q = q * self.scale

#         # cosine sim attention

#         if self.cosine_sim_attn:
#             q, k = map(l2norm, (q, k))

#         # similarities

#         sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.cosine_sim_scale

#         # masking

#         max_neg_value = -torch.finfo(sim.dtype).max

#         if exists(mask):
#             mask = F.pad(mask, (1, 0), value = True)
#             mask = rearrange(mask, 'b j -> b 1 1 j')
#             sim = sim.masked_fill(~mask, max_neg_value)

#         attn = sim.softmax(dim = -1, dtype = torch.float32)
#         attn = attn.to(sim.dtype)

#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)