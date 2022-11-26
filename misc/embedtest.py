import numpy as np

# d = x.shape[-1]
# pe = jnp.zeros((1, d))
# position = jnp.array([t]).reshape(-1, 1)
# div_term = jnp.power(self.n, jnp.arange(0, d, 2) / d)
# pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
# pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
# pe = pe[jnp.newaxis, jnp.newaxis, :]
# pe = jnp.repeat(pe, x.shape[1], axis=1)
# pe = jnp.repeat(pe, x.shape[2], axis=2)
# x = x + pe
# # TODO add text/image encoding to x
# text_embeds = s
# text_embeds = jnp.mean(text_embeds, axis=1) # mean pooling across sequence
# text_embeds = jnp.repeat(text_embeds, x.shape[1], axis=0) # repeat across height
# text_embeds = jnp.repeat(text_embeds, x.shape[2], axis=1) # repeat across width
# text_embeds = text_embeds[jnp.newaxis, :] # 
# x = x + text_embeds

n = 10000
x = np.ones((4, 100, 100, 50))

t = np.array([1, 1, 1, 1])
print(t.shape)
d = x.shape[-1]
pe = np.zeros((x.shape[0], d))
position = np.array([t]).reshape(-1, 1)
print(position.shape)
print(position)
div_term = np.power(n, np.arange(0, d, 2) / d)
pe[:, 0::2] = np.sin(position * div_term)
pe[:, 1::2] = np.cos(position * div_term)
print(pe)
print(pe.shape)
pe = pe[:, np.newaxis, np.newaxis, :]
print(pe.shape)
pe = np.repeat(pe, x.shape[1], axis=1)
pe = np.repeat(pe, x.shape[2], axis=2)
print(pe.shape)
x = x + pe
print(x.shape)
# TODO add text/image encoding to x
# text_embeds = s
# text_embeds = np.mean(text_embeds, axis=1) # mean pooling across sequence
# text_embeds = np.repeat(text_embeds, x.shape[1], axis=0) # repeat across height
# text_embeds = np.repeat(text_embeds, x.shape[2], axis=1) # repeat across width
# text_embeds = text_embeds[np.newaxis, :] # 
# x = x + text_embeds