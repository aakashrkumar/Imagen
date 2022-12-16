from jax.experimental import checkify
import jax
import jax.numpy as jnp

def f(x, i):
  checkify.check(i >= 0, "index needs to be non-negative!")
  y = x[i]
  z = jnp.sin(y)
  return z

jittable_f = checkify.checkify(f)

err, z = jax.jit(jittable_f)(jnp.ones((5,)), -1)
print(err.get())
