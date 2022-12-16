import jax
import jax.numpy as jnp

@jax.jit
def f(x):
  jax.debug.print("🤯 {x} 🤯", x=x)
  y = jnp.sin(x)
  jax.debug.breakpoint()
  jax.debug.print("🤯 {y} 🤯", y=y)
  return y
  
f(2.)
