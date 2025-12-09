import jax
import jax.numpy as jnp
import time


# to check which device jax is running on
print(f"device: {jax.devices()[0]}")

# initialise the matrices
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
m_a = jax.random.normal(k1, (5000, 5000))
m_b = jax.random.normal(k2, (5000, 5000))

def mult(a, b):
    return jnp.dot(a, b)

jit_exec = jax.jit(mult)
# now we perform hpc analysis - time (mostly)

start = time.time()
_ = jit_exec(m_a, m_b).block_until_ready()
end = time.time()
print(f"Time taken for the first execution: {end - start: .6f}")

#--- end of the first compilation thing -------

# now we see the cached and fused version
start = time.time()
result = jit_exec(m_a, m_b).block_until_ready()
end = time.time()
print(f"Time taken for the cached version: {end - start : .6f}")

print(f"Result shape : {result.shape}")