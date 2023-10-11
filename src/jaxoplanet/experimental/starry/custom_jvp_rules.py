import jax
import jax.numpy as jnp


@jax.custom_jvp
def zero_safe_sqrt(x):
    return jnp.sqrt(x)


@zero_safe_sqrt.defjvp
def zero_safe_sqrt_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = jnp.sqrt(x)
    cond = jnp.less(x, 10 * jnp.finfo(jax.dtypes.result_type(x)).eps)
    denom = jnp.where(cond, jnp.ones_like(x), x)
    tangent_out = 0.5 * x_dot * primal_out / denom
    return primal_out, tangent_out


@jax.custom_jvp
def zero_safe_arctan2(x, y):
    return jnp.arctan2(x, y)


@zero_safe_arctan2.defjvp
def zero_safe_arctan2_jvp(primals, tangents):
    (x, y) = primals
    (x_dot, y_dot) = tangents
    primal_out = zero_safe_arctan2(x, y)
    tol = 10 * jnp.finfo(jax.dtypes.result_type(x)).eps
    cond_x = jnp.logical_and(x > -tol, x < tol)
    cond_y = jnp.logical_and(y > -tol, y < tol)
    cond = jnp.logical_and(cond_x, cond_y)
    denom = jnp.where(cond, jnp.ones_like(x), x**2 + y**2)
    tangent_out = (y * x_dot - x * y_dot) / denom
    return primal_out, tangent_out


@jax.custom_jvp
def zero_safe_power(x, y):
    return x**y


@zero_safe_power.defjvp
def zero_safe_power_jvp(primals, tangents):
    (x, y) = primals
    (x_dot, _) = tangents
    primal_out = zero_safe_power(x, y)
    tol = 10 * jnp.finfo(jax.dtypes.result_type(x)).eps
    cond_x = jnp.logical_and(x > -tol, x < tol)
    cond_y = jnp.less(y, 1.0)
    cond = jnp.logical_and(cond_x, cond_y)
    denom = jnp.where(cond, jnp.ones_like(x), x)
    tangent_out = y * primal_out * x_dot / denom
    return primal_out, tangent_out
