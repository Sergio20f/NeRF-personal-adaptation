import tensorflow as tf


def pos_encoding(pos, L):
    gamma = [pos]

    for i in range(L):
        gamma.append(tf.sin((2**i) * pos))
        gamma.append(tf.cos((2**i) * pos))

    # List to positional vector
    gamma = tf.concat(gamma, axis=-1)

    return gamma


def hier_sampling(t_mids, weights, n_f, u_shape: list):
    # Add a samll value to the weights to prevent it from nan
    weights += 1e-5

    # Normalise the weights to get a pdf
    pdf = weights/tf.reduce_sum(weights, axis=-1, keepdims=True)
    # Cumulative distribution function
    cdf = tf.cumsum(pdf, axis=-1)

    u_shape.append(n_f)
    u = tf.random.uniform(shape=u_shape)

    # Get the indices of the points of u when it is inserted into the cdf in a sorted manner
    indices =  tf.searchsorted(cdf, u, side="right")

    # Boundaries
    below = tf.maximum(0, indices-1)
    above = tf.minimum(cdf.shape[-1] - 1, indices)
    indices_stack = tf.stack([below, above], axis=-1)

    # Sample the indices from cdf
    cdf_idx = tf.gather(cdf, indices_stack, axis=-1, batch_dims=len(indices_stack.shape)-2)

    # Gather the t's according to the indices
    t_mids_idx = tf.gather(t_mids, indices_stack, axis=-1, batch_dims=len(indices_stack.shape)-2)

    # Inverse sampling
    denom = cdf_idx[..., 1] - cdf_idx[..., 0]
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u - cdf_idx[..., 0]) / denom
    samples = t_mids_idx[..., 0] + t * (t_mids_idx[..., 1] - t_mids_idx[..., 0])

    return samples
