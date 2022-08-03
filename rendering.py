import tensorflow as tf


class Rays:

    def __init__(self, f_length, width, height, near_bound, far_bound, nC):
        # Image properties
        self.f_length = f_length
        self.width = width
        self.height = height

        # Bounding values
        self.near_bound = near_bound
        self.far_bound = far_bound

        # Number of samples for the coarse model
        self.nC = nC

    def __call__(self, c2w):
        # Meshgrid with image dimnesions
        (x, y) = tf.meshgrid(
            tf.range(self.width, dtype=tf.float32),
            tf.range(self.height, dtype=tf.float32),
            indexing="xy")

        # Create the camera coordinate frame
        x_cam = (x - self.width * 0.5)/self.f_length
        y_cam = (y - self.height * 0.5)/self.f_length

        # Camera vector
        cam_vector = tf.stack([x_cam, -y_cam, -tf.ones_like(x)], axis=1)

        # Split the camera vector into rotation and translation matrices
        r_matrix = c2w[:3, :3]
        t_matrix = c2w[:3, -1]

        # Change cam_vector dimensions for matrix multiplication
        cam_vector = cam_vector[..., None, :]
        world_vector = cam_vector * r_matrix

        # 1. Direction of the rays
        ray_dir = tf.reduce_sum(world_vector, axis=-1)
        # Norm vector
        ray_dir = ray_dir/tf.norm(ray_dir, axis=-1, keepdims=True)

        # 2. Use the translation matrix to compute the origin of the ray
        ray_or = tf.broadcast_to(t_matrix, tf.shape(ray_dir))

        # Sample along t values
        t = tf.linspace(self.near_bound, self.far_bound, self.nC)
        noise_shape = list(ray_or.shape[:-1]) + list(self.nC)
        # Normalised noise
        noise = tf.random.uniform(shape=noise_shape) * (self.far_bound - self.near_bound)/self.nC

        # Add noise
        t = t + noise

        return ray_or, ray_dir, t


def volume_render(rgb, sigma, t, t_shape: list):
    delta_t = t[, 1:] - t[..., :-1]
    delta_t = tf.concat([delta_t, tf.broadcast_to([1e10], shape=t_shape)], axis=-1)

    sigma = sigma[..., 0]
    a = 1 - tf.exp(-sigma * delta_t)

    e = 1 - a
    epsilon = 1e-10

    # Compute transmittance and weights
    transmittance = tf.math.cumprod(e + epsilon, axis=-1, exclusive=True)
    weights = a * transmittance

    # Build the image and depth map from the points of the rays
    image = tf.reduce_sum(weights[..., None] * rgb, axis=-2)
    depth = tf.reduce_sum(weights * t, axis=-1)

    return image, depth, weights
