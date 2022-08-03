import tensorflow as tf
from tensorflow.keras.metrics import Mean


class NeRF_Trainer(tf.keras.Model):

    def __init__(self, coarse, fine, l_coor, l_dir, encoding, image_render, sample_pdf, n_f):
        super().__init__()

        self.coarse = coarse
        self.fine = fine
        self.l_coor = l_coor
        self.l_dir = l_dir
        self.encoding = encoding
        self.image_render = image_render
        self.sample_pdf = sample_pdf
        self.n_f = n_f

    def compile(self, optimizer_c, optimizer_f, loss):
        super().compile()
        self.optimizer_c = optimizer_c
        self.optimizer_f = optimizer_f
        self.loss = loss

        # Loss and psnr tracker
        self.loss_tracker = Mean(name="loss")
        self.psnr_tracker = Mean(name="psnr")

    def train_step(self, inputs):
        elements, images = inputs
        ray_or_c, ray_dir_c, t_c = elements

        # Generate the coarse rays
        ray_c = ray_or_c[..., None, :] + (ray_dir_c[..., None, :] * t_c[..., None])

        # Positional encoding
        ray_c = self.encoding(ray_c, self.l_coor)
        dir_c_shape = tf.shape(ray_c[..., 3])
        dir_c = tf.broadcast_to(ray_dir_c[..., None, :], shape=dir_c_shape)
        dir_c = self.encoding(dir_c, self.l_dir)

        # Forward propagation coarse network
        with tf.GradientTape() as c_tape:
            rgb_c, sigma_c = self.coarse([ray_c, dir_c])
            # Render images from predictions
            images_c, _, weights_c = self.image_render(rgb=rgb_c, sigma=sigma_c, t=t_c)

            # Compute loss
            loss_c = self.loss(images, images_c)

        t_mid = 0.5 * (t_c[..., 1:] + t_c[..., :-1])

        # Apply hierarchical sampling and get the input for the fine network
        t_f = self.sample_pdf(t_mid=t_mid, weights=weights_c, n_f=self.n_f)
        t_f = tf.sort(tf.concat([t_c, t_f], axis=-1), axis=-1)

        # Rays for the fine network
        ray_f = ray_c[..., None, :] + (ray_dir_c[..., None, :] * t_f[..., None])
        ray_f = self.encoding(ray_f, self.l_coor)

        # Build fine directions
        dir_f_shape = tf.shape(ray_f[..., :3])
        dir_f = tf.broadcast_to(ray_dir_c[..., None, :], shape=dir_f_shape)
        dir_f = self.encoding(dir_f, self.l_dir)

        # Forward propagation fine network
        with tf.GradientTape() as f_tape:
            rgb_f, sigma_f = self.fine([ray_f, dir_f])

            # Render image from prediction
            images_f, _, _ = self.image_render(rgb=rgb_f, sigma=sigma_f, t=t_f)
            loss_f = self.loss(images, images_f)

        # Back-propagation
        t_c_var = self.coarse.trainable_variables
        grads_c = c_tape.gradient(loss_c, t_c_var)
        self.optimizer_c.apply_gradients(zip(grads_c, t_c_var))

        t_f_var = self.fine.trainable_variables
        grads_f = f_tape.gradients(loss_c, t_c_var)
        self.optimizer_f.apply_gradients(zip(grads_f, t_f_var))
        psnr = tf.image.psnr(images, images_f, max_val=1)

        # Compute loss and psnr
        self.loss_tracker.update_state(loss_f)
        self.psnr_tracker.update_state(psnr)

        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_tracker.result()}

    def test_step(self, inputs):
        elements, images = inputs
        ray_or_c, ray_dir_c, t_c = elements

        # Generate the coarse rays
        ray_c = ray_or_c[..., None, :] + (ray_dir_c[..., None, :] * t_c[..., None])
        # Positional encode the rays and directions
        ray_c = self.encoding(ray_c, self.l_coor)
        dir_c_shape = tf.shape(ray_c[..., 3])
        dir_c = tf.broadcast_to(ray_dir_c[..., None, :], shape=dir_c_shape)
        dir_c = self.encoding(dir_c, self.l_dir)

        # Compute the predictions from the coarse network
        rgb_c, sigma_c = self.coarse([ray_c, dir_c])
        # Render the image from predictions
        images_c, _, weights_c = self.image_render(rgb=rgb_c, sigma=sigma_c, t=t_c)

        t_mid = 0.5 * (t_c[..., 1:] + t_c[..., :-1])

        # Apply hierarchical sampling and get the input for the fine network
        t_f = self.sample_pdf(t_mid=t_mid, weights=weights_c, n_f=self.n_f)
        t_f = tf.sort(tf.concat([t_c, t_f], axis=-1), axis=-1)

        # Rays for the fine network
        ray_f = ray_c[..., None, :] + (ray_dir_c[..., None, :] * t_f[..., None])
        ray_f = self.encoding(ray_f, self.l_coor)

        # Build fine directions
        dir_f_shape = tf.shape(ray_f[..., :3])
        dir_f = tf.broadcast_to(ray_dir_c[..., None, :], shape=dir_f_shape)
        dir_f = self.encoding(dir_f, self.l_dir)

        # Fine predictions
        rgb_f, sigma_f = self.fine([ray_f, dir_f])
        # Render image from prediction
        images_f, _, _ = self.image_render(rgb=rgb_f, sigma=sigma_f, t=t_f)

        loss_f = self.loss(images, images_f)
        psnr = tf.image.psnr(images, images_f, max_val=1)

        # Compute the loss and psnr
        self.loss_tracker.update_state(loss_f)
        self.psnr_tracker.update_state(psnr)

        return {"loss": self.loss_tracker.results(), "psnr": self.psnr_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.psnr_tracker]
