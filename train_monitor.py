import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import array_to_img


def get_monitor(test_data, encoding, l_coor, l_dir, path):
    test_elements, test_images = next(iter(test_data))
    test_ray_or_c, test_ray_dir_c, test_t_c = test_elements

    # Build test coarse rays
    test_ray_c = test_ray_or_c[..., None, :] + (test_ray_dir_c[..., None, :] * test_t_c[..., None])

    test_ray_c = encoding(test_ray_c, l_coor)
    test_ray_dirshape_c = tf.shape(test_ray_c[..., :3])
    test_dir_c = tf.broadcast_to(test_ray_dir_c[..., None, :], shape=test_ray_dirshape_c)
    test_dir_c = encoding(test_dir_c, l_dir)

    class Monitor(Callback):
        def on_epoch_end(self, epoch, logs=None):
            test_rgb_c, test_sigma_c = self.model.coarse.predict([test_ray_c, test_dir_c])

            # Render image
            test_image_c, _, test_weights_c = self.model.image_render(rgb=test_rgb_c, sigma=test_sigma_c, t=test_t_c)
            # Middle values of t
            test_t_mid = 0.5 * (test_t_c[..., 1:] + test_t_c[..., :-1])

            # Hierarchical sampling
            test_t_f = self.model.sample_pdf(t_mid=test_t_mid, weights=test_weights_c, n_f=self.model.n_f)
            test_t_f = tf.sort(tf.concat([test_t_c, test_t_f], axis=-1), axis=-1)

            # Rays for fine model
            test_ray_f = test_ray_or_c[..., None, :] + (test_ray_dir_c[..., None, :] * test_t_f[..., None])
            test_ray_f = self.model.encoding(test_ray_f, l_coor)

            # Fine rays direction
            test_dir_f_shape = tf.shape(test_ray_f[..., :3])
            test_dir_f = tf.broadcast_to(test_ray_dir_c[..., None, :], shape=test_dir_f_shape)
            test_dir_f = self.model.encoding(test_dir_f, l_dir)

            # Final model prediction
            test_rgb_f, test_sigma_f = self.model.fine.predict([test_ray_f, test_dir_f])

            # Render
            test_render_f = self.model.image_render(rgb=test_rgb_f, sigma=test_sigma_f, t=test_t_f)
            test_image_f, test_depth_f, _ = test_render_f

            # Plot the coarse and fine images, the depth map, and the target image
            _, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))

           ax[0].imshow(array_to_img(test_image_c[0]))
           ax[0].set_title(f"Coarse Image")
           ax[1].imshow(array_to_img(test_image_f[0]))
           ax[1].set_title(f"Fine Image")
           ax[2].imshow(array_to_img(test_depth_f[0, ..., None]),
                        cmap="inferno")
           ax[2].set_title(f"Fine Depth Image")
           ax[3].imshow(array_to_img(test_images[0]))
           ax[3].set_title(f"Real Image")
           plt.savefig(f"{path}/{epoch:03d}.png")
           plt.close()

    train_monitor = Monitor()
    return train_monitor
