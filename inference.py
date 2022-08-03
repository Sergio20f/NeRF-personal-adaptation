import tensorflow as tf
import os
import numpy as np
import imageio

from tqdm import tqdm
from tensorflow.keras.models import load_model

from config_loader import config
from utils import get_focal_from_fow, read_json, pose_spherical
from enhancers import hier_sampling, pos_encoding
from rendering import Rays, volume_render


# Store the novel views
novel_c2w = []
for theta in np.linspace(0, 360, config.SAMPLE_THETA_POINTS, endpoint=False):
    c2w = pose_spherical(theta=theta, phi=-30, t=4)
    novel_c2w.append(c2w)

train_data = read_json(config.TRAIN_JSON)
f_length = get_focal_from_fow(field_of_view=train_data["camera_angle_x"], width=config.IMG_WIDTH)

rays = Rays(f_length=f_length, width=config.IMG_WIDTH, height=config.IMG_HEIGHT, near_bound=config.NEAR,
            far_bound=config.FAR, nC=config.N_C)

# Create a dataset from the novel views
dataset = tf.data.Dataset.from_tensor_slices(novel_c2w).map(rays).batch(config.BATCH_SIZE)

coarse = load_model(config.COARSE_PATH, compile=False)
fine = load_model(config.FINE_PATH, compile=False)

# List to store all the novelviews from NeRF
frames = []
for i in tqdm(dataset):
    rays_or_c, rays_dir_c, t_c = i
    rays_c = rays_or_c[..., None, :] + (rays_dir_c[..., None, :] * t_c[..., None])

    # Positional encoding
    rays_c = pos_encoding(rays_c, config.L_COOR)
    dir_c_shape = tf.shape(rays_c[..., :3])
    dir_c = tf.broadcast_to(rays_dir_c[..., None, :], shape=dir_c_shape)
    dir_c = pos_encoding(dir_c, config.L_COOR)

    # Predictions from coarse model
    rgb_c, sigma_c = coarse.predict([rays_c, dir_c])
    # Render from predictions
    render_c = volume_render(rgb=rgb_c, sigma=sigma_c, t=t_c)
    _, _, weights_c = render_c

    # Middle values of t
    t_c_mid = 0.5 * (t_c[..., 1:] + t_c[..., :-1])

    # Hierarchichal sampling
    t_f = hier_sampling(t_mids=t_c_mid, weights=weights_c, n_f=config.N_F)
    t_f = tf.sort(tf.concat([t_c. t_f], axis=-1), axis=-1)

    # Build fine rays + pos_encoding
    rays_f = rays_or_c[..., None, :] + (rays_dir_c[..., None, :] * t_f[..., None])
    rays_f = pos_encoding(rays_f, config.L_COOR)

    # Build fine directions
    dir_f_shape = tf.shape(rays_f[..., :3])
    dir_f = tf.broadcast_to(rays_dir_c[..., None, :], shape=dir_f_shape)
    dir_f = pos_encoding(dir_f, config.L_DIR)

    # Compute predictions from the fine model
    rgb_f, sigma_f = fine.predict([rays_f, dir_f])

    # Render image from predictions
    render_f = volume_render(rgb=rgb_f, sigma=sigma_f, t=t_f)
    image_f, _, _ = render_f

    # Insert rendered fine image to the list
    frames.append(image_f.numpy()[0])

# Implement video-gif functionality
if not os.path.exists(config.VIDEO_PATH):
    os.makedirs(config.VIDEO_PATH)

imageio.mimwrite(config.VIDEO_PATH, frames, fps=config.FPS, quality=config.QUALITY,
                 macro_block_size=config.MACRO_BLOCK_SIZE)
