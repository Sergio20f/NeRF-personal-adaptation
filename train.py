import os
import tensorflow as tf


tf.random.set_seed(21)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from config_loader import config
from utils import read_json, GetImages, img_c2w, get_focal_from_fow
from rendering import Rays, volume_render
from nerf_train import NeRF_Trainer
from train_monitor import get_monitor
from nerf_mlp import NeRF
from enhancers import pos_encoding, hier_sampling


# Get train and validation data
json_train_data = read_json(config.TRAIN_JSON)
json_val_data = read_json(config.VAL_JSON)
json_test_data = read_json(config.TEST_JSON)

f_length = get_focal_from_fow(field_of_view=json_train_data["camera_angle_x"],
                              width=config.IMG_WIDTH)

# Train, val, and test images with their c2w matrices
train_img_path, train_c2w = img_c2w(json_data=json_train_data, path=config.DATASET)
val_img_path, val_c2w = img_c2w(json_data=json_val_data, path=config.DATASET)
test_img_path, test_c2w = img_c2w(json_data=json_test_data, path=config.DATASET)

# Get images
images = GetImages(width=config.IMG_WIDTH, height=config.IMG_HEIGHT)

train_data = tf.data.Dataset.from_tensor_slices(train_img_path).map(images, num_parallel_calls=tf.data.AUTOTUNE)
val_data = tf.data.Dataset.from_tensor_slices(val_img_path).map(images, num_parallel_calls=tf.data.AUTOTUNE)
test_data = tf.data.Dataset.from_tensor_slices(test_img_path).map(images, num_parallel_calls=tf.data.AUTOTUNE)

# Instantiate the rays object
rays = Rays(f_length=f_length, width=config.IMG_WIDTH, height=config.IMG_HEIGHT, near_bound=config.NEAR,
           far_bound=config.FAR, nC=config.N_C)

# Define train, val, and test ray datasets
train_ray_data = tf.data.Dataset.from_tensor_slices(train_c2w).map(rays, num_parallel_calls=tf.data.AUTOTUNE)
val_ray_data = tf.data.Dataset.from_tensor_slices(val_c2w).map(rays, num_parallel_calls=tf.data.AUTOTUNE)
test_ray_data = tf.data.Dataset.from_tensor_slices(test_c2w).map(rays, num_parallel_calls=tf.data.AUTOTUNE)

# Zip the images and rays dataset together
train_data = tf.data.Dataset.zip((train_ray_data, train_data))
val_data = tf.data.Dataset.zip((val_ray_data, val_data))
test_data = tf.data.Dataset.zip((test_ray_data, test_data))

# Input pipeline
train_data = train_data.shuffle(config.BATCH_SIZE).batch(config.BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
val_data = val_data.shuffle(config.BATCH_SIZE).batch(config.BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Instantiate the coarse and fine models
coarse = NeRF(l_coor=config.L_COOR, l_dir=config.L_DIR, batch_size=config.BATCH_SIZE, dense_units=config.DENSE_UNITS,
              skip=config.SKIP_LAYER)

fine = NeRF(l_coor=config.L_COOR, l_dir=config.L_DIR, batch_size=config.BATCH_SIZE, dense_units=config.DENSE_UNITS,
              skip=config.SKIP_LAYER)

# Now, instantiate the model trainer
nerf_trainer = NeRF_Trainer(coarse=coarse, fine=fine, l_coor=config.L_COOR, l_dir=config.L_DIR, encoding=pos_encoding,
                            image_render=volume_render, sample_pdf=hier_sampling, n_f=config.N_F)

# Compile the model with Adam and MSE
nerf_trainer.compile(optimizer_c=Adam(), optimizer_f=Adam(), loss=MeanSquaredError())

if not os.path.exists(config.IMG_PATH):
    os.makedirs(config.IMG_PATH)

train_monitor_callback = get_monitor(test_data=test_data, encoding=pos_encoding, l_coor=config.L_COOR,
                                     l_dir=config.L_DIR, path=config.IMG_PATH)

# Train NeRF model
nerf_trainer.fit(train_data, steps_per_epoch=config.STEPS_PER_EPOCH, validation_data=val_data,
                 validation_steps=config.VALIDATION_STEPS, epochs=config.EPOCHS, callbacks=[train_monitor_callback],)

# Save models
nerf_trainer.coarse.save(config.COARSE_PATH)
nerf_trainer.fine.save(config.FINE_PATH)
