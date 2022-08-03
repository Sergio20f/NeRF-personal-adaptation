import tensorflow as tf
import json
import numpy as np


def read_json(path):
    with open(path, "r") as fp:
        data = json.load(fp)

    return data


def img_c2w(json_data, path):
    images = []
    c2w = []

    for i in json_data["frames"]:
        img_path = i["file_path"]
        img_path = img_path.replace(".", path)
        images.append(f"{img_path}.png")

        c2w.append(i["transform_matrix"])

    return img_path, c2w


class GetImages:

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, img_path):
        image = tf.io.read_file(img_path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Resize and reshape
        image = tf.image.resize(image, (self.width, self.height))
        image = tf.reshape(image, (self.width, self.height, 3))

        return image


def get_focal_from_fow(field_of_view, width): return 0.5 * width / tf.tan(0.5 * field_of_view)


def get_translation_t(t):
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]
    matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)

    return matrix


def get_rotation_phi(phi):
    matrix = [
        [1, 0, 0, 0],
        [0, tf.cos(phi), -tf.sin(phi), 0],
        [0, tf.sin(phi), tf.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)

    return matrix


def get_rotation_theta(theta):
    matrix = [
        [tf.cos(theta), 0, -tf.sin(theta), 0],
        [0, 1, 0, 0],
        [tf.sin(theta), 0, tf.cos(theta), 0],
        [0, 0, 0, 1],
    ]


def pose_spherical(theta, phi, t):
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi/180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta/180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w

    return c2w
