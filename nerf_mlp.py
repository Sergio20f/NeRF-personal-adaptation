import tensorflow as tf
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras import Model, Input


def NeRF(l_coor, l_dir, batch_size, dense_units, skip):

    # We will need two identical networks -> 'coarse' and 'fine'
    # 'Coarse'
    ray = Input(shape=(None, None, None, 2 * 3 * l_coor + 3), batch_size=batch_size)
    # 'Fine'
    direction = Input(shape=(None, None, None, 2 * 3 * l_dir + 3), batch_size=batch_size)

    x = ray
    for i in range(8): # 8 is the number of layers
        x = Dense(dense_units, activation="relu")(x)

        # check for residual connection
        if i % skip == 0 and i > 0:
            x = concatenate([x, ray], axis=-1)

    # Define the volume density
    sigma = Dense(1, activation="relu")(x)

    # Feature vector
    feature = Dense(dense_units)(x)

    # Union of the feature vector with the direction input
    feature = concatenate([feature, direction], axis=-1)
    x = Dense(dense_units//2, activation="relu")(feature) # passing the feature vector through a Dense layer

    # Layer for the rgb values
    rgb_layer = Dense(units=3, activation="sigmoid")(x)

    nerf = Model(inputs=[ray, direction], outputs=[rgb_layer, sigma])

    return nerf
