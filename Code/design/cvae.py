# %%
import os
from tensorflow.keras.layers import Lambda
import tensorflow as tf
from tensorflow.keras import layers, Model

# %%
# ==============================
# 0️⃣ Cargar Dataset
# ==============================

train_dir = os.path.join("..", "..", "Dataset", "preprocessed", "train")
test_dir = os.path.join("..", "..", "Dataset", "preprocessed", "test")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels=None,
    image_size=(64,64),
    batch_size=32,
    shuffle=True
)

#test_dataset = tf.keras.utils.image_dataset_from_directory(
#    test_dir,
#    labels=None,
#    image_size=(64,64),
#    batch_size=32,
#    shuffle=False
#)

train_dataset = train_dataset.map(lambda x: (tf.cast(x, tf.float32)/255.0,
                                             tf.cast(x, tf.float32)/255.0))
#test_dataset = test_dataset.map(lambda x: (tf.cast(x, tf.float32)/255.0,
#                                           tf.cast(x, tf.float32)/255.0))


# ==============================
# 1️⃣ Parámetros
# ==============================
img_shape = (64, 64, 3)
latent_dim = 128
kl_weight = 0.001

# ==============================
# 2️⃣ Encoder
# ==============================
inputs = layers.Input(shape=img_shape)

x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)   # 32x32
x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)       # 16x16
x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)      # 8x8
x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)      # 4x4
x = layers.Conv2D(512, 3, strides=2, padding="same", activation="relu")(x)      # 2x2

x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)

# Media y logvar para el espacio latente
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# ==============================
# 3️⃣ KL Layer
# ==============================
class KLLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=1
        )
        self.add_loss(kl_weight * tf.reduce_mean(kl))
        return inputs

z_mean, z_log_var = KLLayer()([z_mean, z_log_var])

# ==============================
# 4️⃣ Reparametrización (Sampling)
# ==============================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])

encoder = Model(inputs, z, name="encoder")
encoder.summary()

# ==============================
# 5️⃣ Decoder
# ==============================
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(2*2*512, activation="relu")(latent_inputs)
x = layers.Reshape((2, 2, 512))(x)

x = layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(x)  # 4x4
x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)  # 8x8
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)   # 16x16
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)   # 32x32
x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)   # 64x64

outputs = layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid")(x)

decoder = Model(latent_inputs, outputs, name="decoder")
decoder.summary()

# ==============================
# 6️⃣ VAE completo
# ==============================
outputs = decoder(encoder(inputs))
conv_vae = Model(inputs, outputs, name="conv_vae")
conv_vae.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
conv_vae.summary()

# ==============================
# 7️⃣ Entrenamiento
# ==============================
conv_vae.fit(
    train_dataset,
    epochs=50,
#    validation_data=test_dataset
)



