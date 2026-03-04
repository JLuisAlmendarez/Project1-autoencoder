import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ==============================================================================
# Utility Classes
# ==============================================================================

def build_vgg_extractor():
    """
    Extrae features de dos niveles:
    - block1_conv2 → texturas finas, bordes
    - block3_conv3 → estructuras, formas
    """
    vgg = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(64, 64, 3)
    )
    vgg.trainable = False
    outputs = [
        vgg.get_layer("block1_conv2").output,
        vgg.get_layer("block3_conv3").output,
    ]
    return tf.keras.Model(vgg.input, outputs, name="vgg_extractor")

vgg_extractor = build_vgg_extractor()


def perceptual_loss(x_real, x_recon):
    x_real_vgg  = tf.keras.applications.vgg16.preprocess_input(x_real  * 255.0)
    x_recon_vgg = tf.keras.applications.vgg16.preprocess_input(x_recon * 255.0)
    feats_real  = vgg_extractor(x_real_vgg,  training=False)
    feats_recon = vgg_extractor(x_recon_vgg, training=False)
    loss = 0.0
    for f_real, f_recon in zip(feats_real, feats_recon):
        loss += tf.reduce_mean(tf.square(f_real - f_recon))
    return loss

class Sampling(layers.Layer):
    """Reparametrization trick: z = mean + exp(0.5 * log_var) * epsilon."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ==============================================================================
# DATASET
# ==============================================================================

train_dir  = os.path.join("..", "..", "Dataset", "preprocessed", "train")

batch_size = 16  # Reducido de 32 por VGG en VRAM

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels=None,
    image_size=(64, 64),
    shuffle=True,
    batch_size=batch_size,
)

train_dataset = train_dataset.map(lambda x: tf.cast(x, tf.float32) / 255.0)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

# ==============================================================================
# Arquitectura: Convolutional VAE
# ==============================================================================

img_shape  = (64, 64, 3)
latent_dim = 128

class ConvVAE(Model):
    """
    VAE convolucional con perdida ELBO + Perceptual Loss:
        loss = reconstruction_loss + beta * kl_loss + lambda * perceptual_loss

    beta=1  -> VAE estandar (Kingma & Welling 2013)
    beta>1  -> beta-VAE (mejor disentanglement, menor fidelidad)
    lam     -> peso de VGG perceptual loss
    """
    def __init__(self, latent_dim, beta=1.0, lam=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta       = beta
        self.lam        = lam

        self.encoder_conv = tf.keras.Sequential([
            layers.Conv2D(32,  3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(64,  3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(256, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(512, 3, strides=2, padding="same", activation="relu"),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
        ], name="encoder_conv")

        self.z_mean_layer    = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var_layer = layers.Dense(latent_dim, name="z_log_var")
        self.sampling        = Sampling()

        self.decoder = tf.keras.Sequential([
            layers.Dense(2 * 2 * 512, activation="relu"),
            layers.Reshape((2, 2, 512)),
            layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(32,  3, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(16,  3, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(3,   3, padding="same", activation="sigmoid"),
        ], name="decoder")

    def encode(self, x):
        h         = self.encoder_conv(x)
        z_mean    = self.z_mean_layer(h)
        z_log_var = self.z_log_var_layer(h)
        z         = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def decode(self, z):
        return self.decoder(z)

    def call(self, x, training=False):
        z_mean, z_log_var, z = self.encode(x)
        x_reconstructed      = self.decode(z)
        return x_reconstructed, z_mean, z_log_var

    def _compute_losses(self, x, x_reconstructed, z_mean, z_log_var):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(x - x_reconstructed), axis=[1, 2, 3])
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        p_loss = perceptual_loss(x, x_reconstructed)
        total  = reconstruction_loss + self.beta * kl_loss + self.lam * p_loss
        return total, reconstruction_loss, kl_loss, p_loss

    def train_step(self, x):
        with tf.GradientTape() as tape:
            x_rec, z_mean, z_log_var    = self(x, training=True)
            total, recon, kl, p         = self._compute_losses(x, x_rec, z_mean, z_log_var)
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {
            "loss":                total,
            "reconstruction_loss": recon,
            "kl_loss":             kl,
            "perceptual_loss":     p,
        }

    def test_step(self, x):
        x_rec, z_mean, z_log_var = self(x, training=False)
        total, recon, kl, p      = self._compute_losses(x, x_rec, z_mean, z_log_var)
        return {
            "loss":                total,
            "reconstruction_loss": recon,
            "kl_loss":             kl,
            "perceptual_loss":     p,
        }

# ==============================================================================
# INSTANCIA, COMPILACIÓN
# ==============================================================================

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=2000 * 6,
    alpha=1e-6,
)

cvae = ConvVAE(latent_dim=latent_dim, beta=1, lam=0.0001)
cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))

# ==============================================================================
# ENTRENAMIENTO
# ==============================================================================
cvae.fit(
    train_dataset,
    epochs=3,
)
