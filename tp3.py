import numpy as np
import tensorflow as tf


cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()

# Normaliser les valeurs des pixels entre 0 et 1
x_train, x_test = x_train / 255.0 , x_test / 255.0

# Création de la couche d'entrée
input_layer = tf.keras.layers.Input(name="input_layer", shape=(None, None,3))

# Première couche de convolution
x = tf.keras.layers.Conv2D(kernel_size=(1,1), filters=128)(input_layer)

input_to_block = x

# Premier bloc de convolution
x= tf.keras.layers.Conv2D(kernel_size=(1,1), filters=32 ,padding="same")(x)
for _ in range ((7-1)//2):
    x = tf.keras.layers.Conv2D(kernel_size=(1,3), filters=32 ,padding="same")(x)
    x = tf.keras.layers.Conv2D(kernel_size=(3,1), filters=32 ,padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
x= tf.keras.layers.Conv2D(kernel_size=(1,1), filters=128 ,padding="same")(x)
x += input_to_block

input_to_block = x
x = tf.keras.layers.Conv2D(kernel_size=(1,1), filters=32 ,padding="same")(x)

for _ in range ((7-1)//2):
    x = tf.keras.layers.Conv2D(kernel_size=(1,3), filters=32 ,padding="same")(x)
    x = tf.keras.layers.Conv2D(kernel_size=(3,1), filters=32 ,padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
x= tf.keras.layers.Conv2D(kernel_size=(1,1), filters=128 ,padding="same")(x)
x += input_to_block

input_to_block = x
x = tf.keras.layers.Conv2D(kernel_size=(1,1), filters=32 ,padding="same")(x)

for _ in range ((7-1)//2):
    x = tf.keras.layers.Conv2D(kernel_size=(1,3), filters=32 ,padding="same")(x)
    x = tf.keras.layers.Conv2D(kernel_size=(3,1), filters=32 ,padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
x= tf.keras.layers.Conv2D(kernel_size=(1,1), filters=128 ,padding="same")(x)
x += input_to_block


hidden_layer = tf.keras.layers.Conv2D(kernel_size=(1,1), filters=10, padding="same")(x)
hidden_layer = tf.keras.layers.GlobalAveragePooling2D()(hidden_layer)
output_layer = tf.keras.layers.Activation("softmax", name="output_layer")(hidden_layer)

model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

model.summary(100)
model.compile(
    optimizer = "Adam",
    loss = {
        "output_layer": tf.keras.losses.SparseCategoricalCrossentropy()
    },
    metrics = ["acc"],
)

model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size = 32,
    epochs = 5
)