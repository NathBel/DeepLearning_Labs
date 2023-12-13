import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0 , x_test / 255.0

input_layer = tf.keras.layers.Input(name="input_layer", shape=(None, None, 1))
conv1 = input_layer

# Save the input before the first block
input_to_block = conv1

# First convolutional block
conv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1))(conv1)
conv1 = tf.keras.layers.BatchNormalization()(conv1)
conv1 = tf.keras.layers.Activation("relu")(conv1)
conv1 = tf.keras.layers.Dropout(0.2)(conv1)

# Save input before the second block
input_to_block_2 = conv1

# Second convolutional block
conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same')(conv1)
conv2 = tf.keras.layers.BatchNormalization()(conv2)
conv2 = tf.keras.layers.Activation("relu")(conv2)
conv2 = tf.keras.layers.Dropout(0.2)(conv2)

# Add the residual from the first block
conv2 = conv2 + input_to_block

# Save input before the third block
input_to_block_3 = conv2

# Third convolutional block
conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same')(conv2)
conv3 = tf.keras.layers.BatchNormalization()(conv3)
conv3 = tf.keras.layers.Activation("relu")(conv3)
conv3 = tf.keras.layers.Dropout(0.2)(conv3)

# Add the residual from the second block
conv3 = conv3 + input_to_block_2

# Fourth convolutional block
conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same', activation="relu")(conv3)
conv4 = tf.keras.layers.BatchNormalization()(conv4)
conv4 = tf.keras.layers.Activation("relu")(conv4)
conv4 = tf.keras.layers.Dropout(0.2)(conv4)

# Add the residual from the third block
conv4 = conv4 + input_to_block_3

# Additional convolutional and pooling layers
conv5 = tf.keras.layers.Conv2D(filters=10, kernel_size=(1, 1))(conv4)
global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()(conv5)

output_layer = tf.keras.layers.Activation("softmax", name="output_layer")(global_avg_pool)

model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

model.summary(150)
model.compile(optimizer="Adam", 
              loss={"output_layer": tf.keras.losses.SparseCategoricalCrossentropy()},
              metrics=["acc"])

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=32,
          epochs=5)
