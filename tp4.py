import numpy as np
import tensorflow as tf

#Class pour créer un custom layer
class SequentialySeparatedConv2D(tf.keras.layers.Layer):
    
    def __init__(self, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        
    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
        })
        return config
    
    def build(self, input_shape):
        super().build(input_shape)
        _, _, _, c = input_shape
        self.layers = []
        
        for _ in range((self.kernel_size - 1) // 2):
            self.layers.append(
            tf.keras.layers.Conv2D(filters=c, kernel_size=(3, 1), padding="same")
            )
            self.layers.append(
            tf.keras.layers.Conv2D(filters=c, kernel_size=(1, 3), padding="same")
            )
            self.layers.append(tf.keras.layers.BatchNormalization())
            self.layers.append(tf.keras.layers.Activation("relu"))
            self.layers.append(tf.keras.layers.Dropout(0.2))
    

class EDSequentialySeparatedConv2D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, compression_value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.compression_value = compression_value
        
    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "compression_value": self.compression_value
        })
        return config
    
    def build(self, input_shape):
        super().build(input_shape)
        bs, w, h, c = input_shape
        
        self.layers = []
        
        # compression layer
        self.layers.append(
            tf.keras.layers.Conv2D(filters=self.compression_value, kernel_size=(1, 1), padding="same")
        )
        
        self.layers.append(
            SequentialySeparatedConv2D(kernel_size=self.kernel_size)
        )
        
        # decompression layer
        self.layers.append(
            tf.keras.layers.Conv2D(filters=c, kernel_size=(1, 1), padding="same")
        )

cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
  
x_train, x_test = x_train / 255.0 , x_test / 255.0
input_layer = tf.keras.layers.Input(name="input_layer", shape=(None, None,3))

# modifier le conv2D
x = tf.keras.layers.Conv2D(kernel_size=(1,1), filters=128)(input_layer)

input_to_block = x
#Use the custom layer EDSequentialySeparatedConv2D
x = EDSequentialySeparatedConv2D(kernel_size=7, compression_value=32)(x)
x += input_to_block


input_to_block = x

#Use the custom layer 
x = EDSequentialySeparatedConv2D(kernel_size=7, compression_value=32)(x)

x += input_to_block


input_to_block = x

#Use the custom layer
x = EDSequentialySeparatedConv2D(kernel_size=7, compression_value=32)(x)

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