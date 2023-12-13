import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


x_train, x_test = x_train / 255.0 , x_test / 255.0


input_layer = tf.keras.layers.Input(name="input_layer", shape=(28, 28))
flatten_layer = tf.keras.layers.Flatten()(input_layer)
#hidden_layer = tf.keras.layers.Dense(units=128)(flatten_layer)
#hidden_layer = tf.keras.layers.Activation("relu")(hidden_layer)
hidden_layer = tf.keras.layers.Dense(units=128, activation='relu')(flatten_layer)
hidden_layer = tf.keras.layers.Dropout(0.2)(hidden_layer)
output_layer = tf.keras.layers.Dense(units=10, name="output_layer")(hidden_layer)
model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])
model.summary(150)

model.compile(
    optimizer = "Adam",
    loss = {
        "output_layer": tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )},
    metrics = ["acc"],
)

test_logits = model(x_test[:20, ...])
test_probabilities = tf.keras.activations.softmax(test_logits).numpy()
test_predictions = np.argmax(test_probabilities, axis=-1)

model.fit(
    x_train,
    y_train,
    batch_size = 32,
    epochs = 5
)

test_logits = model(x_test[:20, ...])
test_probabilities = tf.keras.activations.softmax(test_logits).numpy()
test_predictions = np.argmax(test_probabilities, axis=-1)
test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=32)
print(f"[test][pred] : {test_predictions}")
print(f"[test][real] : {y_test[:20]}")
print(f"[test][loss] : {test_loss}")
print(f"[test][accu] : {test_accuracy}")