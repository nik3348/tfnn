import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow.keras.layers import Dense, Flatten, Softmax
from tensorflow.keras import Model

import numpy as np

class Mnist(Model):
  def __init__(self):
    super(Mnist, self).__init__()
    self.flatten = Flatten(input_shape=(28, 28))
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

  def train_step(self, data):
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    x, y = data

    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}


FROMFILE = True

def start():
  print('starting')
  if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
  else:
    print("TensorFlow **IS NOT** using the GPU")

  mnist = tf.keras.datasets.mnist

  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  train_images, test_images = train_images / 255.0, test_images / 255.0

  model = Mnist()
  if FROMFILE:
    try:
      model = tf.keras.models.load_model('models/mnist')
    except:
      print("model doesn't exist")

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  model.fit(train_images, train_labels, epochs=3)

  test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
  print('\nTest accuracy:', test_acc)

  probability_model = tf.keras.Sequential([model, Softmax()])
  predictions = probability_model.predict(test_images)
  print(f'{np.argmax(predictions[0])} : {test_labels[0]}')

  if FROMFILE:
    # Save the entire model as a SavedModel.
    model.save('models/mnist')
