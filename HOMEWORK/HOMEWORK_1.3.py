import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Data (y = 12x + 2 with noise)
X = np.linspace(-18, 10, 167)
Y = 12 * X + 2 + np.random.randn(*X.shape) * 2

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mse')

# Train
model.fit(X, Y, epochs=200, verbose=0)

# Predict
Y_pred = model.predict(X)

# Plot with labels
plt.scatter(X, Y, color="blue", label="Data")
plt.plot(X, Y_pred, color="red", label="Fitted Line")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Linear Regression using TensorFlow")
plt.legend()
plt.show()