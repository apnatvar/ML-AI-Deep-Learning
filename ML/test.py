import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt

def buildModel(learningRate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learningRate),loss="mean_squared_error",metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def trainModel(model, feature, label, epochs, batchSize):
    history = model.fit(x=feature, y=label, batch_size=batchSize,epochs=epochs)
    trainedWeight = model.get_weights()[0]
    trainedBias = model.get_weights()[1]
    epochs = history.epochs
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]
    return trainedWeight, trainedBias, epochs, rmse

print("Defined createModel and trainModel")

def plot_the_model(trained_weight, trained_bias, feature, label):
  plt.xlabel("feature")
  plt.ylabel("label")

  # Plot the feature values vs. label values.
  plt.scatter(feature, label)

  # Create a red line representing the model. The red line starts
  # at coordinates (x0, y0) and ends at coordinates (x1, y1).
  x0 = 0
  y0 = trained_bias
  x1 = feature[-1]
  y1 = trained_bias + (trained_weight * x1)
  plt.plot([x0, x1], [y0, y1], c='r')

  # Render the scatter plot and the red line.
  plt.show()

def plot_the_loss_curve(epochs, rmse):

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.97, rmse.max()])
  plt.show()

print("Defined the plot_the_model and plot_the_loss_curve functions.")
