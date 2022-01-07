#@title Import relevant modules
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting.
#pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
# Import the dataset.
training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

# Scale the label.
training_df["median_house_value"] /= 1000.0

# Print the first rows of the pandas DataFrame.
#print(training_df)
#print(training_df.head())
#print(training_df.describe())

def build_model(my_learning_rate):
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate),loss="mean_squared_error",metrics=[tf.keras.metrics.RootMeanSquaredError()])
  return model

def train_model(model, feature, label, epochs, batch_size):
  history = model.fit(x=training_df[feature],y=training_df[label],batch_size=batch_size,epochs=epochs)
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]
  epochs = history.epoch
  hist = pd.DataFrame(history.history)
  rmse = hist["root_mean_squared_error"]
  return trained_weight, trained_bias, epochs, rmse

print("Defined create_model and train_model")

def plot_the_model(trained_weight, trained_bias, feature, label):
  plt.xlabel("feature")
  plt.ylabel("label")
  rnadom_examples = training_df.sample(n=200)
  plt.scatter(rnadom_examples[feature], rnadom_examples[label])
  x0 = 0
  y0 = trained_bias
  x1 = 10000
  y1 = trained_bias + (trained_weight * x1)
  plt.plot([x0, x1], [y0, y1], c='r')
  #plt.show()

def plot_the_loss_curve(epochs, rmse):
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")
  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.97, rmse.max()])
  #plt.show()

print("Defined the plot_the_model and plot_the_loss_curve functions.")

my_feature = "rooms_per_person"
my_label   = "median_house_value"

training_df[my_feature] = training_df["total_rooms"] / training_df["population"]
learning_rate = 0.06
epochs = 24
my_batch_size = 30

my_model = None

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, my_label, epochs, my_batch_size)

print("\nThe learned weight for this model is %.4f" %trained_weight)
print("The learned bias for your model is %.4f\n" %trained_bias)

plot_the_model(trained_weight, trained_bias, my_feature, my_label)
#plot_the_loss_curve(epochs, rmse)

def predict_house_values(n, feature, label):
  batch = training_df[feature][10000:10000 + n]
  predicted_values = my_model.predict_on_batch(x=batch)

  print("feature   label          predicted")
  print("  value   value          value")
  print("          in thousand$   in thousand$")
  print("--------------------------------------")
  for i in range(n):
    print ("%5.0f %6.0f %15.0f %f" % (training_df[feature][10000 + i],
                                   training_df[label][10000 + i],
                                   predicted_values[i][0],(-training_df[label][10000 + i]+predicted_values[i][0])*100/training_df[label][10000 + i]))

predict_house_values(10, my_feature, my_label)

print(training_df.corr())
