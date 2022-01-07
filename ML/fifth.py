import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
tf.keras.backend.set_floatx('float32')

print("Modules Imported")

train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index))

train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean)/train_df_std
print(train_df_norm.head())

test_df_mean = test_df.mean()
test_df_std  = test_df.std()
test_df_norm = (test_df - test_df_mean)/test_df_std

threshold = 265000
train_df_norm["median_house_value_is_high"] = (train_df["median_house_value"] > threshold).astype(float)
test_df_norm["median_house_value_is_high"] = (test_df["median_house_value"] > threshold).astype(float)
print(train_df_norm["median_house_value_is_high"].head(8000))

#below is an alternative based on the z score values
# threshold_in_Z = 1.0
# train_df_norm["median_house_value_is_high"] = (train_df_norm["median_house_value"] > threshold_in_Z).astype(float)
# test_df_norm["median_house_value_is_high"] = (test_df_norm["median_house_value"] > threshold_in_Z).astype(float)

feature_columns = []
median_income = tf.feature_column.numeric_column("median_income")
tr = tf.feature_column.numeric_column("total_rooms")
feature_columns.append(median_income)
feature_columns.append(tr)
feature_layer = layers.DenseFeatures(feature_columns)
print(feature_layer(dict(train_df_norm)))

def create_model(my_learning_rate, feature_layer, my_metrics):
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,), activation=tf.sigmoid),)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),loss=tf.keras.losses.BinaryCrossentropy(),metrics=my_metrics)
    return model

def train_model(model, dataset, epochs, label_name, batch_size=None, shuffle=True):
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs, shuffle=shuffle)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs, hist

print("Defined the create_model and train_model functions")

def plot_curve(epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel("Value")
    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:],x[1:],label=m)
    plt.legend()
    plt.show()

print("Defined the plot_curve function")
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

learning_rate = 0.001
epochs = 20
batch_size = 100
label_name = "median_house_value_is_high"
classification_threshold = 0.35
# A `classification_threshold` of 0.52 appears to produce the highest accuracy (about 83%).
# Raising the `classification_threshold` to 0.9 drops accuracy by about 5%.
# Lowering the classification_threshold` to 0.3 drops accuracy by about 3%.
METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=classification_threshold), tf.keras.metrics.Precision(thresholds=classification_threshold, name='precision'), tf.keras.metrics.Recall(thresholds=classification_threshold, name='recall'),tf.keras.metrics.AUC(num_thresholds=100, name='auc')]

my_model = None
my_model = create_model(learning_rate, feature_layer, METRICS)
epochs, hist = train_model(my_model, train_df_norm, epochs, label_name, batch_size)
list_of_metrics_to_plot = ['accuracy', 'precision', 'recall', 'auc']
plot_curve(epochs, hist, list_of_metrics_to_plot)

features = {name:np.array(value) for name, value in test_df_norm.items()}
label = np.array(features.pop(label_name))

my_model.evaluate(x = features, y = label, batch_size=batch_size)
