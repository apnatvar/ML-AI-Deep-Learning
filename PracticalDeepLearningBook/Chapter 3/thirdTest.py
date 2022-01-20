import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import second

model = load_model('model.h5')

#img_path = "dog.9482.jpg"
#img = image.load_img(img_path, target_size=(224,224))
#plt.imshow(img)
#plt.show()

def classify():
    predictions = model.predict(second.valGen)
    prediction_table = {}
    for index, val in enumerate(predictions):
        indexHighestProb = np.argmax(val)
        valHighestProb = val[indexHighestProb]
        prediction_table[index] = [valHighestProb, indexHighestProb, groundTruth[index]]
        #assert len(predictions) == len(groundTruth) == len(prediction_table)

    def get_images_with_sorted_probabilities(prediction_table,
                                             get_highest_probability,
                                             label,
                                             number_of_items,
                                             only_false_predictions=False):
        sorted_prediction_table = [(k, prediction_table[k])
                                   for k in sorted(prediction_table,
                                                   key=prediction_table.get,
                                                   reverse=get_highest_probability)]
        result = []
        for index, key in enumerate(sorted_prediction_table):
            image_index, [probability, predicted_index, gt] = key
            if predicted_index == label:
                if only_false_predictions == True:
                    if predicted_index != gt:
                        result.append(
                            [image_index, [probability, predicted_index, gt]])
                else:
                    result.append(
                        [image_index, [probability, predicted_index, gt]])
        return result[:number_of_items]

    fnames = second.valGen.filenames

    def plot_images(filenames, distances, message):
        images = []
        for filename in filenames:
            images.append(mpimg.imread(filename))
        plt.figure(figsize=(40, 30))
        columns = 5
        for i, image in enumerate(images):
            ax = plt.subplot(-(len(images) // - columns - 1), columns, i + 1)
            ax.set_title("\n\n\n\n\n\n" + filenames[i].split("\\")[-1] + " - " +
                         "Probability: " +
                         str(float("{0:.2f}".format(distances[i]))), fontsize=5)
            plt.suptitle(message, fontsize=15, fontweight='bold')
            plt.axis('off')
            plt.imshow(image)
        plt.show()

    def display(sorted_indices, message):
        similarImgPaths = []
        distances = []
        for name, val in sorted_indices:
            [prob, predictedIndex, gt] = val
            similarImgPaths.append(valData + fnames[name])
            distances.append(prob)
        plot_images(similarImgPaths, distances, message)

    indices = get_images_with_sorted_probabilities(prediction_table,
        get_highest_probability=True,
        label=1,
        number_of_items=10,
        only_false_predictions=False)
    message = 'Images with the highest probability of dogs'

    display(indices[:10], message)

classify()
