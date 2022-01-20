import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    print("Modules Imported")

    trainData = 'train\\'
    valData = 'validate\\'
    trainSamples = 100
    valSamples = 100
    numClasses = 2
    imgWidth, imgHeight = 224, 224
    batchSize = 10

    trainDataGen = ImageDataGenerator(preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2)

    valDataGen = ImageDataGenerator(preprocessing_function=preprocess_input)

    trainGen = trainDataGen.flow_from_directory(trainData,
        target_size=(imgWidth, imgHeight),
        batch_size=batchSize,
        shuffle=True,
        seed=12345,
        class_mode='categorical')

    valGen = valDataGen.flow_from_directory(valData,
        target_size=(imgWidth, imgHeight),
        batch_size=batchSize,
        shuffle=True,
        seed=12345,
        class_mode='categorical')
    groundTruth = valGen.classes

    print("Data Initialised")

    def createModel():
        base_model = MobileNet(include_top=False, input_shape=(imgWidth, imgHeight, 3))
        for layer in base_model.layers[:]:
            layer.trainable = False
        inputt = Input(shape=(imgWidth, imgHeight, 3))
        custom_model = base_model(inputt)
        custom_model = GlobalAveragePooling2D()(custom_model)
        custom_model = Dense(64, activation='relu')(custom_model)
        custom_model = Dropout(0.5)(custom_model)
        predictions = Dense(numClasses, activation='softmax')(custom_model)
        return Model(inputs=inputt, outputs=predictions)

    model = None
    #model = createModel()
    model = load_model('model.h5')

    print("Model Created")

    model.compile(loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['acc'])
    numSteps = math.ceil((float(trainSamples)/batchSize))
    model.fit(trainGen,
        steps_per_epoch=numSteps,
        epochs=10,
        validation_data = valGen,
        validation_steps = numSteps)

    #print(valGen.class_indices)

    #model.save('model.h5')

    predictions = model.predict(valGen)
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

    fnames = valGen.filenames

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

    indices = get_images_with_sorted_probabilities(prediction_table,
        get_highest_probability=False,
        label=1,
        number_of_items=10,
        only_false_predictions=False)
    message = 'Images with the lowest probability of dogs'

    display(indices[:10], message)

    indices = get_images_with_sorted_probabilities(prediction_table,
        get_highest_probability=True,
        label=1,
        number_of_items=10,
        only_false_predictions=True)
    message = 'Images of cats with the highest probability of containing dogs'

    display(indices[:10], message)

    indices = get_images_with_sorted_probabilities(prediction_table,
        get_highest_probability=True,
        label=0,
        number_of_items=10,
        only_false_predictions=False)
    message = 'Images with the highest probability of cats'

    display(indices[:10], message)

    indices = get_images_with_sorted_probabilities(prediction_table,
        get_highest_probability=False,
        label=0,
        number_of_items=10,
        only_false_predictions=False)
    message = 'Images with the lowest probability of cats'

    display(indices[:10], message)

    indices = get_images_with_sorted_probabilities(prediction_table,
        get_highest_probability=True,
        label=0,
        number_of_items=10,
        only_false_predictions=True)
    message = 'Images of dogs with the highest probability of containing cats'

    display(indices[:10], message)

if __name__ == '__main__':
    main()
