## freeCodeCampl.org Machine Learning with Python Certification

#### Intro
These codes were orginally given in a Google Colab file. I added code to complete the assigned task. These are not very accurate models nor they have high benchmarks, but they do have the expected accuray / results.
A copy of these Colab files were saved and tagged here with 'Original'. This is the code I wrote and as soon as it passed th expected results I submitted the file.
To provide a more detailed explanation for what the code is doing and what choices were made, I made a separate .py file where the code is commented with much detail about each and and general direction of thought. These are marked as 'formatted'.

#### Models
1. Book Recommender - Uses a sparse matrix and NearestNeighbour from scikit-learn to recommend books. User can enter a book that exists in the data and the program will output 5 similar books. No error handling is built in so if the original book does not exist in the database it will throw an error.
2. Cat v Dog Image Classifier - a binary response CNN model to classify an image as that of a dog or a cat. 
3. Medical Bill Predictor - Uses linear regression to plot sample dataset and construct a line of best fit. The data was normalized using TensorFlow and Keras libraries.
4. Spam Detector - Implements NLP by cleaning the texts, tokenization and then feeds it into an embedded LSTM layer with a Dense NN on top activated with a Sigmoid function to return 0 for 'ham' and 1 for 'spam'

#### Running the Codes
The .py codes were onyl formatted in my local environment but not tested. Hence I would recommend running .ipynb files on Google Colab. Some commands are specific to the Colaboratory and so it makes most sense to use it. An alterantive would to run the .py files locally and correct erros until there are none. These should be limited to importing and dataset errors. The integral part of the code remains the same.
