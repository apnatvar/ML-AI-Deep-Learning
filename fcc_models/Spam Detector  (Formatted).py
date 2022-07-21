# This python code is just the formatted code from the .ipynb file.
# This was formatted to provide a more detailed explanation for the code written
# Some of the variables might also renamed to provide a germane name
# Please run this on Google Colab, I did not test it on my local environment


################################### Code given by freeCodeCamp ###################################
try:
  # %tensorflow_version only exists in Colab.
  !pip install tf-nightly
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer 
print(tf.__version__)
# get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv
train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"
################################### Code given by freeCodeCamp ###################################

################################### Code written by Apnatva Singh Rawat ###################################
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

df_train = pd.read_csv(train_file_path, sep='\t', header=0,
    names=['h_or_s', 'text'])
df_test = pd.read_csv(test_file_path, sep='\t', header=0,
    names=['h_or_s', 'text'])

maxLen = 50
vocabSize = 40000
df_train.head()
df_test.head()
df_train.isnull().sum()
df_train['h_or_s'] = [0 if i=='ham' else 1 for i in df_train['h_or_s']]
# defining 0 and 1 for the sigmoid function
print(df_train.groupby('h_or_s').count())
print(df_test.groupby('h_or_s').count())
# it is obvious that the dataset has way more 'ham' than 'spam'

# the cleaning process should be ideally followed for both train and test dataset
# but since fcc had their own function to test against some cases I used that function instead of testing it on df_test
# cleaning the punctuation marks from the texts
cleanText = []
for sentence in df_train['text']:
  k = ''
  for i in sentence:
    if i not in string.punctuation:
      k += i
  cleanText.append(k)
df_train['text'] = cleanText

# print(df_train.groupby(['h_or_s','text']).size().reset_index(name='Count'))
print(len(df_train))
print(len(df_train[['h_or_s','text']].drop_duplicates()))
# checking if any duplicates exist by comparing the len of both.
# there are clearly better ways to do this, but I could just run this section
# on google colab and compare it quickly so it made sense
# would not advise using this in a production model

# dropping the duplicates
df_train = df_train[['h_or_s','text']].drop_duplicates()
# comparing the counts of 'ham's and 'spam's now
print(df_train.groupby('h_or_s').count())
# 'ham's are still disproportionately large

# splitting strings by words
df_train['text'] = [i.split() for i in df_train['text']]
df_train.head()

# from this point forwrd there can be two trains of thought
# 1 - remove all 'non-words' and stopwords, use this cleaned text to train
# 2 - remove all 'non-words' but keep stopwords since most 'ham's can be pretty short comprising almost entirely of stopwords
# I followed the second
cleanText = []
for sentences in df_train['text']:
  k = ''
  for words in sentences:
    if words.isalpha():
      # if words.lower() in stopwords.words('english'): # uncomment if you would like to remove the stopwords too
      k += words + " "
  cleanText.append(k)
df_train['ctext'] = cleanText
# saving in an extra column just in case i find a use for them later
df_train.head()

df_train['length1'] = [len(i) for i in df_train['text']]
df_train.describe()
# finding number of worrds in each text
# you can have a minimum / maximum number here and the clean the dataset based on number of words
# this would be specially useful if you also deleted stopwords

# getting a random sammple of 550 'ham's to make it somewhat proportion to number of 'spams'
temp1 = df_train.query("h_or_s == 0").sample(n=550)
temp2 = df_train.query("h_or_s == 1").sample(n=498)
# getting all 'spam's. There are better ways to do this but this would also get them shuffled.
temp1 = temp1.append(temp2) # saving them in a temp dataframe
df_train_f = temp1.sample(frac = 1) # shuffling the dataset renaming it
df_train_f.head()

print(df_train_f.groupby('h_or_s').count()) # now they seem similar

`# this part of the code was something I learned on youtube or elsewhere

tokenizer = Tokenizer(num_words=vocabSize)#, oov_token='<UNK>') # establishing the vocablary
tokenizer.fit_on_texts(df_train_f['ctext']) # splitting text strings as a list of words
word_index = tokenizer.word_index
trainSequence = tokenizer.texts_to_sequences(df_train_f['ctext']) # converting texts to numeric equivalents
print(trainSequence)
trainPadded = pad_sequences(trainSequence, maxlen=maxLen, padding='post', truncating='post') # adding a padding to make sure all inputs are the same size
print(trainPadded)
# testSequence = tokenizer.texts_to_sequences(df_test['text'])
# testPadded = pad_sequences(testSequence, maxlen=maxLen, padding='post', truncating='post')

model = '' # this model description was part of the freeCodeCamp tutorial
# long short term memory or LSTM work well for NLP
# the final layer is sigmoid since we want our output to be bet (0, 1) ('ham', 'spam')
model = tf.keras.Sequential([
                             tf.keras.layers.Embedding(vocabSize, 64, input_length=maxLen),
                             tf.keras.layers.LSTM(64),
                             tf.keras.layers.Dense(1, activation="sigmoid")
])
model.summary()
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
# loss as binary_crossentropy since our classification is 0 or 1 i.e. binary
df_train_f.head()

history = ''
history = model.fit(np.asarray(trainPadded), df_train_f['h_or_s'], epochs=5, batch_size=32, verbose=True)
# training the model
# played around with epochs and batch sized to get a good accuracy rate
# since the 'ham' samples are chosen at random from the dataset, the accurate measurements can vary wildly

p_t = tokenizer.texts_to_sequences(pd.Series(["hello how are you friend"]))
p_t_padded = pad_sequences(p_t, maxlen=maxLen, padding='post', truncating='post')
sheesh = model.predict(p_t_padded)
if round(sheesh[0][0])==1:
  predic = 'spam'
else:
  predic = 'ham'
print(predic,sheesh)
# testing the model on a random string

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
# function as described by freeCodeCamp
def predict_message(pred_text):
  p_t = tokenizer.texts_to_sequences(pd.Series([pred_text]))
  p_t_padded = pad_sequences(p_t, maxlen=maxLen, padding='post', truncating='post')
  sheesh = model.predict(p_t_padded)
  if round(sheesh[0][0])==1:
    predic = 'spam'
  else:
    predic = 'ham'
  prediction = [sheesh[0][0], predic]
  print(prediction)
  return (prediction)
pred_text = "how are you doing today?"
prediction = predict_message(pred_text)[0]
print(prediction)
################################### Code written by Apnatva Singh Rawat ###################################

################################### Code given by freeCodeCamp for testing ###################################
# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
################################### Code given by freeCodeCamp for testing ###################################
