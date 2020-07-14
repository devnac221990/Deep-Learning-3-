import os
import numpy as np
from keras.layers import Activation, Conv1D, Dense, Embedding, Flatten, Input, MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.base import get_data_home
from keras.metrics import categorical_accuracy
categories = ['alt.atheism', 'soc.religion.christian']
from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(subset='train', shuffle=True, categories= categories, download_if_missing= False )
texts = dataset.data # Extract text
target = dataset.target # Extract target

print (target[:10])

print (len(texts))
print (len(target))
print (len(texts[0].split()))
print (texts[0])
print (target[0])
print (dataset.target_names[target[0]])

vocab_size = 20000

tokenizer = Tokenizer(num_words=vocab_size) # Setup tokenizer
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts) # Generate sequences


word_index = tokenizer.word_index
print('Found {:,} unique words.'.format(len(word_index)))

# Create inverse index mapping numbers to words
inv_index = {v: k for k, v in tokenizer.word_index.items()}

# Print out text again
for w in sequences[0]:
    x = inv_index.get(w)
    print(x,end = ' ')

max_length = 100
data = pad_sequences(sequences, maxlen=max_length)

from keras.utils import to_categorical
labels = to_categorical(np.asarray(target))




embedding_dim = 100 # We use 100 dimensional glove vectors

word_index = tokenizer.word_index
nb_words = min(vocab_size, len(word_index)) # How many words are there actually

embedding_matrix = np.zeros((nb_words, embedding_dim))


# Loop over all words in the word index
for word, i in word_index.items():
    # If we are above the amount of words we want to use we do nothing
    if i >= vocab_size:
        continue

print (embedding_matrix[100])
model = Sequential()
model.add(Embedding(vocab_size,
                    embedding_dim,
                    input_length=max_length,
                    weights = [embedding_matrix],
                    trainable = False))

model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[categorical_accuracy])

model.fit(data, labels, validation_split=0.2, epochs=20)