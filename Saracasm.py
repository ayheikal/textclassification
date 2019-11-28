import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

imdb, info = tfds.load("imdb_reviews",with_info=True, as_supervised=True)
train_data, test_data = imdb['train'],imdb['test']
print(len(train_data))
training_sentences=[]
training_labels=[]

testing_sentences=[]
testing_labels=[]


for s,l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())
print("training: ",training_sentences)
for s,l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

vocab_size=10000
embedding_dim=16
max_len=20
trunc_type='post'
oov_token='<OOV>'
"""traing the data"""
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(training_sentences)
padded=pad_sequences(sequences, maxlen=max_len,truncating=trunc_type)

"""testing the data"""
testing_sequences=tokenizer.texts_to_sequences(testing_sentences)
testing_padding=pad_sequences(testing_sequences,maxlen=max_len)

""""""
reverse_word_index=dict([(value,key)for (key,value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?')for i in text])

print(decode_review(padded[1]))
print(training_sentences[1])



#
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()






