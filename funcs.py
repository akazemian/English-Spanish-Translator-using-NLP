
import string
import re
from numpy import array, argmax, random, take
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, RepeatVector
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import pickle

spa_length = 20
eng_length = 20

#%matplotlib inline
#pd.set_option('display.max_colwidth', 200)

loaded_model = load_model('model_spa_to_eng', compile = False)
loaded_model_2 = load_model('model_eng_to_spa', compile = False)
# loading
with open('spa_tokenizer.pickle', 'rb') as handle:
    spa_tokenizer = pickle.load(handle)

# loading
with open('eng_tokenizer.pickle', 'rb') as handle:
    eng_tokenizer = pickle.load(handle)


import re
# function to read raw text file
def read_text(filename):
        # open the file
        file = open(filename, mode='rt', encoding='utf-8')
        
        # read all text
        text = file.read()
        file.close()
        return text

# split a text into sentences
def to_lines(text):
      sents = text.strip().split('\n')
      sents = [i.split('\t') for i in sents]
      return sents

def clean(data):
# Remove punctuation
  eng_spa[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in eng_spa[:,0]]
  eng_spa[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in eng_spa[:,1]]

  for i in range(len(eng_spa)):
    eng_spa[i,0] = eng_spa[i,0].lower()
    eng_spa[i,0] = eng_spa[i,0].replace('/\s\s+/g', ' ');

    eng_spa[i,1] = eng_spa[i,1].lower()
    eng_spa[i,1] = eng_spa[i,1].replace('/\s\s+/g', ' ');
  return eng_spa

# function to build a tokenizer
def tokenization(lines):
      tokenizer = Tokenizer()
      tokenizer.fit_on_texts(lines)
      return tokenizer

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
         # integer encode sequences
         seq = tokenizer.texts_to_sequences(lines)
         # pad sequences with 0 values
         seq = pad_sequences(seq, maxlen=length, padding='post')
         return seq

# build NMT model
def define_model(in_vocab,out_vocab, in_timesteps,out_timesteps,units):
      model = Sequential()
      model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
      model.add(LSTM(units))
      model.add(RepeatVector(out_timesteps))
      model.add(LSTM(units, return_sequences=True))
      model.add(Dense(out_vocab, activation='softmax'))
      return model

def get_word(n, tokenizer):
      for word, index in tokenizer.word_index.items():
          if index == n:
              return word
      return None
#def translatePhrase(phrase):
#def translatePhrase(phrase):
#def translatePhrase(phrase):
def getEngTrans(phrase): 

  phrase = phrase.translate(str.maketrans('', '', string.punctuation))

  phrase = phrase.lower()

  encPhrase = encode_sequences(spa_tokenizer, spa_length, [phrase])

  prediction = loaded_model.predict_classes(encPhrase.reshape((encPhrase.shape[0],encPhrase.shape[1]))).tolist()[0]
  reverse_word_map = dict(map(reversed, spa_tokenizer.word_index.items()))
  [reverse_word_map.get(letter) for letter in prediction]

  preds_text = []
  temp = []
  for j in range(len(prediction)):
      t = get_word(prediction[j], eng_tokenizer)
      if j > 0:
          if (t == get_word(prediction[j-1], eng_tokenizer)) or (t == None):
              pass
          else:
              temp.append(t)
      else:
          if(t == None):
              temp.append('')
          else:
              temp.append(t)
    
  return " ".join(temp)

#def translatePhrase(phrase):
def getSpaTrans(phrase): 

  phrase = phrase.translate(str.maketrans('', '', string.punctuation))

  phrase = phrase.lower()

  encPhrase = encode_sequences(eng_tokenizer, eng_length, [phrase])

  prediction = loaded_model_2.predict_classes(encPhrase.reshape((encPhrase.shape[0],encPhrase.shape[1]))).tolist()[0]
  reverse_word_map = dict(map(reversed, eng_tokenizer.word_index.items()))
  [reverse_word_map.get(letter) for letter in prediction]

  preds_text = []
  temp = []
  for j in range(len(prediction)):
      t = get_word(prediction[j], spa_tokenizer)
      if j > 0:
          if (t == get_word(prediction[j-1], spa_tokenizer)) or (t == None):
              pass
          else:
              temp.append(t)
      else:
          if(t == None):
              temp.append('')
          else:
              temp.append(t)
    
  return " ".join(temp)

