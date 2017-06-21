---
layout: post
title:  "The wisdom of Marx with char-rnn"
date:   2017-06-17 15:43:00 +0700
tag: AI, marx, rnn, deep-learning
---

### Quick recurrent neural network text generation with Marx

Marx had many great and poor insights about society. I'm keen to see if a simple character LSTM recurrent neural network can reproduce either of the above by reading Das Kapital Vol.1, his masterpiece. Let's go!


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

I acquired the MS word version of [Das Kapital Volume 1](https://www.marxists.org/archive/marx/works/1867-c1/) and turned it into a text format I could load.


```python
raw_text = open('capital-vol1.txt', 'r', encoding='latin-1').read()
```

After chopping off the copyrights and the introduction, let's see snippets of it to get some idea of what we're dealing with.


```python
raw_text[:1000]
```




    'Chapter 1: Commodities\nSection 1: The Two Factors of a Commodity:\nUse-Value and Value\n(The Substance of Value and the Magnitude of Value)\nThe wealth of those societies in which the capitalist mode of production prevails, presents itself as \x93an immense accumulation of commodities,\x941 its unit being a single commodity. Our investigation must therefore begin with the analysis of a commodity. \nA commodity is, in the first place, an object outside us, a thing that by its properties satisfies human wants of some sort or another. The nature of such wants, whether, for instance, they spring from the stomach or from fancy, makes no difference.2 Neither are we here concerned to know how the object satisfies these wants, whether directly as means of subsistence, or indirectly as means of production. \nEvery useful thing, as iron, paper, &c., may be looked at from the two points of view of quality and quantity. It is an assemblage of many properties, and may therefore be of use in various ways. To d'



Hmm, there are a lot of non-english characters in there. Let's see them all listed in one place


```python
chars = sorted(set(raw_text))
print('corpus has ' + str(len(raw_text)) + ' letters altogether')
print ('corpus has ' + str(len(chars)) + ' unique characters:', chars)
```

    corpus has 1468303 letters altogether
    corpus has 108 unique characters: ['\n', ' ', '!', '"', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '\x91', '\x92', '\x93', '\x94', '\x96', '\xa0', '£', '°', '¼', '½', '¾', '×', 'à', 'â', 'æ', 'è', 'é', 'ê', 'î', 'ï', 'ô', 'û', 'ü']


Let's remove the numbers, the capitals, and the non-english characters to reduce the range of our output.


```python
import string
import unicodedata

#https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string
all_letters = string.ascii_lowercase + " .,;'-"
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

text = unicodeToAscii(raw_text)

chars = sorted(set(text))
print('corpus has ' + str(len(text)) + ' letters altogether')
print ('corpus has ' + str(len(chars)) + ' unique characters:', chars)
```

    corpus has 1427555 letters altogether
    corpus has 32 unique characters: [' ', "'", ',', '-', '.', ';', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

Looks much better!

### Creating input and output data for training
Now we need to create our training data. Our input (X) should be a string of text, say 100 characters long (we will call that the window size), and the output (y) should be the next character following it. We can decide how many samples we want by determining step size. If our step size is 1, we are essentially creating an output for every single letter in the text. If our step size is 20, we skip 20 letters ahead between each output, and our sample size would be (total letters)/(step size). 

Code and idea source: [Keras' example site](https://github.com/fchollet/keras/tree/master/examples), [Udacity's AI Nanodegree](http://udacity.com/ai) code, and [Andrej's 100 line python RNN code](https://gist.github.com/karpathy/d4dee566867f8291f086).


```python
chars_to_indices = dict((c, i) for i, c in enumerate(chars))
indices_to_chars = dict((i, c) for i, c in enumerate(chars))
```


```python
# create i/o by iterating over text, each step being step_size, to create a list of window of text of input, and the 
# letter following it for output

def window_transform_text(text, window_size, step_size):
    inputs = []
    outputs = []
    for i in range(0, len(text) - window_size, step_size):
        window = text[i:window_size+i]
        inputs.append(window)
    outputs = [i for i in text[window_size::step_size]]
    return inputs,outputs
```


```python
# Turn windows of text into vectors of one-hot encoding. 
# Andrej's implementation shapes both X and y to be window-sized. But I think predicting for 1 letter after X is more intuitive for
# a wider audience though, so we will go with that. 
    
def encode_io_pairs(text,window_size,step_size):
    chars = sorted(list(set(text)))
    num_chars = len(chars)
    
    inputs, outputs = window_transform_text(text,window_size,step_size)
    
    X = np.zeros((len(inputs), window_size, num_chars), dtype=np.bool)
    y = np.zeros((len(inputs), num_chars), dtype=np.bool)

    for i, sentence in enumerate(inputs):
        for t, char in enumerate(sentence):
            X[i, t, chars_to_indices[char]] = 1
        y[i, chars_to_indices[outputs[i]]] = 1
        
    return X, y
```


```python
window_size = 50
step_size = 3 
X, y = encode_io_pairs(text,window_size,step_size)
```

### Training the RNN model for text generation


```python
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint  
import keras
import random

model = Sequential()
model.add(LSTM(300, input_shape=(window_size, len(chars)), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(300))
model.add(Dropout(0.1))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.summary()

optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_11 (LSTM)               (None, 50, 300)           399600    
    _________________________________________________________________
    dropout_11 (Dropout)         (None, 50, 300)           0         
    _________________________________________________________________
    lstm_12 (LSTM)               (None, 300)               721200    
    _________________________________________________________________
    dropout_12 (Dropout)         (None, 300)               0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 32)                9632      
    _________________________________________________________________
    activation_6 (Activation)    (None, 32)                0         
    =================================================================
    Total params: 1,130,432
    Trainable params: 1,130,432
    Non-trainable params: 0
    _________________________________________________________________

Let's save two weights. The one with the least validation loss and the last

```python
checkpointer = ModelCheckpoint(filepath='best_RNN_large_textdata_weights.hdf5', verbose=1, save_best_only=True)
hist = model.fit(X, y, batch_size=512, nb_epoch=50, verbose = 2, validation_split=0.05, callbacks=[checkpointer])
model.save_weights('last_RNN_large_textdata_weights.hdf5')
```

    C:\Users\pete_tanru\AppData\Local\conda\conda\envs\tensorflow\lib\site-packages\keras\models.py:851: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
      warnings.warn('The `nb_epoch` argument in `fit` '


    Train on 452043 samples, validate on 23792 samples
    Epoch 1/50
    Epoch 00000: val_loss improved from inf to 1.70779, saving model to best_RNN_large_textdata_weights.hdf5
    128s - loss: 2.0944 - acc: 0.3958 - val_loss: 1.7078 - val_acc: 0.4957
    Epoch 2/50
    Epoch 00001: val_loss improved from 1.70779 to 1.46738, saving model to best_RNN_large_textdata_weights.hdf5
    126s - loss: 1.4946 - acc: 0.5602 - val_loss: 1.4674 - val_acc: 0.5662
    Epoch 3/50
    Epoch 00002: val_loss improved from 1.46738 to 1.38134, saving model to best_RNN_large_textdata_weights.hdf5
    126s - loss: 1.3278 - acc: 0.6043 - val_loss: 1.3813 - val_acc: 0.5876
    Epoch 4/50
    Epoch 00003: val_loss improved from 1.38134 to 1.33474, saving model to best_RNN_large_textdata_weights.hdf5
    126s - loss: 1.2436 - acc: 0.6261 - val_loss: 1.3347 - val_acc: 0.6039
    Epoch 5/50
    Epoch 00004: val_loss improved from 1.33474 to 1.30740, saving model to best_RNN_large_textdata_weights.hdf5
    126s - loss: 1.1851 - acc: 0.6416 - val_loss: 1.3074 - val_acc: 0.6120
    Epoch 6/50
    Epoch 00005: val_loss improved from 1.30740 to 1.28825, saving model to best_RNN_large_textdata_weights.hdf5
    126s - loss: 1.1403 - acc: 0.6533 - val_loss: 1.2883 - val_acc: 0.6183
    Epoch 7/50
    Epoch 00006: val_loss improved from 1.28825 to 1.27816, saving model to best_RNN_large_textdata_weights.hdf5
    126s - loss: 1.1030 - acc: 0.6632 - val_loss: 1.2782 - val_acc: 0.6206
    Epoch 8/50
    Epoch 00007: val_loss improved from 1.27816 to 1.26846, saving model to best_RNN_large_textdata_weights.hdf5
    126s - loss: 1.0692 - acc: 0.6722 - val_loss: 1.2685 - val_acc: 0.6275
    Epoch 9/50
    Epoch 00008: val_loss did not improve
    126s - loss: 1.0378 - acc: 0.6808 - val_loss: 1.2765 - val_acc: 0.6256
    Epoch 10/50
    Epoch 00009: val_loss did not improve
    126s - loss: 1.0109 - acc: 0.6880 - val_loss: 1.2790 - val_acc: 0.6289
    Epoch 11/50
    Epoch 00010: val_loss did not improve
    126s - loss: 0.9823 - acc: 0.6958 - val_loss: 1.2879 - val_acc: 0.6256
    Epoch 12/50
    Epoch 00011: val_loss did not improve
    126s - loss: 0.9580 - acc: 0.7019 - val_loss: 1.3045 - val_acc: 0.6224
    Epoch 13/50
    Epoch 00012: val_loss did not improve
    126s - loss: 0.9338 - acc: 0.7093 - val_loss: 1.3066 - val_acc: 0.6245
    Epoch 14/50
    Epoch 00013: val_loss did not improve
    126s - loss: 0.9109 - acc: 0.7146 - val_loss: 1.3203 - val_acc: 0.6223
    Epoch 15/50
    Epoch 00014: val_loss did not improve
    126s - loss: 0.8886 - acc: 0.7217 - val_loss: 1.3381 - val_acc: 0.6221
    Epoch 16/50
    Epoch 00015: val_loss did not improve
    126s - loss: 0.8699 - acc: 0.7266 - val_loss: 1.3513 - val_acc: 0.6210
    Epoch 17/50
    Epoch 00016: val_loss did not improve
    126s - loss: 0.8502 - acc: 0.7321 - val_loss: 1.3699 - val_acc: 0.6185
    Epoch 18/50
    Epoch 00017: val_loss did not improve
    126s - loss: 0.8322 - acc: 0.7369 - val_loss: 1.3677 - val_acc: 0.6197
    Epoch 19/50
    Epoch 00018: val_loss did not improve
    126s - loss: 0.8147 - acc: 0.7422 - val_loss: 1.3847 - val_acc: 0.6192
    Epoch 20/50
    Epoch 00019: val_loss did not improve
    126s - loss: 0.7991 - acc: 0.7460 - val_loss: 1.4077 - val_acc: 0.6182
    Epoch 21/50
    Epoch 00020: val_loss did not improve
    127s - loss: 0.7855 - acc: 0.7498 - val_loss: 1.4145 - val_acc: 0.6184
    Epoch 22/50
    Epoch 00021: val_loss did not improve
    126s - loss: 0.7719 - acc: 0.7539 - val_loss: 1.4412 - val_acc: 0.6150
    Epoch 23/50
    Epoch 00022: val_loss did not improve
    128s - loss: 0.7584 - acc: 0.7576 - val_loss: 1.4520 - val_acc: 0.6163
    Epoch 24/50
    Epoch 00023: val_loss did not improve
    128s - loss: 0.7463 - acc: 0.7615 - val_loss: 1.4710 - val_acc: 0.6111
    Epoch 25/50
    Epoch 00024: val_loss did not improve
    128s - loss: 0.7358 - acc: 0.7640 - val_loss: 1.4797 - val_acc: 0.6128
    Epoch 26/50
    Epoch 00025: val_loss did not improve
    127s - loss: 0.7251 - acc: 0.7672 - val_loss: 1.4965 - val_acc: 0.6101
    Epoch 27/50
    Epoch 00026: val_loss did not improve
    126s - loss: 0.7151 - acc: 0.7697 - val_loss: 1.5257 - val_acc: 0.6099
    Epoch 28/50
    Epoch 00027: val_loss did not improve
    127s - loss: 0.7054 - acc: 0.7720 - val_loss: 1.5272 - val_acc: 0.6107
    Epoch 29/50
    Epoch 00028: val_loss did not improve
    127s - loss: 0.6981 - acc: 0.7743 - val_loss: 1.5384 - val_acc: 0.6083
    Epoch 30/50
    Epoch 00029: val_loss did not improve
    127s - loss: 0.6897 - acc: 0.7768 - val_loss: 1.5377 - val_acc: 0.6112
    Epoch 31/50
    Epoch 00030: val_loss did not improve
    128s - loss: 0.6813 - acc: 0.7794 - val_loss: 1.5594 - val_acc: 0.6068
    Epoch 32/50
    Epoch 00031: val_loss did not improve
    129s - loss: 0.6743 - acc: 0.7810 - val_loss: 1.5694 - val_acc: 0.6057
    Epoch 33/50
    Epoch 00032: val_loss did not improve
    129s - loss: 0.6664 - acc: 0.7836 - val_loss: 1.5734 - val_acc: 0.6062
    Epoch 34/50
    Epoch 00033: val_loss did not improve
    128s - loss: 0.6591 - acc: 0.7857 - val_loss: 1.5918 - val_acc: 0.6062
    Epoch 35/50
    Epoch 00034: val_loss did not improve
    129s - loss: 0.6553 - acc: 0.7871 - val_loss: 1.5989 - val_acc: 0.6063
    Epoch 36/50
    Epoch 00035: val_loss did not improve
    127s - loss: 0.6491 - acc: 0.7885 - val_loss: 1.6176 - val_acc: 0.6049
    Epoch 37/50
    Epoch 00036: val_loss did not improve
    129s - loss: 0.6413 - acc: 0.7902 - val_loss: 1.6161 - val_acc: 0.6056
    Epoch 38/50
    Epoch 00037: val_loss did not improve
    129s - loss: 0.6354 - acc: 0.7923 - val_loss: 1.6333 - val_acc: 0.6032
    Epoch 39/50
    Epoch 00038: val_loss did not improve
    129s - loss: 0.6320 - acc: 0.7928 - val_loss: 1.6505 - val_acc: 0.6044
    Epoch 40/50
    Epoch 00039: val_loss did not improve
    129s - loss: 0.6278 - acc: 0.7945 - val_loss: 1.6527 - val_acc: 0.6036
    Epoch 41/50
    Epoch 00040: val_loss did not improve
    129s - loss: 0.6227 - acc: 0.7958 - val_loss: 1.6627 - val_acc: 0.6031
    Epoch 42/50
    Epoch 00041: val_loss did not improve
    129s - loss: 0.6181 - acc: 0.7970 - val_loss: 1.6738 - val_acc: 0.6005
    Epoch 43/50
    Epoch 00042: val_loss did not improve
    126s - loss: 0.6155 - acc: 0.7979 - val_loss: 1.6741 - val_acc: 0.6014
    Epoch 44/50
    Epoch 00043: val_loss did not improve
    126s - loss: 0.6104 - acc: 0.7993 - val_loss: 1.6885 - val_acc: 0.6027
    Epoch 45/50
    Epoch 00044: val_loss did not improve
    127s - loss: 0.6053 - acc: 0.8010 - val_loss: 1.6994 - val_acc: 0.6018
    Epoch 46/50
    Epoch 00045: val_loss did not improve
    127s - loss: 0.6020 - acc: 0.8022 - val_loss: 1.6984 - val_acc: 0.6026
    Epoch 47/50
    Epoch 00046: val_loss did not improve
    126s - loss: 0.5972 - acc: 0.8031 - val_loss: 1.7063 - val_acc: 0.6030
    Epoch 48/50
    Epoch 00047: val_loss did not improve
    127s - loss: 0.5947 - acc: 0.8040 - val_loss: 1.7197 - val_acc: 0.6001
    Epoch 49/50
    Epoch 00048: val_loss did not improve
    126s - loss: 0.5913 - acc: 0.8045 - val_loss: 1.7275 - val_acc: 0.5998
    Epoch 50/50
    Epoch 00049: val_loss did not improve
    127s - loss: 0.5872 - acc: 0.8061 - val_loss: 1.7334 - val_acc: 0.6015



```python
# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


![png]({{https://petetanru.github.io}}/images/marx_files/output_18_0.png)



![png]({{https://petetanru.github.io}}/images/marx_files/output_18_1.png)

Neat! Now let's try to generate texts based on those two weights. We will also try to generate with different temperature of softmax.

```python
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
```

We'll try out different level of diversity in characters through softmax temperature as well.

```python
def pred_text(weight):
    model.load_weights(weight)
    start_index = random.randint(0, len(text) - window_size - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + window_size]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, window_size, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, chars_to_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_to_chars[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
        print()

print("Printing out the best weight")
pred_text('best_RNN_large_textdata_weights.hdf5')

print("printing out the last weight")
pred_text('last_RNN_large_textdata_weights.hdf5')
```

    Printing out the best weight

    ----- diversity: 0.2
    ----- Generating with seed: "also secure its reproduction on a progressive scal"
    also secure its reproduction on a progressive scale, the capitalist as the manufacturers in the process of production and the productiveness of labour in the labourer is not the same time the manufacturers of the commodities with the same time the sum of the product of the labourer as the form of the capitalist produces the labourer is not the product of the working day as the same time the constant capital and the labourer is not the substance o


    ----- diversity: 0.5
    ----- Generating with seed: "also secure its reproduction on a progressive scal"
    also secure its reproduction on a progressive scale, and the increase of the self-expansion of the manufacturers expended in the orking to the coat, of the manufacturers, exclude in the compasit of the capitalist produces at the enormous of the product of the workman. n the other hand, the increase of industry with the labourer are isolated by the form of that machine, and the labourer, and the labourer in the labourer is not the capitalist as th


    ----- diversity: 1.0
    ----- Generating with seed: "also secure its reproduction on a progressive scal"
    also secure its reproduction on a progressive scale, as the number of its yard. he labour of all commodities. n the one hand, the for-allow that the igner of the counding ought of the first play, or gradual parts of a given count, which he wreeched the seller hour a soliary branch of its ewhomeous in the actory, what what imporeates his nire, the two years to pit into the cultivated name, theroby on the concrant many determinat person for two mea


    ----- diversity: 1.2
    ----- Generating with seed: "also secure its reproduction on a progressive scal"
    also secure its reproduction on a progressive scale, yet taxks great stranging pisturt, in which their natural fulsing, easily aux we his will, or hows these sinct of these-whole, in his labour imlements in the-exploitaty mode of production, of wealth has, therefore, asayt nscehence when, cately death by dovel devel -ready and particular. otencion cost, and ined this has given to that fasely beccal uncome, to yarf,   lbokting less uyed in the old
    
    printing out the last weight
    
    ----- diversity: 0.2
    ----- Generating with seed: "is both the product of a previous process, and a m"
    is both the product of a previous process, and a most of the operatives which in a more productive form, is the more rapid also use values, in the same proportion as the means of production, the most self-expansion of value, capital is to be a from the form of the capitalist mode of production, and to the increase of that labour, and the length of the working day, belong to the machine, that the work is determined by the capitalist and the price 


    ----- diversity: 0.5
    ----- Generating with seed: "is both the product of a previous process, and a m"
    is both the product of a previous process, and a motion of in a commodity, in the one case, the more rapid general conditions of capitalist production, has the bounds of capitalist production at the same time must have been represented, and that in the shape of the commodities comes time in the form of a six hours work of the day being consumed, and the consequence of a series of movement of capital. asts such as in the magnitude of surplus-value


    ----- diversity: 1.0
    ----- Generating with seed: "is both the product of a previous process, and a m"
    is both the product of a previous process, and a more and contaqued expression at the possessor of the surplus-produce does in order to develop a century part of the badent with the plaged good of the whole mill, while the assume of the activity of equation between lift of course that commodity  is not consequent in regulated the sum of the prices of commodities with fewer and conficulated off flought and degree of exploitation of labour, the lat


    ----- diversity: 1.2
    ----- Generating with seed: "is both the product of a previous process, and a m"
    is both the product of a previous process, and a most in the cost of machines must deperd but the individual labourers time of labour as profits of surplus-value sholds the very single commodity of an ordan ear are ,  fem in , w  sell, have completed which broad it replacates only their inventing the relation by individual production. he towns mines and it steps, during that portion of the general constitutive of their sense, the part of each day


We can observe that when we are really conservative (temperature = 0.2) we start to repeat outselves a lot. On the other extreme, pushing diversity beyond normal (T=1.2) starts to ruin our spelling ability. The sweet spot seems to be around 0.5 to 1.0 for our model. Further,
Also, to my surprise I think the latest weight (crazy overfit with horrific validation loss) performs better for this task. It makes less spelling mistakes.

### Marx, continue my sentence!

Now let's try to build some sentences ourselves and let the wisdom of Marx finish our sentence!
We will use our last weight and be a bit conservative (T=0.5) to deal with our ridiculous starting statement.

```python
def pred_my_text(text):
    model.load_weights('last_RNN_large_textdata_weights.hdf5')
    diversity = 0.5
    print()
    print('----- diversity:', diversity)

    generated = ''
    sentence = text
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(400):
        x = np.zeros((1, window_size, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, chars_to_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_to_chars[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
    print()
```


```python
pred_my_text("someday a mutant may invade the world and we will ")
```


    ----- diversity: 0.5
    ----- Generating with seed: "someday a mutant may invade the world and we will "
    someday a mutant may invade the world and we will be developed commence to the sum of the prices of the commodities produced by machinery, compare which a six days to , is stage merely advanced in different industries, and the possessive labour-power and in the last to realise the requires of the commodities not only commodities, as they would have to be represented by a population of in the enample of the instruments of labour increases, and the

Not bad!

```python
pred_my_text("someday a mutant may invade the world and we will")
```

We remove one character (making our text length 49 rather than the trained 50) and things break down right away

    ----- diversity: 0.5
    ----- Generating with seed: "someday a mutant may invade the world and we will"
    someday a mutant may invade the world and we willnls apetai  ubeoiyy aditac...  ..  rsdi n edaalyua epcs-ni rg,se,al i i etyo i rvno prcoetn.. ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  .. .. .. eepitetyyddrplszd....eros.eanoo,oea,w,nnee,cekn,c,clhresrciknnc,cnin,ohr,o,cal,oecor,mnlxin,,c,ilweesykntelsaslecedermn,cutevl,ols,ralkkhoselpos,selqietlsevrsrkls,,oeculyryosls,ckl-l-sigatd,mels,erals,iretese,wetalsrrasssmeti

Conversely, a blank statement of 50 character length picks itself up and starts to generate legible material

```python
pred_my_text("                                                  ")
```


    ----- diversity: 0.5
    —---- Generating with seed: "                                                  "
                                                              b.bbllof,, ,,  c. o ut that the ingulation of a commodity of a house, the labourer has no existence of a hoars of labour. he continual productivity of old commodities, as the only form of the capitalist mode of production, and of commodities is required to replace and of furnacty the labour of the necessary labour-time in no more than an article of the money that the labourer has no length

### Conclusion (so far)
1. Char-rnn basically overfits a text to regurgitates stuff back out. The best performing model is not the one with least loss.
2. Not much insight to be gained over bag of words..
3. Really big models and texts don't fit into my GPU memory. Perhaps there's a way to do it on Pytorch.

### TODO
1. Test whether random sampling can do as well reading text from left to write (X, y generation function)
2. Test whether redundant windows actually improve the performance (as opposed to just increasing epochs)
