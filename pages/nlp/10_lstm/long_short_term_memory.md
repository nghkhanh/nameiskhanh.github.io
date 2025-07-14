# Long Short Term Memory

## Introduction

Long Short-Term Memory (LSTM) networks enhance Recurrent Neural Networks by effectively handling long-term dependencies using gate mechanisms. This makes them ideal for tasks like language translation, speech recognition, and time series prediction. Let's dive into how these powerful networks work!

## What is Long Short Term Memory?
As we discussed in **Recurrent Neural Network** lesson, **long range dependencies** as well as **Gradient Vanishing** are  popular problems when training RNN models , the appearance of **Long Short Term Memory** is a savior.

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) designed to overcome the limitations of traditional RNNs in sequential data. It achieves this by using a memory cell that can maintain information over long periods, allowing it to remember relevant context even when the input sequence is lengthy. LSTMs accomplish this through a series of specialized gates that control the flow of information into and out of the memory cell, enabling effective learning and retention of sequential patterns.

## Why we need to use Long Short Term Memory?
![](images/LSTMAdvantages.png)

## Long Short Term Memory Architecture
![](images/LSTMArchitecute.png)

## How Long Short Term Memory works?
### Forget Gate

![](images/LSTMForgetGate.png)

This equation represents the forget gate in a Long Short-Term Memory (LSTM) network. It calculates the forget gate's activation by combining the current input $x_{t}$ and the previous hidden state $h_{t-1}$, and then applying the sigmoid function to control the amount of information to forget from the previous cell state.

### Input Gate
![](images/LSTMInputGate.png)

These equations represent the input gate and the candidate cell state in a Long Short-Term Memory (LSTM) network with:
+ Input Gate: This equation calculates the input gate's activation, combining the current input $x_{t}$ and the previous hidden state $h_{t-1}$, and then applying the sigmoid function to control how much new information to allow into the cell state.
+ Candidate Cell State: This equation calculates the candidate cell state by combining the current input $x_{t}$ and the previous hidden state $h_{t-1}$, and then applying the $tanh$ function to produce the new information to be added to the cell state.

### Update Operation
![](images/LSTMUpdateOperation.png)

This equation updates the cell state $C_{t}$ by combining:
+ The previous cell state $C_{t-1}$ scaled by the forget gate $f_{t}$, deciding how much of the previous state to keep.
+ The candidate cell state $\tilde{C}_{t}$ scaled by the input gate $i_{t}$, deciding how much new information to add.

### Output Gate
![](images/LSTMOutputGate.png)

These equations represent the output gate and the hidden state in a Long Short-Term Memory (LSTM) network with:
+ Output Gate: This equation calculates the output gate's activation by combining the current input $x_{t}$ and the previous hidden state $h_{t-1}$​, and then applying the sigmoid function to control the output from the cell state.
+ Hidden State: This equation determines the hidden state $h_{t}$ by applying the $tanh$ function to the cell state $C_{t}$ to produce an output candidate, and then scaling it by the output gate's activation $o_{t}$. This determines what portion of the cell state will be passed to the next time step and as the output of the LSTM at the current time step.


### Make a decision
![](images/LSTMDecision.png)

### Sentence Classification
![](images/SentenceProblem.png)

## Implement LSTM Model
In this session, we will build a LSTM model for Sarcasm Detection. This lab was implemented on **Google Colab**.

### Download dataset
We will use **News Headlines Dataset** for this project. This dataset is collected from two news website are [TheOnion](https://www.theonion.com/) and [HuffPost](https://www.huffpost.com/).
Each record of the dataset consists of three attributes:
- **is_sarcastic**: 1 if the record is sarcastic otherwise 0
- **headline**: the headline of the news article
- **article_link**: link to the original news article. Useful in collecting supplementary data

```python
!wget https://raw.githubusercontent.com/dunghoang369/data/master/Sarcasm_Headlines_Dataset.json
```


### Import necessary libraries
```python
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_reportEach record consists of three attributes:
from tensorflow.keras.layers import Embedding, RNN, Dense, LSTM, Bidirectional
```

### Load dataset
```python
df = pd.read_json("Sarcasm_Headlines_Dataset.json", lines=True)
datastore = df.to_json()
datastore = json.loads(datastore)
```

### Split features
```python
article_link_datastore = datastore["article_link"]
headline_datastore = datastore["headline"]
sarcastic_datastore = datastore["is_sarcastic"]

sentences = []
urls = []
labels = []
table = str.maketrans('', '', string.punctuation)
for key in article_link_datastore:
    sentences.append(headline_datastore[key].lower())
    urls.append(article_link_datastore[key])
    labels.append(sarcastic_datastore[key])

# Print some samples
print("Sample 1: ", sentences[0], urls[0], labels[0])
print("Sample 2: ", sentences[1], urls[1], labels[1])

Sample 1: former versace store clerk sues over secret 'black code' for minority shoppers, https://www.huffingtonpost.com/entry/versace-black-code_us_5861fbefe4b0de3a08f600d5, 0
Sample 2: the 'roseanne' revival catches up to our thorny political mood, for better and worse, https://www.huffingtonpost.com/entry/roseanne-revival-review_us_5ab3a497e4b054d118e04365, 0
```

### Define some hyperparameters
```python
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
```

### Split train, test datasets
```python
training_sentences = np.array(sentences[:training_size])
training_labels = np.array(labels[:training_size])
test_sentences = np.array(sentences[training_size:])
test_labels = np.array(labels[training_size:])
```

### Build tokenizer
```python
tokenizer = tf.keras.preprocessing.text.Tokenizer(vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
training_sequences = tokenizer.texts_to_sequences(training_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)
```

### Padding whole dataset
```python
training_padded = tf.keras.preprocessing.sequence.pad_sequences(training_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')
```
We will pad 0 to the back of each sequence in **train_dataset** and **test_dataset** to create the same length senquences in one batch.

### Build Bidirectional LSTM
```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=120))
model.add(Bidirectional(LSTM(32)))
model.add(tf.keras.layers.Dense(24, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
print(model.summary())

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 120, 16)           16000     
                                                                 
 bidirectional (Bidirection  (None, 64)                12544     
 al)                                                             
                                                                 
 dense (Dense)               (None, 24)                1560      
                                                                 
 dense_1 (Dense)             (None, 1)                 25        
                                                                 
=================================================================
Total params: 30129 (117.69 KB)
Trainable params: 30129 (117.69 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```


### Train the model
```python
# Set up callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint('best.h5',
                                                save_best_only=True,
                                                mode='min')
callbacks = [checkpoint]

# Set up optimizer, loss function and metrics
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(training_padded, training_labels, batch_size=32, epochs=50, callbacks=callbacks, validation_data=(test_padded, test_labels))
```

### Evaluate  the model
```python
predictions = model.predict(test_padded)
predictions = np.array([1 if prediction[0] > 0.5 else 0 for prediction in predictions])
print(classification_report(test_labels, predictions))

210/210 [==============================] - 2s 8ms/step
              precision    recall  f1-score   support

           0       0.82      0.83      0.83      3779
           1       0.78      0.76      0.77      2930

    accuracy                           0.80      6709
   macro avg       0.80      0.80      0.80      6709
weighted avg       0.80      0.80      0.80      6709
```
### Inference
```python
def inference(text):
  text = np.array([text])
  sequences = tokenizer.texts_to_sequences(text)
  padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
  predict = model.predict(padded)
  if predict > 0.5:
    label = 1
  else: 
    label = 0
  print(f"Label: {['Sarcastic', 'Normal'][label]}")

1/1 [==============================] - 0s 20ms/step
Label: Sarcastic
```

## Conclusion

In conclusion, today's lesson Long Short-Term Memory (LSTM) networks has provided a deep dive into their structure and functionality. We explored how LSTMs use gates—forget, input, and output—to effectively manage long-term dependencies and overcome the limitations of traditional RNNs.

## References

+ C. Olah, “Understanding LSTM Networks,” Github.io, Aug. 27, 2015. https://colah.github.io/posts/2015-08-Understanding-LSTMs/
+ M. Phi, “Illustrated Guide to LSTM’s and GRU’s: A step by step explanation,” Medium, Jul. 10, 2019. https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21