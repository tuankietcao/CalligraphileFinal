import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import streamlit as st
import keras

def train_model(sample_per_letter, epochs):
  data_path = r'C:\Users\Admin\Desktop\PROJECT\Images\Images'
  data = []
  folders = os.listdir(data_path)

  for folder in folders:
      files = os.listdir(os.path.join(data_path, folder))
      count = 0
      for f in files:
          if count >= sample_per_letter:  #giảm để cho nhanh
              break
          if f.endswith('.png'):
              img = Image.open(os.path.join(data_path, folder, f))
              img = img.resize((32, 32))
              img = img.convert('L') #grayscale
              img = np.asarray(img)
              data.append([img, folder])
          count += 1

  train_data, test_data = train_test_split(data, test_size=0.2)

  X_train = []
  Y_train = []
  for img, label in train_data:
      X_train.append(img)
      Y_train.append(label)

  X_test = []
  Y_test = []
  for img, label in test_data:
      X_test.append(img)
      Y_test.append(label)


  alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  char_to_index = {char: i for i, char in enumerate(alphabet)}

  Y_train = to_categorical([char_to_index[label] for label in Y_train], num_classes=26)
  Y_test = to_categorical([char_to_index[label] for label in Y_test], num_classes=26)


  X_train = np.array(X_train) / 255.0
  Y_train = np.array(Y_train)

  X_test = np.array(X_test) / 255.0
  Y_test = np.array(Y_test)
  print(X_train.shape, Y_train.shape, Y_test.shape)

  model = Sequential()

  model.add(Input(shape=X_train.shape[1:]))
  model.add(Flatten())  
  model.add(Dense(26, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  history = model.fit(X_train, Y_train, epochs=epochs, verbose=1)
  model.summary()
  model.evaluate(X_train, Y_train)
  # model.add(Input(shape=X_train.shape[1:]))

  # model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  # model.add(Flatten())
  # model.add(Dense(256, activation='relu'))
  # model.add(Dropout(0.2))
  # model.add(Dense(len(set(folders)), activation='softmax'))

  model_dir = "saved_models"
  os.makedirs(model_dir, exist_ok=True)
  model.save(os.path.join(model_dir, "my_model.keras"))
  return model, history

# Function to load the trained model from a file (optional)
def load_model(model_path):
  try:
    return keras.models.load_model(model_path)
  except FileNotFoundError:
    st.error("Trained model not found. Please train a model first.")
    return None
  