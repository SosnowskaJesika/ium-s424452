import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np

df = pd.read_csv('prepared_data.csv')

# Podział na cechy i etykiety
X = df.drop(['rating'], axis=1)
y = df['rating']

# Konwersja etykiet na klasy
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Wczytanie wytrenowanego modelu
model = tf.keras.models.load_model('model.h5')

# Ewaluacja modelu
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=-1)

accuracy = accuracy_score(y, y_pred_classes)

print(f'Accuracy: {accuracy}')
