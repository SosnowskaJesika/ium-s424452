import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Wczytanie danych
df = pd.read_csv('anime_test.csv')
df = df.replace('Unknown', -1)
df = df.drop(['anime_id'], axis=1)
df['genre'] = df['genre'].fillna('')
df['type'] = df['type'].fillna('')
df['rating'] = df['rating'].fillna(0)
df['members'] = df['members'].fillna(0)
encoder = LabelEncoder()
df['name'] = encoder.fit_transform(df['name'])
df['genre'] = encoder.fit_transform(df['genre'])
df['type'] = encoder.fit_transform(df['type'])
df = df.astype('float32')

# Podział na cechy i etykiety
X = df.drop(['rating'], axis=1)
y = df['rating']

# Wczytanie wytrenowanego modelu
model = tf.keras.models.load_model('model.h5')

# Ewaluacja modelu
y_pred = model.predict(X)
y_pred_classes = [round(pred[0]) for pred in y_pred]

accuracy = accuracy_score(y, y_pred_classes)
print(f'Accuracy: {accuracy}')