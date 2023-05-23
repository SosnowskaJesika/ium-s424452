import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

df = pd.read_csv('prepared_data.csv')

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