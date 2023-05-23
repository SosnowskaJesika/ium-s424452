import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('data/prepared_data.csv')

X_train, X_test, y_train, y_test = train_test_split(df.drop(['rating'], axis=1), df['rating'], test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='RMSprop', loss='mse', metrics=['mae', 'mse'])

model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2)

model_filename = 'models/trained_model.pkl'
model.save(model_filename)
