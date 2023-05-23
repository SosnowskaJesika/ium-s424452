import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./anime.csv')
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
df.to_csv('prepared_data.csv', index=False)
