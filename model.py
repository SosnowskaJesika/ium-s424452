import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver

ex = Experiment("s424452")
ex.observers.append(MongoObserver(url="mongodb://admin:IUM_2021@172.17.0.1:27017"))
ex.observers.append(FileStorageObserver.create('my_runs'))

@ex.config
def my_config():
    batch_size = 32
    epochs = 100
    validation_split = 0.2
    optimizer = "RMSprop"
    loss = "mse"
    metrics = ["mae", "mse"]

@ex.automain
def my_main(batch_size, epochs, validation_split, optimizer, loss, metrics):
    df = pd.read_csv('anime.csv')
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

    X_train, X_test, y_train, y_test = train_test_split(df.drop(['rating'], axis=1), df['rating'], test_size=0.2, random_state=42)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    model_filename = 'anime_model.h5'
    model.save(model_filename)
    ex.add_artifact(model_filename)

    metrics_dict = {metric: score for metric, score in zip(model.metrics_names, model.evaluate(X_test, y_test))}
    ex.log_scalar("metrics", metrics_dict)

    ex.add_resource('anime.csv')
    ex.add_resource('anime_test.csv')

    ex.add_config({"batch_size": batch_size, "epochs": epochs, "validation_split": validation_split, "optimizer": optimizer, "loss": loss, "metrics": metrics})
