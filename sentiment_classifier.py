import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print('gpu:', gpu)
    tf.config.experimental.set_memory_growth(gpu, True)
    print('memory growth:' , tf.config.experimental.get_memory_growth(gpu))

dir_path = "/home/dawidkubicki/Datasets/IMDB"

import pandas as pd

train_data = pd.read_csv("/home/dawidkubicki/Datasets/IMDB/Train.csv")
test_data = pd.read_csv("/home/dawidkubicki/Datasets/IMDB/Test.csv")
valid_data = pd.read_csv("/home/dawidkubicki/Datasets/IMDB/Valid.csv")

def clean_text(text):
    text = text.lower()
    return text

train_data["text"] = train_data["text"].apply(lambda x: clean_text(x)) 
valid_data["text"] = valid_data["text"].apply(lambda x: clean_text(x))

print(train_data["text"].head())
print(valid_data["text"].head())

X_train = train_data["text"]
y_train = train_data["label"]

X_valid = valid_data["text"]
y_valid = valid_data["label"]

X_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(y_train)
X_valid = tf.convert_to_tensor(X_valid)
y_valid = tf.convert_to_tensor(y_valid)

#embedding hyperparameters

vocab_size = 10000
sequence_length = 100
emb_dim = 256
rnn_units = 128

vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)

vectorize_layer.adapt(X_train)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(32)
valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).shuffle(10000).batch(32)

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim, rnn_units):
        super(MyModel, self).__init__()
        self.emb = tf.keras.layers.Embedding(vocab_size, emb_dim)
        self.lstm = tf.keras.layers.LSTM(rnn_units, activation='relu')
        self.d1 = tf.keras.layers.Dense(2)

    def call(self, x):
        x = self.emb(x)
        x = self.lstm(x)
        return  self.d1(x)


model = MyModel(vocab_size, emb_dim, rnn_units)
