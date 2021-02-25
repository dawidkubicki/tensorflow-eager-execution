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

X_train = train_data["text"]
y_train = train_data["label"]

X_valid = valid_data["text"]
y_valid = valid_data["label"]


