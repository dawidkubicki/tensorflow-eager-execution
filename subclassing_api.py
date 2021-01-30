#tensorflow subclassing API


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print('gpu', gpu)
    tf.config.experimental.set_memory_growth(gpu, True)
    print('memory growth:' , tf.config.experimental.get_memory_growth(gpu))

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

#batch and shuffle data
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)


#model with init and call
class MyModel(tf.keras.Model):
	def __init__(self):
		super(MyModel, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu")
		self.flatten = tf.keras.layers.Flatten()
		self.d1 = tf.keras.layers.Dense(128, activation="relu")
		self.d2 = tf.keras.layers.Dense(1, activation="sigmoid")
	def call(self, x):
		x = self.conv1(x)
		x = self.flatten(x)
		x = self.d1(x)
		return self.d2(x)

model = MyModel()

#optimizer and loss function

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.Adam()

#loss and accuracy metrics

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

#gradient_tape to train a model

def train_step(images, labels):
	with tf.GradientTape() as tape:
		predictions = model(images, training=True)
		loss = loss_object(labels, predictions)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	train_loss(loss)
	train_accuracy(labels, predictions)


#gradient_tape to test a model


def test_step(images, labels):
	predictions = model(images, training=False)
	t_loss = loss_object(labels, predictions)

	test_loss(t_loss)
	test_accuracy(labels, predictions)


#training 

EPOCHS=5

for epoch in range(EPOCHS):
	train_loss.reset_states()
	train_accuracy.reset_states()
	test_loss.reset_states()
	test_accuracy.reset_states()

	for images, labels in train_ds:
		train_step(images, labels)

	for test_images, test_labels in test_ds:
		test_step(test_images, test_labels)

	print(
		f'Epoch {epoch + 1}. '
		f'Loss {train_loss.result()}. '
		f'Accuracy {train_accuracy.result()}, * 100 '
		f'Test loss {test_loss.result()}, '
		f'Test accuracy {test_accuracy.result()}, * 100'
	)
