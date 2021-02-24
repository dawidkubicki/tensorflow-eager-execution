Preprocessing subtasks:

1. Split dataset into validation (15k examples) and testing (10k examples)
2. With tf.data create a datasets
3. Create a model fo a binary classification with some text preparation
	a) TextVectorization will preprocess the data
	b) Embedding layer with mean representation multiply by mean squared from the number of words.
	c) Train the model
	d) Download tfrom the number of he dataset easier - using TFDS: tfds.load("imdb_reviews)

