import tensorflow as tf
#\
import numpy as np
a = np.random.uniform(size=(100,2))
dataset = tf.data.Dataset.from_tensor_slices(a)