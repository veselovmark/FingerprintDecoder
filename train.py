import tensorflow as tf

from utils import DataSet
from models import EncoderModel

tf.app.flags.DEFINE_string("dataset", None, 'Path to dataset npz.')
tf.app.flags.DEFINE_integer("batch_size", 128, 'Size of train and valid batches.')
FLAGS = tf.app.flags.FLAGS

def main(*args):
	# Hyper parameters
	learning_rate = 0.001
	training_steps = 10000
	valid_step = 50

	dataset = DataSet(FLAGS.dataset)
	print(dataset.samples_shape)
	print(dataset.labels_shape)

if __name__ == "__main__":
    tf.app.run()
