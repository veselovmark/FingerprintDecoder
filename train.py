import tensorflow as tf

from utils import DataSet
from models import EncoderModel as Model

tf.app.flags.DEFINE_string("dataset", None, 'Path to dataset npz.')
tf.app.flags.DEFINE_integer("batch_size", 128, 'Size of train and valid batches.')
FLAGS = tf.app.flags.FLAGS

def main(*args):
	# Hyper parameters
	learning_rate = 0.001
	training_steps = 10000
	valid_step = 50
	cell_size = 256
	num_rnn_layers = 2

	dataset = DataSet(FLAGS.dataset)
	model = Model(dataset.samples_shape[1],
				  dataset.labels_shape[1],
				  dataset.labels_shape[2],
				  cell_size,
				  num_rnn_layers,
				  learning_rate)

if __name__ == "__main__":
    tf.app.run()
