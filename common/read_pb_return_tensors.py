import logging

import tensorflow as tf

logger = logging.getLogger(__name__)
tf.compat.v1.disable_eager_execution()


def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.compat.v1.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.compat.v1.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def, return_elements=return_elements)

    return return_elements
