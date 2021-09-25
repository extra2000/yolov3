import logging

import tensorflow as tf

from yolov3 import YOLOv3

logger = logging.getLogger(__name__)
tf.compat.v1.disable_eager_execution()


def freeze_graph(config, chkpt_filename, pb_filename, classnames_filename):
    """Freeze graph

    Parameters
    ----------
    config : dict
        Configurations
    chkpt_filename : str
        Inference checkpoint filename
    pb_filename : src
        PB filename to be written
    classnames_filename : str
        A file containing classnames
    """

    pb_file = pb_filename
    ckpt_file = chkpt_filename
    output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

    config['yolov3']['classnames'] = classnames_filename

    with tf.name_scope('input'):
        input_data = tf.compat.v1.placeholder(dtype=tf.float32, name='input_data')

    model = YOLOv3(input_data, trainable=False, config=config)
    print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

    sess  = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, ckpt_file)

    converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                input_graph_def  = sess.graph.as_graph_def(),
                                output_node_names = output_node_names)

    with tf.io.gfile.GFile(pb_file, "wb") as f:
        f.write(converted_graph_def.SerializeToString())
