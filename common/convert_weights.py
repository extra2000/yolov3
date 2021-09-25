import logging

import tensorflow as tf

from yolov3 import YOLOv3

logger = logging.getLogger(__name__)
tf.compat.v1.disable_eager_execution()


def convert_weights(config, src_weights, dst_weights, target, classnames_filename):
    """Convert weights

    Parameters
    ----------
    config : dict
        Configurations
    src_weights : str
        Source weights
    dst_weights : str
        Destination weights
    target : str
        Type of conversion to perform
    classnames_filename : str
        A file containing classnames
    """

    preserve_dst_names = ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']
    preserve_src_names = ['Conv_6', 'Conv_14', 'Conv_22']

    config['yolov3']['classnames'] = classnames_filename

    src_weights_mess = []
    tf.Graph().as_default()
    load = tf.compat.v1.train.import_meta_graph('{}.meta'.format(src_weights))
    with tf.compat.v1.Session() as sess:
        load.restore(sess, src_weights)
        for var in tf.compat.v1.global_variables():
            var_name = var.op.name
            var_name_mess = str(var_name).split('/')
            var_shape = var.shape
            if target == 'training':
                if var_name_mess[-1] not in ['weights', 'gamma', 'beta', 'moving_mean', 'moving_variance']:
                    # Removes "yolov3/darknet-53/Conv_*/biases" and "yolov3/yolo-v3/Conv_*/biases" layers
                    continue
                if (var_name_mess[1] == 'yolo-v3') and (var_name_mess[-2] in preserve_src_names):
                    # Removes "yolov3/yolo-v3/Conv_*/*" if matched in
                    # ['yolov3/yolo-v3/Conv_6/*', 'yolov3/yolo-v3/Conv_14/*', 'yolov3/yolo-v3/Conv_22/*']
                    continue
            src_weights_mess.append([var_name, var_shape])
            logger.debug('=> {} {}'.format(str(var_name).ljust(50), var_shape))

    tf.compat.v1.reset_default_graph()

    cur_weights_mess = []
    tf.Graph().as_default()
    with tf.name_scope('input'):
        # NOTE: 416x416 is YOLOv3 native resolution.
        input_data = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 416, 416, 3), name='input_data')
        training = tf.compat.v1.placeholder(dtype=tf.bool, name='trainable')
    model = YOLOv3(input_data, training, config)
    for var in tf.compat.v1.global_variables():
        var_name = var.op.name
        var_name_mess = str(var_name).split('/')
        var_shape = var.shape
        print(var_name_mess[0])
        if target == 'training':
            if var_name_mess[0] in preserve_dst_names:
                continue
        cur_weights_mess.append([var_name, var_shape])
        print("=> " + str(var_name).ljust(50), var_shape)

    org_weights_num = len(src_weights_mess)
    cur_weights_num = len(cur_weights_mess)
    if cur_weights_num != org_weights_num:
        raise RuntimeError

    print('=> Number of weights that will rename:\t%d' % cur_weights_num)
    cur_to_org_dict = {}
    for index in range(org_weights_num):
        org_name, org_shape = src_weights_mess[index]
        cur_name, cur_shape = cur_weights_mess[index]
        if cur_shape != org_shape:
            print(src_weights_mess[index])
            print(cur_weights_mess[index])
            raise RuntimeError
        cur_to_org_dict[cur_name] = org_name
        print("=> " + str(cur_name).ljust(50) + ' : ' + org_name)

    with tf.name_scope('load_save'):
        name_to_var_dict = {var.op.name: var for var in tf.compat.v1.global_variables()}
        restore_dict = {cur_to_org_dict[cur_name]: name_to_var_dict[cur_name] for cur_name in cur_to_org_dict}
        load = tf.compat.v1.train.Saver(restore_dict)
        save = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        for var in tf.compat.v1.global_variables():
            print("=> " + var.op.name)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        print('=> Restoring weights from:\t %s' % src_weights)
        load.restore(sess, src_weights)
        save.save(sess, dst_weights)
    tf.compat.v1.reset_default_graph()
