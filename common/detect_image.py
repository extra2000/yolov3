import logging

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

from common.image_preprocess import image_preprocess
from common.postprocess_boxes import postprocess_boxes
from common.nms import nms
from common.draw_bbox import draw_bbox
from common.read_class_names import read_class_names
from common.read_pb_return_tensors import read_pb_return_tensors

logger = logging.getLogger(__name__)
tf.compat.v1.disable_eager_execution()


def detect_image(pb_filename, classnames_filename, src_img, dst_img):
    """Perform YOLOv3 object detection

    Parameters
    ----------
    pb_filename : str
        Source weights in PB file format
    classnames_filename : str
        A file containing classnames
    src_img : str
        Source image
    dst_img : str
        Output image
    """

    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    classnames      = read_class_names(classnames_filename)
    pb_file         = pb_filename
    image_path      = src_img
    num_classes     = len(classnames)
    input_size      = 416
    graph           = tf.Graph()

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]

    return_tensors = read_pb_return_tensors(graph, pb_file, return_elements)

    with tf.compat.v1.Session(graph=graph) as sess:
        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = nms(bboxes, 0.45, method='nms')
    image = draw_bbox(original_image, bboxes, classnames)
    cv2.imwrite(dst_img, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # image = Image.fromarray(image)
    # image.show()
