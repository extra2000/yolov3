import os
import logging
import time

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


def detect_video(pb_filename, classnames_filename, src_vid, png_seq_pattern):
    """Perform YOLOv3 object detection on a video and saves as PNG sequences

    Parameters
    ----------
    pb_filename : str
        Source weights in PB file format
    classnames_filename : str
        A file containing classnames
    src_vid : str
        Source video
    png_seq_pattern : str
        Directory including filename pattern to store output PNG sequences.
    """

    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    pb_file         = pb_filename
    video_path      = src_vid
    # video_path      = 0
    num_classes     = 80
    input_size      = 416
    graph           = tf.Graph()
    return_tensors  = read_pb_return_tensors(graph, pb_file, return_elements)

    if os.path.exists(os.path.dirname(png_seq_pattern)):
        raise FileExistsError(os.path.dirname(png_seq_pattern))

    os.makedirs(os.path.dirname(png_seq_pattern))

    with tf.compat.v1.Session(graph=graph) as sess:
        vid = cv2.VideoCapture(video_path)
        frame_counter = 1
        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                logger.info("No more frames to process, exiting ...")
                break
            frame_size = frame.shape[:2]
            image_data = image_preprocess(np.copy(frame), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]
            prev_time = time.time()

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={ return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
            bboxes = nms(bboxes, 0.45, method='nms')
            image = draw_bbox(frame, bboxes, read_class_names(classnames_filename))

            curr_time = time.time()
            exec_time = curr_time - prev_time

            cv2.imwrite(png_seq_pattern % frame_counter, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            frame_counter += 1

            info = "time: %.2f ms" %(1000*exec_time)
