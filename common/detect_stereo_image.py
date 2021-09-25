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


class StereoVision:
    """Stereo Vision
    """

    def __init__(self):
        """Implement Stereo Vision
        """
        self.cfg = None


    @staticmethod
    def get_disparity_map(leftimage, rightimage, **kwargs):
        """
        Combine stereo image into a single image which represents disparity map.

        Parameters
        ----------
        leftimage : str
            Left image filename.
        rightimage : str
            Right image filename.
        min_disparity : int
            Minimum disparity. Default is 0.
        num_disparities : int
            Number of disparities. Default is 300.
        block_size : int
            Block size. Default is 1.
        disp12_max_diff : int
            Maximum allowed difference. Default is -1.
        uniqueness_ratio : int
            Uniqueness ratio. Default is 15.
        speckle_window_size : int
            Speckle window size. Default is 200.
        speckle_range : int
            Speckle range. Default is 64.
        sgbm_mode : str
            SGBM mode. Available choices are 'hh', '3way', or 'sgbm'. Default is '3way'.

        Returns
        -------
        cv2.image
            Disparity map with the same dimension as input image.
        """

        img_left = cv2.imread(leftimage, 0)
        img_right = cv2.imread(rightimage, 0)

        # Setting parameters for StereoSGBM algorithm
        min_disparity = kwargs.get('min_disparity', 0)
        num_disparities = kwargs.get('num_disparities', 300)
        block_size = kwargs.get('block_size', 1)
        disp12_max_diff = kwargs.get('disp12_max_diff', -1)
        uniqueness_ratio = kwargs.get('uniqueness_ratio', 15)
        speckle_window_size = kwargs.get('speckle_window_size', 200)
        speckle_range = kwargs.get('speckle_range', 64)

        param_sgbm_mode = kwargs.get('sgbm_mode', '3way')

        if param_sgbm_mode == 'hh':
            sgbm_mode = cv2.STEREO_SGBM_MODE_HH
        elif param_sgbm_mode == '3way':
            sgbm_mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        elif param_sgbm_mode == 'sgbm':
            sgbm_mode = cv2.STEREO_SGBM_MODE_SGBM
        else:
            raise ValueError(param_sgbm_mode)

        # Creating an object of StereoSGBM algorithm
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            disp12MaxDiff=disp12_max_diff,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            mode=sgbm_mode
        )

        # Calculating disparity using the StereoSGBM algorithm
        disparity_map = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

        return disparity_map


    @staticmethod
    def write_disparity_map(filename, outimage):
        """Write disparity map to image file

        Parameters
        ----------
        filename : str
            Filename to write output image.
        outimage : cv2.image
            Output image.
        """

        normimg = np.abs(cv2.normalize(outimage, 0, 255, cv2.NORM_MINMAX)*255).astype(np.uint8)
        cv2.imwrite(filename, normimg)


def detect_stereo_image(pb_filename, classnames_filename, src_left_img, src_right_img, dst_img):
    """Perform YOLOv3 object detection on a single stereo image

    Parameters
    ----------
    pb_filename : str
        Source weights in PB file format
    classnames_filename : str
        A file containing classnames
    src_left_img : str
        Source left image filename
    src_right_img : str
        Source right image filename
    dst_img : str
        Output image filename
    """

    stereovision = StereoVision()
    disparity_map = stereovision.get_disparity_map(
        src_left_img,
        src_right_img
    )
    disparity_map_norm = np.abs(cv2.normalize(disparity_map, 0, 255, cv2.NORM_MINMAX)*255).astype(np.uint8)
    disparity_map_rgb = cv2.cvtColor(disparity_map_norm, cv2.COLOR_GRAY2RGB)

    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    classnames      = read_class_names(classnames_filename)
    pb_file         = pb_filename
    image_path      = src_left_img
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
    # image = draw_bbox(disparity_map_rgb, bboxes, classnames)

    xmin = bboxes[0][0].astype(int)
    ymin = bboxes[0][1].astype(int)
    xmax = bboxes[0][2].astype(int)
    ymax = bboxes[0][3].astype(int)
    cropped_image = disparity_map_rgb[ymin:ymax, xmin:xmax]

    cv2.imwrite(dst_img, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
    # image = Image.fromarray(image)
    # image.show()
