import logging
import random
import colorsys

import numpy as np
import cv2

logger = logging.getLogger(__name__)


def draw_bbox(image, bboxes, classes, show_label=True):
    """Draw Bounding Boxes

    Parameters
    ----------
    bboxes : array
        For example, [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 0.25, 1.0) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    alpha_bbox = 0.25
    alpha_label = 0.35

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.85
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])

        overlay = image.copy()

        cv2.rectangle(overlay, c1, c2, bbox_color, bbox_thick)
        cv2.addWeighted(overlay, alpha_bbox, image, 1 - alpha_bbox, 0, image)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(overlay, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(overlay, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

            cv2.addWeighted(overlay, alpha_label, image, 1 - alpha_label, 0, image)

    return image
