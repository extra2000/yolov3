import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_anchors(anchors_path):
    """Loads the anchors from a file

    DEPRECATED: Unused.

    Parameters
    ----------
    anchors_path : str
        Filename containing anchors.
    """

    with open(anchors_path) as f:
        anchors = f.readline()

    anchors = np.array(anchors.split(','), dtype=np.float32)

    return anchors.reshape(3, 3, 2)
