import logging

import yaml

logger = logging.getLogger(__name__)


def read_class_names(filename):
    """Load class name from a file.
    """

    names = {}
    with open(filename, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
