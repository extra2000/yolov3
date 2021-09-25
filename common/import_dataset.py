import logging
import os
import json
import xml.etree.ElementTree as ET

import numpy as np

logger = logging.getLogger(__name__)


def import_dataset(from_format, dataset_rootdir, train_annot_file, test_annot_file):
    """Import Dataset

    Parameters
    ----------
    from_format : str
        Dataset format.
    dataset_rootdir : str
        Root directory of the dataset.
    train_annot_file : str
        Training annotation file.
    test_annot_file : str
        Testing annotation file.
    """

    if os.path.exists(train_annot_file):
        raise FileExistsError(train_annot_file)
    if os.path.exists(test_annot_file):
        raise FileExistsError(test_annot_file)

    if from_format == 'voc':
        _import_voc_dataset(
            os.path.join(dataset_rootdir, 'train', 'VOCdevkit', 'VOC2007'),
            'trainval',
            train_annot_file,
            False
        )
        _import_voc_dataset(
            os.path.join(dataset_rootdir, 'train', 'VOCdevkit', 'VOC2012'),
            'trainval',
            train_annot_file,
            False
        )
        _import_voc_dataset(
            os.path.join(dataset_rootdir, 'test', 'VOCdevkit', 'VOC2007'),
            'test',
            test_annot_file,
            False
        )
    elif from_format == 'labelme':
        _import_labelme_dataset(dataset_rootdir, 'train', train_annot_file)
        _import_labelme_dataset(dataset_rootdir, 'test', test_annot_file)
    else:
        raise NotImplementedError(from_format)


def _import_voc_dataset(data_path, data_type, anno_path, use_difficult_bbox=True):

    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            f.write(annotation + "\n")

    return len(image_inds)


def _import_labelme_dataset(dataset_rootdir, dataset_type, annot_txtfname):
    """Import Labelme dataset

    Parameters
    ----------
    dataset_rootdir : str
        Dataset rootdir.
    dataset_type : str
        Either 'train' or 'test'.
    annot_txtfname : str
        Text filename to write annotations.
    """

    annotdir = os.path.join(dataset_rootdir, 'annotations')

    with open(os.path.join(dataset_rootdir, 'labels.txt')) as labelfile:
        labels = labelfile.read().splitlines()

    with open(annot_txtfname, 'w') as annot_txtfile:
        for jsonfilename in os.listdir(annotdir):

            with open(os.path.join(annotdir, jsonfilename)) as jsonfile:
                annotdata = json.load(jsonfile)

            if annotdata['flags'][dataset_type] == True:
                imgpath = os.path.abspath(os.path.join(annotdir, annotdata['imagePath']))
                annotation = imgpath

                for shape in annotdata['shapes']:
                    label = shape['label']
                    label_idx = labels.index(label)
                    points = np.array(shape['points'])
                    xmin = str(np.int(np.round(np.min(points[:, 0]))))
                    xmax = str(np.int(np.round(np.max(points[:, 0]))))
                    ymin = str(np.int(np.round(np.min(points[:, 1]))))
                    ymax = str(np.int(np.round(np.max(points[:, 1]))))
                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(label_idx)])
                
                logger.debug(annotation)
                annot_txtfile.write(annotation + '\n')
