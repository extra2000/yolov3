#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""Entry-point for yolov3 executable.
"""

import os
import sys
import logging
import logging.handlers
import argparse
from pprint import PrettyPrinter
from functools import partial

import argcomplete
import tensorflow as tf

from yolov3 import YOLOv3, __version__
from yolov3 import YOLOTrain
from common.load_config import load_config
from common.import_dataset import import_dataset
from common.convert_weights import convert_weights
from common.freeze_graph import freeze_graph
from common.detect_image import detect_image
from common.detect_stereo_image import detect_stereo_image
from common.detect_video import detect_video

logger = logging.getLogger(__name__)
pp = PrettyPrinter(indent=2, width=80, compact=True)
tf.compat.v1.disable_eager_execution()


def main():
    try:
        _app()
    except KeyboardInterrupt as e:
        logger.exception('Received keyboard interrupt. Exiting.', exc_info=True)


def _app():
    args, parser = _parse_args()
    _init_logger(args.loglevel)

    command = {
        'import-dataset': partial(_import_dataset, args),
        'convert-weights': partial(_convert_weights, args),
        'freeze-model': partial(_freeze_model, args),
        'detect-image': partial(_detect_image, args),
        'detect-stereo-image': partial(_detect_stereo_image, args),
        'detect-video': partial(_detect_video, args),
        'train': partial(_train, args),
        'evaluate': partial(_evaluate, args)
    }

    choice = command.get(args.command, lambda: parser.print_help())
    result = choice()

    if result is not None:
        if isinstance(result, dict):
            print(result if args.no_pprint else pp.pformat(result))
        else:
            print(result)


def _init_logger(loglevel='WARNING'):
    """Initialize logger.

    Parameters
    ----------
    loglevel : str
        Log level for ``logging`` module. Default is ``'WARNING'``.
    """

    formatter = logging.Formatter('%(asctime)s %(process)d:%(levelname)s:%(name)s:%(lineno)d: %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(loglevel)

    yolov3logger = logging.getLogger('yolov3')
    yolov3logger.addHandler(stream_handler)
    yolov3logger.setLevel(loglevel)

    commonlogger = logging.getLogger('common')
    commonlogger.addHandler(stream_handler)
    commonlogger.setLevel(loglevel)


def _parse_args():
    parser = argparse.ArgumentParser(
        description='YOLOv3 suite {}'.format(__version__),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--loglevel', help='Log level', default='WARNING')
    parser.add_argument('--config', help='Config file')

    subparser = parser.add_subparsers(help='Command', dest='command')

    import_dataset = subparser.add_parser('import-dataset', help='Import dataset')
    import_dataset_args = import_dataset.add_argument_group('required arguments')
    import_dataset_args.add_argument('--from-format', type=str, choices=['voc', 'labelme'], help='Dataset format')
    import_dataset_args.add_argument('--dataset-rootdir', type=str, help='Dataset rootdir')
    import_dataset_args.add_argument('--train-annot', type=str, help='Training annotation .txt file')
    import_dataset_args.add_argument('--test-annot', type=str, help='Testing annotation .txt file')

    convert_weights = subparser.add_parser('convert-weights', help='Convert weights')
    convert_weights_args = convert_weights.add_argument_group('required arguments')
    convert_weights_args.add_argument(
        '--target',
        type=str,
        choices=['training', 'inference'],
        help='Type of conversion to be done, whether for training or for inference (deployment)',
        required=True)
    convert_weights_args.add_argument(
        '--classnames',
        type=str,
        help='A CSV file containing a list of class names',
        required=True)
    convert_weights_args.add_argument('input-file', type=str, help='Source filename')
    convert_weights_args.add_argument('output-file', type=str, help='Destination filename')

    freeze_model = subparser.add_parser('freeze-model', help='Freeze model for deployment')
    freeze_model_args = freeze_model.add_argument_group('required arguments')
    freeze_model_args.add_argument(
        '--classnames',
        type=str,
        help='A CSV file containing a list of class names',
        required=True)
    freeze_model_args.add_argument('chkpt-filename', help='Input filename')
    freeze_model_args.add_argument('pb-filename', help='Output filename')

    detect = subparser.add_parser('detect-image', help='Perform object detection on a single image')
    detect_args = detect.add_argument_group('required arguments')
    detect_args.add_argument('--model', help='The frozen model in PB file format')
    detect_args.add_argument(
        '--classnames',
        type=str,
        help='A CSV file containing a list of class names',
        required=True)
    detect_args.add_argument('input-filename', help='Input filename')
    detect_args.add_argument('output-filename', help='Output filename')

    detect = subparser.add_parser('detect-stereo-image', help='Perform object detection on a single stereo image')
    detect_args = detect.add_argument_group('required arguments')
    detect_args.add_argument('--model', help='The frozen model in PB file format')
    detect_args.add_argument(
        '--classnames',
        type=str,
        help='A CSV file containing a list of class names',
        required=True)
    detect_args.add_argument('input-left-filename', help='Left input image filename')
    detect_args.add_argument('input-right-filename', help='Left right image filename')
    detect_args.add_argument('output-filename', help='Output filename')

    detect = subparser.add_parser('detect-video', help='Perform object detection on a video')
    detect_args = detect.add_argument_group('required arguments')
    detect_args.add_argument('--model', help='The frozen model in PB file format')
    detect_args.add_argument(
        '--classnames',
        type=str,
        help='A CSV file containing a list of class names',
        required=True)
    detect_args.add_argument('input-filename', help='Input filename')
    detect_args.add_argument('output-pattern', help='Output pattern')

    train = subparser.add_parser('train', help='Train model')
    train_args = train.add_argument_group('required arguments')
    train_args.add_argument(
        '--initial-weight',
        type=str,
        help='Initial weight filename',
        required=True)
    train_args.add_argument(
        '--train-log-dir',
        type=str,
        help='Directory to store training logs'
    )
    train_args.add_argument(
        '--output-weight-dir',
        type=str,
        help='Directory to store trained weights'
    )

    evaluate = subparser.add_parser('evaluate', help='Evaluate model')
    evaluate_args = evaluate.add_argument_group('required arguments')
    evaluate_args.add_argument(
        '--enable-augmentation',
        action='store_true',
        help='Enable audmentation on testing sets')

    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    return args, parser


def _import_dataset(args):
    import_dataset(
        vars(args)['from_format'],
        vars(args)['dataset_rootdir'],
        vars(args)['train_annot'],
        vars(args)['test_annot']
    )


def _convert_weights(args):
    config = load_config(args.config)
    convert_weights(
        config,
        vars(args)['input-file'],
        vars(args)['output-file'],
        vars(args)['target'],
        vars(args)['classnames']
    )


def _freeze_model(args):
    config = load_config(args.config)
    freeze_graph(
        config,
        vars(args)['chkpt-filename'],
        vars(args)['pb-filename'],
        vars(args)['classnames']
    )


def _detect_image(args):
    detect_image(
        vars(args)['model'],
        vars(args)['classnames'],
        vars(args)['input-filename'],
        vars(args)['output-filename']
    )


def _detect_stereo_image(args):
    detect_stereo_image(
        vars(args)['model'],
        vars(args)['classnames'],
        vars(args)['input-left-filename'],
        vars(args)['input-right-filename'],
        vars(args)['output-filename']
    )


def _detect_video(args):
    detect_video(
        vars(args)['model'],
        vars(args)['classnames'],
        vars(args)['input-filename'],
        vars(args)['output-pattern']
    )


def _train(args):
    config = load_config(args.config)
    yolotrain = YOLOTrain(
        config,
        vars(args)['initial_weight'],
        vars(args)['train_log_dir'],
        vars(args)['output_weight_dir']
    )
    yolotrain.train()


def _evaluate(args):
    raise NotImplementedError
