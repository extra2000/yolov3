Cheatsheets
===========

This Chapter contains cheatsheets for ``yolov3`` command lines.

To execute ``yolov3`` commands using Podman, prefix Podman command for example:

.. code-block:: bash

  podman run -it --rm localhost/extra2000/yolov3 yolov3 [command] [arguments]

Or execute BASH in the container and then use ``yolov3`` commands:

.. code-block:: bash

  podman run -it --rm localhost/extra2000/yolov3 yolov3 bash
  yolov3 [command] [arguments]

For verbose log, use ``--loglevel DEBUG``. For example:

.. code-block:: bash

  yolov3 --loglevel DEBUG [command] [arguments]

List of available commands
--------------------------

.. program-output:: yolov3 --help

Import VOC dataset
------------------

.. program-output:: yolov3 import-dataset --help

Example command:

.. code-block:: bash

  yolov3 --loglevel DEBUG import-dataset --from-format=voc --dataset-rootdir=E:/dataset/VOC --train-annot=E:/dataset/voc_train.txt --test-annot=E:/dataset/voc_test.txt

Converting weights
------------------

.. program-output:: yolov3 convert-weights --help

Example command:

.. code-block:: bash

  yolov3 --loglevel=DEBUG --config=test/config.yaml convert-weights --target=training --classnames=./test/data/classes/voc.names ./test/checkpoint/yolov3_coco.ckpt ./test/checkpoint/yolov3_coco_demo.ckpt

  yolov3 --loglevel=DEBUG --config=test/config.yaml convert-weights --target=inference --classnames=./test/data/classes/coco.names ./test/checkpoint/yolov3_coco.ckpt ./test/checkpoint/yolov3_coco_demo.ckpt

This command will produce the following files:

* yolov3_coco_demo.ckpt.data-00000-of-00001
* yolov3_coco_demo.ckpt.index
* yolov3_coco_demo.ckpt.meta

Freezing model
--------------

.. program-output:: yolov3 freeze-model --help

Example command:

.. code-block:: bash

  yolov3 --loglevel=DEBUG --config=test/config.yaml freeze-model --classnames=./test/data/classes/coco.names ./test/checkpoint/yolov3_coco_demo.ckpt ./test/yolov3.pb

Object detection on a single image
----------------------------------

.. program-output:: yolov3 detect-image --help

Example command:

.. code-block:: bash

  yolov3 --loglevel=DEBUG detect-image --classnames=./test/data/classes/coco.names --model=./test/yolov3.pb ./test/road.jpeg ./test/road-output.jpg

Object detection on a video
---------------------------

.. program-output:: yolov3 detect-video --help

Example command:

.. code-block:: bash

  yolov3 --loglevel=DEBUG detect-video --classnames=./test/data/classes/coco.names --model=./test/yolov3.pb ./test/road.mp4 ./test/road-output/%04d.png

Training
--------

.. program-output:: yolov3 train --help

Example command:

  yolov3 --loglevel=DEBUG --config=test/config.yaml train --initial-weight=./test/checkpoint/yolov3_coco_demo.ckpt --train-log-dir=./test/trainlogs --output-weight-dir=E:/tmp/yolov3

  tensorboard --logdir ./test/trainlogs
