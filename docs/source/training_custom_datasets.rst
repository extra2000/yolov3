Training Custom Datasets
========================


Prerequisites
-------------

``cd`` into your dataset root directory:

.. code-block:: bash

    cd /path/to/your/dataset-root-dir

Clone labelme repository and install:

.. code-block:: bash

    git clone --recursive --branch v4.5.13 https://github.com/wkentaro/labelme.git
    cd labelme
    python -m pip install .


Creating datasets
-----------------

``cd`` to your dataset rootdir:

.. code-block:: bash

    cd /path/to/your/dataset-root-dir

Create ``labels.txt``, for example:

.. code-block:: text

    person
    vehicle
    cat
    dog

Create ``flags.txt``, for example:

.. code-block:: text

    train
    test

Then, start ``labelme``:

.. code-block:: bash

    labelme --nodata --flags flags.txt --labels labels.txt --output annotations images

Import dataset:

.. code-block:: bash

    yolov3 --loglevel DEBUG import-dataset --from-format=labelme --dataset-rootdir=.\my-dataset --train-annot=.\my-dataset\train.txt --test-annot=.\my-dataset\test.txt


Convert COCO pretrained weight for custom dataset training
----------------------------------------------------------

.. code-block:: bash

    wsl --exec mkdir -pv coco-pretrained-weight-tf-training-custom
    yolov3 --loglevel=DEBUG --config=config.yaml convert-weights --target=training --classnames=.\my-dataset\labels.txt .\coco-pretrained-weight-original\yolov3_coco.ckpt .\coco-pretrained-weight-tf-training-custom\yolov3_coco.ckpt

Create an empty directory to store trained weights and logs:

.. code-block:: bash

    wsl --exec mkdir -pv /mnt/e/training-output-01

Begin training:

.. code-block:: bash

    yolov3 --loglevel=DEBUG --config=.\config.yaml train --initial-weight=.\coco-pretrained-weight-tf-training-custom\yolov3_coco.ckpt --train-log-dir=E:\training-output-01\logs --output-weight-dir=E:\training-output-01\checkpoints

To resume training, simply use any checkpoint as the initial weight. For example:

.. code-block:: bash

    wsl --exec mkdir -pv /mnt/e/training-output-02
    yolov3 --loglevel=DEBUG --config=.\config.yaml train --initial-weight=E:\training-output-01\checkpoints\yolov3_test_loss=17.5392.ckpt-3 --train-log-dir=E:\training-output-02\logs --output-weight-dir=E:\training-output-02\checkpoints


Using trained weights for production
------------------------------------

Create an empty directory to store frozen checkpoints:

.. code-block:: bash

    wsl --exec mkdir -pv /mnt/e/training-output-01/checkpoints-freeze

Freeze the trained weight:

.. code-block:: bash

    yolov3 --loglevel=DEBUG --config=.\config.yaml freeze-model --classnames=.\my-dataset\labels.txt E:\training-output-01\checkpoints\yolov3_test_loss=17.5392.ckpt-3 E:\training-output-01\checkpoints-freeze\yolov3_test_loss=17.5392.pb

Finally, test the detection:

.. code-block:: bash
    
    yolov3 --loglevel=DEBUG detect-image --classnames=.\my-dataset\labels.txt --model=E:\training-output-01\checkpoints-freeze\yolov3_test_loss=17.5392.pb .\test-img.tiff .\results\test-img.tiff
