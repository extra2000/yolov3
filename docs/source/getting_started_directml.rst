Getting Started (DirectML)
==========================


Preparing an example project
----------------------------

Create project directory:

.. code-block:: bash

    cd ~\Documents
    mkdir yolov3-example

Then, ``cd`` into ``yolov3-example``:

.. code-block:: bash

    cd yolov3-example

Create ``config.yaml``:

.. code-block:: yaml

    yolov3:
      moving_ave_decay: 0.9995
      strides: [8, 16, 32]
      anchor_per_scale: 3
      anchors:
        baseline: [1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875]
      iou_loss_thresh: 0.5
      upsample_method: resize
      classnames: C:\Users\USERNAME\Documents\yolov3-example\voc.names
      training:
        annot_path: E:\datasets\pascal-voc\voc_train.txt
        batch_size: 8
        input_size: [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
        data_aug: true
        learn_rate_init: 1e-4
        learn_rate_end: 1e-6
        warmup_epochs: 2
        first_stage_epochs: 20
        second_stage_epochs: 30
      testing:
        annot_path: E:\datasets\pascal-voc\voc_test.txt
        batch_size: 8
        input_size: 544
        data_aug: false
        score_threshold: 0.3
        iou_threshold: 0.45


Clone our YOLOv3 repository
---------------------------

.. code-block:: bash

    git clone https://github.com/extra2000/yolov3.git


Prepare conda environment for TensorFlow DirectML
-------------------------------------------------

Prepare conda environment:

.. code-block:: bash

    conda create --name extra2000-yolov3-tf-directml
    conda activate extra2000-yolov3-tf-directml
    conda install python=3.7
    conda install cython
    python -m pip install tensorflow-directml==1.15.5 argcomplete

``cd`` into the cloned YOLOv3 repository:

.. code-block:: bash

    cd ~\Documents\yolov3-example\yolov3

Then, install required packages:

.. code-block:: bash

    python -m pip install .


Prepare conda environment for TensorFlow CPU
--------------------------------------------

TensorFlow CPU can be useful for testing and evaluation without affecting training performance.

Create conda environment:

.. code-block:: bash

    conda create --name extra2000-yolov3-tf-cpu
    conda activate extra2000-yolov3-tf-cpu
    conda install python=3.6
    conda install cython
    python -m pip install tensorflow==1.15.5 argcomplete

``cd`` into the cloned YOLOv3 repository:

.. code-block:: bash

    cd ~\Documents\yolov3-example\yolov3

Then, install required packages:

.. code-block:: bash

    python -m pip install .


Prepare YOLOv3 pretrained weight from COCO dataset
--------------------------------------------------

``cd`` into the project root directory:

.. code-block:: bash

    cd ~\Documents\yolov3-example

Get the pretrained weight and extract:

.. code-block:: bash

    wsl --exec wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
    wsl --exec mkdir -pv coco-pretrained-weight-original
    wsl --exec tar -xvf yolov3_coco.tar.gz --directory coco-pretrained-weight-original
    wsl --exec rm yolov3_coco.tar.gz

Create COCO classnames file ``coco.names``:

.. code-block:: text

    person
    bicycle
    car
    motorbike
    aeroplane
    bus
    train
    truck
    boat
    traffic light
    fire hydrant
    stop sign
    parking meter
    bench
    bird
    cat
    dog
    horse
    sheep
    cow
    elephant
    bear
    zebra
    giraffe
    backpack
    umbrella
    handbag
    tie
    suitcase
    frisbee
    skis
    snowboard
    sports ball
    kite
    baseball bat
    baseball glove
    skateboard
    surfboard
    tennis racket
    bottle
    wine glass
    cup
    fork
    knife
    spoon
    bowl
    banana
    apple
    sandwich
    orange
    broccoli
    carrot
    hot dog
    pizza
    donut
    cake
    chair
    sofa
    pottedplant
    bed
    diningtable
    toilet
    tvmonitor
    laptop
    mouse
    remote
    keyboard
    cell phone
    microwave
    oven
    toaster
    sink
    refrigerator
    book
    clock
    vase
    scissors
    teddy bear
    hair drier
    toothbrush

Convert the original weight into TensorFlow weight for inference:

.. code-block:: bash

    wsl --exec mkdir -pv coco-pretrained-weight-tf-inference
    yolov3 --loglevel=DEBUG --config=.\config.yaml convert-weights --target=inference --classnames=.\coco.names .\coco-pretrained-weight-original\yolov3_coco.ckpt .\coco-pretrained-weight-tf-inference\yolov3_coco.ckpt

Freeze the model into PB file:

.. code-block:: bash

    wsl --exec mkdir -pv coco-pretrained-weight-freeze
    yolov3 --loglevel=DEBUG --config=.\config.yaml freeze-model --classnames=.\coco.names .\coco-pretrained-weight-tf-inference\yolov3_coco.ckpt .\coco-pretrained-weight-freeze\yolov3_coco.pb


Test detection on an example image
----------------------------------

Download ``female.tiff`` image from `SIPI Database`_:

.. _SIPI Database: http://sipi.usc.edu/database/database.php?volume=misc&image=13#top

.. code-block:: bash

    wsl --exec wget "http://sipi.usc.edu/database/download.php?vol=misc&img=4.1.04" -O female.tiff

Create an empty directory ``results`` to store detection output:

.. code-block:: bash

    wsl --exec mkdir -pv results

Test detection:

.. code-block:: bash
    
    yolov3 --loglevel=DEBUG detect-image --classnames=.\coco.names --model=.\coco-pretrained-weight-freeze\yolov3_coco.pb .\female.tiff .\results\female.tiff


Training VOC dataset
--------------------

Prepare and empty directory to store datasets, for example:

.. code-block:: bash

    wsl --exec mkdir -pv /mnt/e/datasets/pascal-voc

``cd`` into the dataset directory:

.. code-block:: bash

    cd E:\datasets\pascal-voc

Download VOC 2007 and 2012 datasets:

.. code-block:: bash

    wsl --exec wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wsl --exec wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    wsl --exec wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

Extract datasets:

.. code-block:: bash

    wsl --exec mkdir -pv train test
    wsl --exec tar -xf VOCtest_06-Nov-2007.tar --directory=test/
    wsl --exec tar -xf VOCtrainval_06-Nov-2007.tar --directory=train/
    wsl --exec tar -xf VOCtrainval_11-May-2012.tar --directory=train/

Import dataset:

.. code-block:: bash

    yolov3 --loglevel DEBUG import-dataset --from-format=voc --dataset-rootdir=E:\datasets\pascal-voc --train-annot=E:\datasets\pascal-voc\voc_train.txt --test-annot=E:\datasets\pascal-voc\voc_test.txt

Create ``voc.names`` file:

.. code-block:: bash

    aeroplane
    bicycle
    bird
    boat
    bottle
    bus
    car
    cat
    chair
    cow
    diningtable
    dog
    horse
    motorbike
    person
    pottedplant
    sheep
    sofa
    train
    tvmonitor

Convert COCO pretrained weight for VOC training:

.. code-block:: bash

    wsl --exec mkdir -pv coco-pretrained-weight-tf-training-voc
    yolov3 --loglevel=DEBUG --config=.\config.yaml convert-weights --target=training --classnames=.\voc.names .\coco-pretrained-weight-original\yolov3_coco.ckpt .\coco-pretrained-weight-tf-training-voc\yolov3_coco.ckpt

Create an empty directory to store trained weights and logs:

.. code-block:: bash

    wsl --exec mkdir -pv /mnt/e/training-output

Begin training:

.. code-block:: bash

    yolov3 --loglevel=DEBUG --config=.\config.yaml train --initial-weight=.\coco-pretrained-weight-tf-training-voc\yolov3_coco.ckpt --train-log-dir=E:\training-output-01\logs --output-weight-dir=E:\training-output-01\checkpoints

To resume training, simply use any checkpoint as the initial weight. For example:

.. code-block:: bash

    yolov3 --loglevel=DEBUG --config=.\config.yaml train --initial-weight=E:\training-output-01\checkpoints\yolov3_test_loss=17.5392.ckpt-3 --train-log-dir=E:\training-output-02\logs --output-weight-dir=E:\training-output-02\checkpoints


Using trained weights for production
------------------------------------

Create an empty directory to store frozen checkpoints:

.. code-block:: bash

    wsl --exec mkdir -pv /mnt/e/training-output/checkpoints-freeze

Freeze the trained weight:

.. code-block:: bash

    yolov3 --loglevel=DEBUG --config=.\config.yaml freeze-model --classnames=.\voc.names E:\training-output-01\checkpoints\yolov3_test_loss=17.5392.ckpt-3 E:\training-output-01\checkpoints-freeze\yolov3_test_loss=17.5392.pb

Finally, test the detection:

.. code-block:: bash
    
    yolov3 --loglevel=DEBUG detect-image --classnames=.\voc.names --model=E:\training-output-01\checkpoints-freeze\yolov3_test_loss=17.5392.pb .\female.tiff .\results\female.tiff
