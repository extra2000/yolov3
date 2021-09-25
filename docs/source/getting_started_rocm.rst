Getting Started (ROCm)
======================


Preparing an example project
----------------------------

Create project directory:

.. code-block:: bash

    cd ~/Documents
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
      classnames: test/data/classes/voc.names
      training:
        annot_path: /opt/datasets/pascal-voc/voc_train.txt
        batch_size: 8
        input_size: [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
        data_aug: false
        learn_rate_init: 1e-4
        learn_rate_end: 1e-6
        warmup_epochs: 2
        first_stage_epochs: 20
        second_stage_epochs: 30
      testing:
        annot_path: /opt/datasets/pascal-voc/voc_test.txt
        batch_size: 8
        input_size: 544
        data_aug: false
        score_threshold: 0.3
        iou_threshold: 0.45


Create SELinux policy
---------------------

Create ``yolov3_rocm.cil`` file:

.. code-block:: text

    (block yolov3_rocm
        (blockinherit container)
        (allow process process ( capability ( chown dac_override fowner fsetid kill net_bind_service setfcap setgid setpcap setuid sys_chroot )))
    )

Import the SELinux policy:

.. code-block:: bash

    sudo semodule -i yolov3_rocm.cil /usr/share/udica/templates/base_container.cil


Clone our YOLOv3 repository
---------------------------

.. code-block:: bash

    git clone https://github.com/extra2000/yolov3.git

For SELinux platforms, label the project root directory as ``container_file_t``:

.. code-block:: bash

    chcon -R -v -t container_file_t yolov3

Build our YOLOv3 TensorFlow ROCm image
--------------------------------------

``cd`` into the cloned YOLOv3 repository:

.. code-block:: bash

    cd ~/Documents/yolov3-example/yolov3

Then, build the image:

.. code-block:: bash

    podman build -t extra2000/yolov3-tf-rocm -f Dockerfile.rocm .


Prepare YOLOv3 pretrained weight from COCO dataset
--------------------------------------------------

``cd`` into the project root directory:

.. code-block:: bash

    cd ~/Documents/yolov3-example

Get the pretrained weight and extract:

.. code-block:: bash

    wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
    mkdir -pv coco-pretrained-weight-original
    tar -xvf yolov3_coco.tar.gz --directory coco-pretrained-weight-original
    rm yolov3_coco.tar.gz

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

    mkdir -pv coco-pretrained-weight-tf-inference
    chcon -R -v -t container_file_t .
    podman run --rm -it --device=/dev/kfd --device=/dev/dri --security-opt label=type:yolov3_rocm.process -v ./config.yaml:/opt/config.yaml:ro -v ./coco-pretrained-weight-original:/opt/original-weight:ro -v ./coco.names:/opt/coco.names:ro -v ./coco-pretrained-weight-tf-inference:/opt/output-weight:rw localhost/extra2000/yolov3-tf-rocm yolov3 --loglevel=DEBUG --config=/opt/config.yaml convert-weights --target=inference --classnames=/opt/coco.names /opt/original-weight/yolov3_coco.ckpt /opt/output-weight/yolov3_coco.ckpt

Freeze the model into PB file:

.. code-block:: bash

    mkdir -pv coco-pretrained-weight-freeze
    chcon -R -v -t container_file_t .
    podman run --rm -it --device=/dev/kfd --device=/dev/dri --security-opt label=type:yolov3_rocm.process -v ./config.yaml:/opt/config.yaml:ro -v ./coco-pretrained-weight-tf-inference:/opt/weight:ro -v ./coco.names:/opt/coco.names:ro -v ./coco-pretrained-weight-freeze:/opt/output:rw localhost/extra2000/yolov3-tf-rocm yolov3 --loglevel=DEBUG --config=/opt/config.yaml freeze-model --classnames=/opt/coco.names /opt/weight/yolov3_coco.ckpt /opt/output/yolov3_coco.pb

Test detection on an example image
----------------------------------

Download ``female.tiff`` image from `SIPI Database`_:

.. _SIPI Database: http://sipi.usc.edu/database/database.php?volume=misc&image=13#top

.. code-block:: bash

    wget "http://sipi.usc.edu/database/download.php?vol=misc&img=4.1.04" -O female.tiff
    chcon -R -v -t container_file_t female.tiff

Create an empty directory ``results`` to store detection output:

.. code-block:: bash

    mkdir -pv results
    chcon -R -v -t container_file_t results

Test detection:

.. code-block:: bash

    podman run --rm -it --device=/dev/kfd --device=/dev/dri --security-opt label=type:yolov3_rocm.process -v ./coco.names:/opt/coco.names:ro -v ./coco-pretrained-weight-freeze:/opt/model:ro -v ./female.tiff:/opt/female.tiff:ro -v ./results:/opt/results:rw localhost/extra2000/yolov3-tf-rocm yolov3 --loglevel=DEBUG detect-image --classnames=/opt/coco.names --model=/opt/model/yolov3_coco.pb /opt/female.tiff /opt/results/female.tiff

Training VOC dataset
--------------------

Prepare and empty directory to store datasets, for example:

.. code-block:: bash

    sudo mkdir -pv /opt/datasets/pascal-voc
    sudo chown ${USER}:${USER} /opt/datasets/pascal-voc

``cd`` into the dataset directory:

.. code-block:: bash

    cd /opt/datasets/pascal-voc

Download VOC 2007 and 2012 datasets:

.. code-block:: bash

    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

Extract datasets:

.. code-block:: bash

    mkdir -pv train test
    tar -xf VOCtest_06-Nov-2007.tar --directory=test/
    tar -xf VOCtrainval_06-Nov-2007.tar --directory=train/
    tar -xf VOCtrainval_11-May-2012.tar --directory=train/

Grant access permissions to containers:

.. code-block:: bash

    chcon -R -v -t container_file_t /opt/datasets/pascal-voc

Import dataset:

.. code-block:: bash

    podman run --rm -it -v /opt/datasets/pascal-voc:/opt/datasets/pascal-voc:rw localhost/extra2000/yolov3-tf-rocm yolov3 --loglevel DEBUG import-dataset --from-format=voc --dataset-rootdir=/opt/datasets/pascal-voc --train-annot=/opt/datasets/pascal-voc/voc_train.txt --test-annot=/opt/datasets/pascal-voc/voc_test.txt

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

    cd ~/Documents/yolov3-example/yolov3
    mkdir -pv coco-pretrained-weight-tf-training-voc
    chcon -R -v -t container_file_t .
    podman run --rm -it --device=/dev/kfd --device=/dev/dri --security-opt label=type:yolov3_rocm.process -v ./config.yaml:/opt/config.yaml:ro -v ./coco-pretrained-weight-original:/opt/original-weight:ro -v ./voc.names:/opt/voc.names:ro -v ./coco-pretrained-weight-tf-training-voc:/opt/output-weight:rw localhost/extra2000/yolov3-tf-rocm yolov3 --loglevel=DEBUG --config=/opt/config.yaml convert-weights --target=training --classnames=/opt/voc.names /opt/original-weight/yolov3_coco.ckpt /opt/output-weight/yolov3_coco.ckpt

Create an empty directory to store trained weights and logs:

.. code-block:: bash

    mkdir -pv training-output-01
    chcon -R -v -t container_file_t training-output-01

Begin training:

.. code-block:: bash

    podman run --rm -it --device=/dev/kfd --device=/dev/dri --ipc=host --group-add video --cap-add=SYS_PTRACE --security-opt label=type:yolov3_rocm.process -v ./config.yaml:/opt/config.yaml:ro -v ./coco-pretrained-weight-tf-training-voc:/opt/initial-weight:ro -v ./voc.names:/opt/voc.names:ro -v /opt/datasets/pascal-voc:/opt/datasets/pascal-voc:ro -v ./training-output-01:/opt/training-output:rw localhost/extra2000/yolov3-tf-rocm yolov3 --loglevel=DEBUG --config=/opt/config.yaml train --initial-weight=/opt/initial-weight/yolov3_coco.ckpt --train-log-dir=/opt/training-output/logs --output-weight-dir=/opt/training-output/checkpoints
