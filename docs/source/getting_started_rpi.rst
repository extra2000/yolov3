Getting Started (Raspberry Pi)
==============================


Install Miniconda
-----------------

Download Miniconda:

.. code-block:: bash

    cd ~/Downloads
    wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-aarch64.sh

Install Miniconda:

.. code-block:: bash

    bash Miniconda3-py37_4.9.2-Linux-aarch64.sh -b
    rm Miniconda3-py37_4.9.2-Linux-aarch64.sh

Initialize conda for BASH and disable auto activate ``(base)``:

.. code-block:: bash

    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda config --set auto_activate_base false

Exit and re-login SSH for the ``conda init bash`` command to take effect.


Prepare Python 3.6 and install TensorFlow
-----------------------------------------

Create conda environment ``extra2000-yolov3-tf-cpu`` with Python 3.7 and activate it:

.. code-block:: bash

    conda create --name extra2000-yolov3-tf-cpu python=3.7
    conda activate extra2000-yolov3-tf-cpu

While the ``extra2000-yolov3-tf-cpu`` environment activated, Install TensorFlow:

.. code-block:: bash

    python -m pip install tensorflow==1.15.5


Preparing an example project
----------------------------

Create project directory:

.. code-block:: bash

    cd ~/Documents
    mkdir yolov3-example

Then, ``cd`` into ``yolov3-example``:

.. code-block:: bash

    cd yolov3-example


Clone our YOLOv3 repository
---------------------------

.. code-block:: bash

    git clone https://github.com/extra2000/yolov3.git


Upload frozen model and classnames into Rasppberry Pi
-----------------------------------------------------

From your PC or laptop, upload the following files into Raspberry Pi:

.. code-block:: bash

    scp -r -P 22 coco-pretrained-weight-freeze ubuntu@rpicam.lan:Documents/yolov3-example/
    scp -P 22 coco.names ubuntu@rpicam.lan:Documents/yolov3-example/

Test detection on an example image
----------------------------------

Download ``female.tiff`` image from `SIPI Database`_:

.. _SIPI Database: http://sipi.usc.edu/database/database.php?volume=misc&image=13#top

.. code-block:: bash

    wget "http://sipi.usc.edu/database/download.php?vol=misc&img=4.1.04" -O female.tiff

Create an empty directory ``results`` to store detection output:

.. code-block:: bash

    mkdir -pv results
    chcon -R -v -t container_file_t results

Test detection:

.. code-block:: bash

    podman run --rm -it -v ./coco.names:/opt/coco.names:ro -v ./coco-pretrained-weight-freeze:/opt/model:ro -v ./female.tiff:/opt/female.tiff:ro -v ./results:/opt/results:rw localhost/extra2000/yolov3-tf-rocm yolov3 --loglevel=DEBUG detect-image --classnames=/opt/coco.names --model=/opt/model/yolov3_coco.pb /opt/female.tiff /opt/results/female.tiff
