import os
from setuptools import setup, find_packages

def _get_version():
    version = {}
    with open(os.path.join('yolov3', 'version.py')) as fp:
        exec(fp.read(), version)
    return version['version']

def _setup_package():
    setup(
        name='yolov3',
        description='YOLOv3 suite.',
        version=_get_version(),
        author='extra2000',
        include_package_data=True,
        packages=[
            'yolov3',
            'yolov3.console',
            'common'
        ],
        entry_points={
            'console_scripts': [
                'yolov3=yolov3.console:main',
            ],
        },
        python_requires='>=3.6.0, <3.8.0',
        install_requires=[
            'PyYAML >=5.4.0, ==5.4.*',
            'pandas >=1.1.0, <1.4',
            'Pillow >=8.3.0, ==8.3.*',
            'scipy >=1.5.0, ==1.5.*',
            'seaborn >=0.11.0, ==0.11.*',
            'opencv-python-headless >=4.5.0, ==4.5.*',
            'tqdm >=4.62.0, ==4.62.*',
            'matplotlib >=3.3.0, ==3.3.*',
        ],
    )

if __name__ == '__main__':
    _setup_package()
