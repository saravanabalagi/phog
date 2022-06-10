# Pyramid Histogram of Gradients (PHOG)

[![language](https://img.shields.io/badge/language-python-blue.svg)]()
[![language](https://img.shields.io/badge/language-c++-blue.svg)]()
[![The MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENCE)

PHOG descriptor implemented in both C++ and Python. An example [image](data/image_sample.jpg) from City Centre Dataset is provided in the [data](data) folder along with the reference [descriptor](data/image_sample.jpg) for testing purposes.

Python Project dependencies (Poetry):
- numpy
- opencv-python

C++ Project dependencies (CMake):
- [OpenCV](https://opencv.org/)
- [cnpy](https://github.com/saravanabalagi/cnpy) (for numpy ops)
- [zlib](https://www.zlib.net/) (for npz compression)
- [boost](https://www.boost.org/) (for reading files)
- [OpenMP](https://www.openmp.org/) (for multiprocessing)

As can be seen in the results, the implementation produces descriptors that are accurate upto 3 decimals. 
