# SimpSOM (Simple Self-Organizing Maps)

## Version 2.0.1

SimpSOM is a lightweight implementation of Kohonen Self-Organizing Maps
(SOM) for Python 3, useful for unsupervised learning,
clustering and dimensionality reduction.

To install this package, clone this repository and install it with
`python setup.py install`. Alternatively you can download it from PyPI,
to retrieve it just type `pip install simpsom`, but make sure the
version you need is available on the database.

It allows you to build and train SOM on your dataset, save/load the trained
network weights, and display or print graphs of the network with
selected features. The function `run_colors_example()` will run a toy
model, where a number of colors will be mapped from the 3D RGB space to
the 2D network map and clustered according to their similarity in the
origin space.

## What\'s New

- Class and function names have been updated to adhere to PEP8.
- Batch training has been added and is now the default algorithm.
- A light parallelization is now possible with RAPIDS.

## Version compatibility

This version introduces a number of changes, while attempting to maintain
the original philosophy of this project: a SOM library easy to understand and customize.
Functions and classes names have been changed to improve readability.
If you are migrating from an older version (<=1.3.4), please make sure to check the API first!

