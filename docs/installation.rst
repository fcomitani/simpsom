============
Installation
============

`simpsom` can be easily installed through python standard
package managers without worrying about dependencies.

With pip

.. code-block:: bash

   pip install simpsom

To install the latest (unreleased) version
you can download it from our GitHub repository by running

.. code-block:: bash

   git clone https://github.com/fcomitani/simpsom
   cd simpsom
   python setup.py install

The current version requires the following
core packages and their inherited dependencies:

   - numpy
   - scikit-learn
   - matplotlib

If available, `CuPy` can be used to run `simpsom` on the GPU.
`CuML` is also optional, but will allow you 
to run clustering on the GPU as well.

For a full list see :code:`requirements.txt`
