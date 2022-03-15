=====
h-NNE
=====
A fast hierarchical dimensionality reduction algorithm.

------------
Installation
------------
The project is available in PyPI. To install run:

`pip install hnne`

----------------
How to use h-NNE
----------------
The main class implements the main methods of the sklearn interface.

.. code-block:: python

  import numpy as np
  from hnne import HNNE

  data = np.random.rand()

  projector = HNNE()
  projection = projector.fit_transform(data)

-----
Demos
-----
The following demo notebooks are available:

1. `Basic Usage`_

2.  `Multiple Projections`_

3. `Clustering for Free`_

4. `Monitor Class Disentanglement`_

.. _Basic Usage: notebooks/demo1_basic_usage.ipynb
.. _Multiple Projections: notebooks/demo2_multiple_projections.ipynb
.. _Clustering for Free: notebooks/demo3_clustering_for_free.ipynb
.. _Monitor Class Disentanglement: notebooks/demo4_monitor_class_disentanglement.ipynb

----------
References
----------
If you make use of this project in your work, please cite the following references:

[1] M. Saquib Sarfraz\*, Marios Koulakis\*, Constantin Seibold, Rainer Stiefelhagen.
  Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction.
  
[2] Sarfraz, Saquib and Sharma, Vivek and Stiefelhagen, Rainer. Efficient Parameter-Free Clustering
    Using First Neighbor Relations. Proceedings of the IEEE/CVF Conference on Computer Vision and
    Pattern Recognition (CVPR). June 2019.
