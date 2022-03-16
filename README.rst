=====
h-NNE: Hierarchical Nearest Neighbor Embedding
=====
A fast hierarchical dimensionality reduction algorithm.

h-NNE is a general purpose dimensionality reduction algorithm such as t-SNE and UMAP. It stands out for its speed,
simplicity and the fact that it provides a hierarchy of clusterings as part of its projection process. The algorithm is
inspired by the FINCH_ clustering algorithm. For more information on the structure of the algorithm, please look at the
corresponding paper:

  M. Saquib Sarfraz\*, Marios Koulakis\*, Constantin Seibold, Rainer Stiefelhagen.
  `Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction.`__

\*Paper link will be added soon

.. _FINCH: https://github.com/ssarfraz/FINCH-Clustering
.. __: https://github.com/koulakis/h-nne

------------
Installation
------------
The project is available in PyPI. To install run:

``pip install hnne``

----------------
How to use h-NNE
----------------
The HNNE class implements the common methods of the sklearn interface.

+++++++++++++++++++++++++
Simple projection example
+++++++++++++++++++++++++

.. code:: python

  import numpy as np
  from hnne import HNNE

  data = np.random.random(size=(1000, 20))

  hnne = HNNE()
  projection = projector.fit_transform(data)

++++++++++++++++++++++++++++
Using the hierarchy clusters
++++++++++++++++++++++++++++

.. code:: python

  hnne = HNNE()
  projection = hnne.fit_transform(data)

  level = 0
  clusters = hnne.hierarchy_parameters.partitions[:, level]

++++++++++++++++++++++++++++
Projecting on new points
++++++++++++++++++++++++++++

.. code:: python

  hnne = HNNE()
  projection = hnne.fit_transform(data)

  new_data_projection = hnne.transform(new_data)

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

--------
Citation
--------
If you make use of this project in your work, it would be appreciated if you cite the hnne paper:

.. code:: bibtex

    @article{hnne-projection,
        title={Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction},
        author={M. Saquib Sarfraz, Marios Koulakis, Constantin Seibold, Rainer Stiefelhagen}
    }

If you make use of the clustering properties of the algorithm please also cite:

.. code:: bibtex

    @inproceedings{finch-clustering,
        author = {Sarfraz, M. and Sharma, Vivek and Stiefelhagen, Rainer},
        year = {2019},
        month = {03},
        pages = {},
        title = {Efficient Parameter-Free Clustering Using First Neighbor Relations},
        doi = {10.1109/CVPR.2019.00914}
    }

----------
References
----------

[1] M. Saquib Sarfraz\*, Marios Koulakis\*, Constantin Seibold, Rainer Stiefelhagen.
  Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction.
  
[2] Sarfraz, Saquib and Sharma, Vivek and Stiefelhagen, Rainer. Efficient Parameter-Free Clustering
    Using First Neighbor Relations. Proceedings of the IEEE/CVF Conference on Computer Vision and
    Pattern Recognition (CVPR). June 2019.
