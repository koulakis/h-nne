==============================================
h-NNE: Hierarchical Nearest Neighbor Embedding
==============================================
A fast hierarchical dimensionality reduction algorithm.

h-NNE is a general purpose dimensionality reduction algorithm such as t-SNE and UMAP. It stands out for its speed,
simplicity and the fact that it provides a hierarchy of clusterings as part of its projection process. The algorithm is
inspired by the FINCH_ clustering algorithm. For more information on the structure of the algorithm, please look at the
corresponding paper:

  M. Saquib Sarfraz\*, Marios Koulakis\*, Constantin Seibold, Rainer Stiefelhagen.
  Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction. CVPR 2022.

*Paper link will be added soon*

.. _FINCH: https://github.com/ssarfraz/FINCH-Clustering

More details are available in the project documentation_.

.. _documentation: https://hnne.readthedocs.io/en/latest/index.html

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

  data = np.random.random(size=(1000, 256))

  hnne = HNNE(dim=2)
  projection = hnne.fit_transform(data)

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

2. `Multiple Projections`_

3. `Clustering for Free`_

4. `Monitor Quality of Network Embeddings`_

.. _Basic Usage: notebooks/demo1_basic_usage.ipynb
.. _Multiple Projections: notebooks/demo2_multiple_projections.ipynb
.. _Clustering for Free: notebooks/demo3_clustering_for_free.ipynb
.. _Monitor Quality of Network Embeddings: notebooks/demo4_monitor_network_embeddings.ipynb

--------
Citation
--------
If you make use of this project in your work, it would be appreciated if you cite the hnne paper:

.. code:: bibtex

    @article{hnne,
      title={Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction},
      author={M. Saquib Sarfraz, Marios Koulakis, Constantin Seibold, Rainer Stiefelhagen},
      booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2022}
    }

If you make use of the clustering properties of the algorithm please also cite:

.. code:: bibtex

    @inproceedings{finch,
      author    = {M. Saquib Sarfraz and Vivek Sharma and Rainer Stiefelhagen},
      title     = {Efficient Parameter-free Clustering Using First Neighbor Relations},
      booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages = {8934--8943},
      year  = {2019}
   }

