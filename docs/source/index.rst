.. h-NNE documentation master file, created by
   sphinx-quickstart on Tue Mar 15 16:51:32 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====
h-NNE: Hierarchical Nearest Neighbor Embedding
=====
A fast hierarchical dimensionality reduction algorithm.

h-NNE is a general purpose dimensionality reduction algorithm such as t-SNE and UMAP. It stands out for its speed,
simplicity and the fact that it provides a hierarchy of clusterings as part of its projection process. The algorithm is
inspired by the FINCH_ clustering algorithm. For more information on the structure of the algorithm, please look at the
corresponding paper:

  M. Saquib Sarfraz\*, Marios Koulakis\*, Constantin Seibold, Rainer Stiefelhagen.
  Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction.

\*Paper link will be added soon

.. _FINCH: https://github.com/ssarfraz/FINCH-Clustering

Github repository: `https://github.com/koulakis/h-nne`__

.. __: https://github.com/koulakis/h-nne

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/install
   guide/basic_usage
   guide/multiple_projections
   guide/clustering_for_free
   guide/monitoring_class_disentanglement

--------
Citation
--------
If you make use of this project in your work, it would be appreciated if you cite the hnne paper:

.. code:: bibtex

    @article{hnne,
      title={Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction},
      author={M. Saquib Sarfraz, Marios Koulakis, Constantin Seibold, Rainer Stiefelhagen}
      booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2022}
    }

If you make use of the clustering properties of the algorithm please also cite:

.. code:: bibtex

    @inproceedings{finch,
      author    = {M. Saquib Sarfraz and Vivek Sharma and Rainer Stiefelhagen},
      title     = {Efficient Parameter-free Clustering Using First Neighbor Relations},
      booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages = {8934--8943}
      year  = {2019}
   }

----------
References
----------

[1] M. Saquib Sarfraz\*, Marios Koulakis\*, Constantin Seibold, Rainer Stiefelhagen.
  Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction.

[2] Sarfraz, Saquib and Sharma, Vivek and Stiefelhagen, Rainer. Efficient Parameter-Free Clustering
    Using First Neighbor Relations. Proceedings of the IEEE/CVF Conference on Computer Vision and
    Pattern Recognition (CVPR). June 2019.
