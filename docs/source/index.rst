.. h-NNE documentation master file, created by
   sphinx-quickstart on Tue Mar 15 16:51:32 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==============================================
h-NNE: Hierarchical Nearest Neighbor Embedding
==============================================

--------
h-NNE v2
--------

Great news, we have a new version of the algorithm 🥳🥳🥳

Some of its features:

- It allocates space proprotional to cluster sizes which helps avoid cluster collapses.
- In general it places points in a more space-efficient way using a form of circle packing.
- It improves some of the original h-NNE mechanics by using robust statistics instead of maximage.

What to pay attention to:

- Now this is the default version used. If you wish to use h-NNE v1, set `hnne_version="v1"` when initializing `HNNE`.
- It is still very new and has not been published in peer-reviewed journal or conference.
- In contrast to h-NNE v1, it does not respect the FINCH clusters on the top levels where the circle packing occurs. Instead it picks the optimal layout based on an original PCA projection and the packing.

--------
h-NNE v1
--------

A fast hierarchical dimensionality reduction algorithm.

h-NNE is a general purpose dimensionality reduction algorithm such as t-SNE and UMAP. It stands out for its speed,
simplicity and the fact that it provides a hierarchy of clusterings as part of its projection process. The algorithm is
inspired by the FINCH_ clustering algorithm. For more information on the structure of the algorithm, please look at `our
corresponding paper in CVPR 2022`__:

  M. Saquib Sarfraz\*, Marios Koulakis\*, Constantin Seibold, Rainer Stiefelhagen.
  Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction. CVPR 2022.

.. __: https://openaccess.thecvf.com/content/CVPR2022/papers/Sarfraz_Hierarchical_Nearest_Neighbor_Graph_Embedding_for_Efficient_Dimensionality_Reduction_CVPR_2022_paper.pdf

.. _FINCH: https://github.com/ssarfraz/FINCH-Clustering

Github repository: `https://github.com/koulakis/h-nne`__

.. __: https://github.com/koulakis/h-nne

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   guide/install
   guide/getting_started
   guide/projection_clusters
   guide/more_examples
   guide/contributions

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference/hnne
   api_reference/finch_clustering
   api_reference/cool_functions

--------
Citation
--------
If you make use of this project in your work, it would be appreciated if you cite the hnne paper:

.. code-block:: bibtex

    @article{hnne,
      title = {Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction},
      author = {M. Saquib Sarfraz, Marios Koulakis, Constantin Seibold, Rainer Stiefelhagen},
      booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2022}
    }

If you make use of the clustering properties of the algorithm please also cite:

.. code-block:: bibtex

    @inproceedings{finch,
      author    = {M. Saquib Sarfraz and Vivek Sharma and Rainer Stiefelhagen},
      title     = {Efficient Parameter-free Clustering Using First Neighbor Relations},
      booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages = {8934--8943},
      year  = {2019}
   }

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`search`
* :ref:`modindex`
