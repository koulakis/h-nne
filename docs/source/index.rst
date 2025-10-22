.. h-NNE documentation master file, created by
   sphinx-quickstart on Tue Mar 15 16:51:32 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==============================================
h-NNE: Hierarchical Nearest Neighbor Embedding
==============================================

Great news, we have a new version of the algorithm 🥳🥳🥳


h-NNE v2
--------

🎉 **New in h-NNE v2:** a fast, geometry-aware **packing layer** for the top (coarse) levels that allocates space proportional to cluster size and preserves local anchor relations. This fixes the “dotifying” look (tiny, collapsed clusters) and the “squeezed big cluster” issue that could appear in v1 when many anchors sat too close.

What changed & why
~~~~~~~~~~~~~~~~~~

- **Space by cluster mass.** Coarse-level anchors are laid out via a lightweight circle-packing step guided by PCA anchors and :math:`1`-NN relations to produce acompact, overlap-free arrangement. Larger clusters get more area; small ones remain visible.
- **Better global framing.** A vectorized overlap resolver plus relaxed/capped :math:`k`-NN edge targets yields tight layouts without blow-ups, reducing unused whitespace and artifacts.
- **Drop-in for h-NNE.** v2 runs on the top few FINCH_ levels, then hands the result to the standard h-NNE refinement. Deeper (fine) levels still use the original fast point-to-anchor updates.

- **Note:** When running v2 on large datasets (>= 1M points), it starts the tree layout by default after some level of FINCH with a minimum number of clusters (10-10,000) to enhance point spread.

Using it
~~~~~~~~

- v2 is **on by default**. To use the legacy pipeline:

  .. code-block:: python

     HNNE(..., hnne_version="v1")

- Defaults generally work well. Optional advanced knobs (e.g., ``start_cluster_view``) let you **steer the look**: choose a **deeper level** for a more uniform, zoomed-in spread, or a **coarser (top) level** for a global, zoomed-out view before refinement.

More technical details (algorithms, complexity, limitations) are in our arXiv paper (linked soon).


h-NNE (overview)
----------------

h-NNE is a hierarchical dimensionality reduction method—akin in spirit to t-SNE/UMAP—but designed for speed, simplicity, and structured views. It first builds a clustering hierarchy with FINCH_, then embeds cluster anchors level-by-level and maps points to their anchors with simple, vectorized updates. This preserves local neighborhoods while providing a natural coarse→fine “zoom” over the data.

Key properties
~~~~~~~~~~~~~~

- **Fast & scalable.** No global nonconvex optimization; point updates are vectorized and memory-friendly.
- **Structure-aware zoom.** Choose a parent level for global context, then expand children for detail.
- **Labels included.** FINCH provides cluster labels out of the box—useful for unlabeled data and structured visualization.
- **Parameter-light.** FINCH is parameter-free; h-NNE uses robust defaults.

See our `corresponding CVPR 2022 paper`__ for the original algorithm and the arXiv addendum for the v2 packing layer.


  M. Saquib Sarfraz\*, Marios Koulakis\*, Constantin Seibold, Rainer Stiefelhagen.
  *Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction*. CVPR 2022.

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
