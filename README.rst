.. image:: https://github.com/koulakis/h-nne/actions/workflows/actions.yml/badge.svg?branch=main
    :target: https://github.com/koulakis/h-nne/actions/workflows/actions.yml
    :alt: Build Status

.. image:: https://readthedocs.org/projects/hnne/badge/?version=latest
    :target: https://hnne.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://static.pepy.tech/badge/hnne
   :target: https://pepy.tech/projects/hnne
   :alt: PyPI Downloads


==============================================
h-NNE: Hierarchical Nearest Neighbor Embedding
==============================================

Great news, we have a new version of the algorithm 🥳🥳🥳


h-NNE v2
--------

🎉 **New in h-NNE v2:** a fast, geometry-aware **packing layer** for the top (coarse) levels that allocates space proportional to cluster size and preserves local anchor relations. This fixes the “dotifying” look (tiny, collapsed clusters) and the “squeezed big cluster” issue that could appear in v1 when many anchors sat too close.

What changed & why
~~~~~~~~~~~~~~~~~~

- **Space by cluster mass.** Coarse-level anchors are laid out via a lightweight circle-packing step guided by PCA anchors and :math:`1`-NN relations to produce a **compact, overlap-free** arrangement. Larger clusters get more area; small ones remain visible.
- **Better global framing.** A vectorized overlap resolver plus relaxed/capped :math:`k`-NN edge targets yields tight layouts without blow-ups, reducing unused whitespace and artifacts.
- **Drop-in for h-NNE.** v2 runs on the top few FINCH_ levels, then hands the result to the standard h-NNE refinement. Deeper (fine) levels still use the original fast point-to-anchor updates.

- **Note:** On FINCH structure at the top levels v2 respects cluster memberships and the hierarchy (children stay inside their parent), but does not freeze coarse anchors to their raw PCA positions. Instead, it finds an optimized packed layout that remains faithful to local PCA :math:`k`-NN structure and cluster sizes.

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

More details are available in the `project documentation`__.

.. __: https://hnne.readthedocs.io/en/latest/index.html


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

Below a dataset of dimensionality 256 is projected to 2 dimensions.

.. code:: python

  import numpy as np
  from hnne import HNNE

  data = np.random.random(size=(1000, 256))

  hnne = HNNE(n_components=2)
  projection = hnne.fit_transform(data)

++++++++++++++++++++++++++++
Projecting on new points
++++++++++++++++++++++++++++

Once a dataset has been projected, one can apply the transform and project new points to the same dimension.

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

.. _Basic Usage: notebooks/hnne_v2/demo1_basic_usage.ipynb
.. _Multiple Projections: notebooks/hnne_v2/demo2_multiple_projections.ipynb
.. _Clustering for Free: notebooks/hnne_v2/demo3_clustering_for_free.ipynb
.. _Monitor Quality of Network Embeddings: notebooks/hnne_v2/demo4_monitor_network_embeddings.ipynb

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

------------
Contributing
------------

Contributions are very welcome :-) Please check the `contributions guide`__ for more details.

.. __: docs/source/guide/contributions.rst
