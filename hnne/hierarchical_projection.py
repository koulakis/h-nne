from enum import Enum

import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise
from sklearn.preprocessing import StandardScaler

from hnne.cool_functions import cool_max_radius, cool_mean
from hnne.point_spreading import norm_angles, norm_angles_3d
from hnne.v2_packer import pack_hierarchy_iterative_k_nd
from hnne.v2_utils import (
    HNNEVersion,
    _normalize_hnne_version,
    choose_v2_level_block,
    layout_to_level_arrays,
    partition_update,
    rubust_scale_per_parent,
)

try:
    from pynndescent import NNDescent

    _HAS_NND = True
except Exception:
    _HAS_NND = False


class PreliminaryEmbedding(str, Enum):
    pca = "pca"
    pca_centroids = "pca_centroids"
    random_linear = "random_linear"


def project_with_pca_centroids(
    data,
    partitions,
    partition_sizes,
    dim=2,
    min_number_of_anchors=1000,
    random_state=None,
    verbose=False,
):
    pca = PCA(n_components=dim, random_state=random_state)
    large_partitions = np.where(np.array(partition_sizes) > min_number_of_anchors)[0]
    partition_idx = large_partitions.max() if any(large_partitions) else 0
    if verbose:
        print(
            f"Projecting on the {partition_idx}th partition with {partition_sizes[partition_idx]} anchors."
        )
    selected_anchors = cool_mean(data, partitions[:, partition_idx])
    pca.fit(selected_anchors)
    return pca.transform(data), pca


def project_with_pca(data, dim=2, random_state=None):
    pca = PCA(n_components=dim, random_state=random_state)
    transformed_data = pca.fit_transform(data)
    return transformed_data, pca


def project_points(
    data,
    dim=2,
    preliminary_embedding="pca",
    partition_sizes=None,
    partitions=None,
    random_state=None,
    verbose=False,
):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    pca = None

    if preliminary_embedding == PreliminaryEmbedding.pca:
        projected_points, pca = project_with_pca(
            data, dim=dim, random_state=random_state
        )
    elif len(data) < dim or preliminary_embedding == PreliminaryEmbedding.pca_centroids:
        projected_points, pca = project_with_pca_centroids(
            data,
            partitions,
            partition_sizes,
            dim=dim,
            random_state=random_state,
            verbose=verbose,
        )
    elif preliminary_embedding == PreliminaryEmbedding.random_linear:
        np.random.seed(random_state)
        random_components = np.random.random((data.shape[1], dim))
        projected_points = np.dot(data, random_components)

    else:
        raise ValueError(f"Invalid preliminary embedding: {preliminary_embedding}")

    # TODO: Handle the case where pca is not defined (e.g. random projection)
    # TODO: Add an option to perform full (randomized) PCA
    return projected_points, pca, scaler


def get_finch_anchors(projected_points, partitions=None):
    all_projected_anchors = []
    for i in range(partitions.shape[-1]):
        projected_anchors = cool_mean(projected_points, partitions[:, i])
        all_projected_anchors.append(projected_anchors)
    return all_projected_anchors


def move_projected_points_to_anchors_v2(
    points: np.ndarray,
    anchors: np.ndarray,
    partition: np.ndarray,
    *,
    radius: float = 0.9,
    anchor_radii: np.ndarray | None = None,
    real_nn_threshold: int = 40000,  # kept for API compat (unused with batching)
    verbose: bool = False,
    # --- safe & robust knobs (sane defaults) ---
    k_radius: int = 2,  # base radius from median of k-NN distances
    clip_quantiles: tuple[float, float] | None = None,  # e.g. (0.02, 0.98); None = off
    cap_eps_frac: float = 1e-3,  # tiny epsilon in half-NN cap (relative to median d1)
    use_pairwise_small_k: bool = True,
    small_k_cutoff: int = 100,
    nn_batch_size: int = 200_000,  # chunk size for KDTree.query
    use_robust_scaling=True,
):
    """
    Map points into anchor-centered discs with vectorized scaling:
      mapped = anchors_per_point + r_anchor_per_point * (points_centered / points_max_radius)

    Radii logic (when anchor_radii=None):
      - Base radius from exact k-NN distances (median over k_radius neighbors),
      - Optional quantile clipping (cosmetic; never enlarges past cap),
      - Symmetric half-1NN cap to prevent cross-cluster mixing.


    Robust Scaling: Differences vs. original:
      - If `use_robust_scaling=True`, per-parent scale uses mean + λ·std of
        ||points - parent_mean|| instead of max, capped by the true max. This
        avoids a few outliers collapsing everyone else to the center.
      - Optional radial power warp `gamma<1` spreads typical points a bit to
        better fill large parents while keeping outliers bounded.

    Everything remains vectorized and O(n).
    """

    points = np.asarray(points, float)
    anchors = np.asarray(anchors, float)
    part = np.asarray(partition, int)

    N, d = points.shape
    K = anchors.shape[0]
    if K == 0:
        raise ValueError("anchors is empty")

    # ---------- compute per-anchor radii (if not supplied) ----------
    if anchor_radii is None:
        if K == 1:
            # No neighbors: set a sensible scale from the data itself
            c = np.median(points, axis=0, keepdims=True)
            rho = np.linalg.norm(points - c, axis=1)
            med = float(np.median(rho)) if rho.size else 1.0
            r0 = radius * (med if med > 0 else 1.0)
            anchor_radii = np.array([[r0]], dtype=float)

        else:
            # we need at least 2 non-self neighbors to be robust to duplicates
            kq = max(3, int(k_radius) + 1)  # neighbors requested incl. self

            if use_pairwise_small_k and K <= small_k_cutoff:
                # ----- small-K exact pairwise path -----
                D2 = ((anchors[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2)
                np.fill_diagonal(D2, np.inf)

                kth = min(kq - 1, K - 1)
                idx = np.argpartition(D2, kth=kth, axis=1)[:, :kth]  # (K, <=kq-1)
                rows = np.arange(K)[:, None]
                Dsmall = D2[rows, idx]  # (K, <=kq-1)

                # sort those columns
                order_small = np.argsort(Dsmall, axis=1)
                idx = idx[rows, order_small]
                nbr_d = np.sqrt(Dsmall[rows, order_small])  # (K, M), M<=kq-1

                # 1-NN and base (median of first kk neighbors)
                M = nbr_d.shape[1]  # actual neighbors available
                d1 = nbr_d[:, 0]  # (K,)
                nn1 = idx[:, 0]  # (K,)

                kk = int(min(k_radius, M))  # how many to use for base
                if kk <= 1:
                    base_vec = nbr_d[:, 0]
                else:
                    base_vec = np.median(nbr_d[:, :kk], axis=1)
                base = base_vec[:, None]  # (K,1)

                # handle exact duplicates: if d1≈0 and a 2-NN exists, use it
                tiny = d1 <= 1e-12
                if np.any(tiny) and M >= 2:
                    d1[tiny] = nbr_d[tiny, 1]
                elif np.any(tiny):
                    # fallback: use median of non-tiny or 1.0
                    med_nonzero = float(np.median(d1[~tiny])) if np.any(~tiny) else 1.0
                    d1[tiny] = med_nonzero

            else:
                # ----- large-K exact KDTree path with batching -----
                tree = cKDTree(anchors)

                d1 = np.empty(K, dtype=float)
                nn1 = np.empty(K, dtype=int)
                base = np.empty((K, 1), dtype=float)

                kk = max(
                    1, int(min(k_radius, kq - 1))
                )  # neighbors for base (excl. self)

                for s in range(0, K, nn_batch_size):
                    e = min(K, s + nn_batch_size)
                    d_batch, i_batch = tree.query(anchors[s:e], k=kq, workers=-1)
                    if d_batch.ndim == 1:  # (rare; only if kq==1)
                        d_batch = d_batch[:, None]
                        i_batch = i_batch[:, None]

                    # drop self at [:,0]
                    nbr_d = d_batch[:, 1:]  # (B, kq-1)
                    nbr_i = i_batch[:, 1:]

                    d1[s:e] = nbr_d[:, 0]
                    nn1[s:e] = nbr_i[:, 0]

                    if kk <= 1:
                        base[s:e, 0] = nbr_d[:, 0]
                    else:
                        base[s:e, 0] = np.median(nbr_d[:, :kk], axis=1)

                # duplicates: patch d1≈0 by asking for 2-NN for those only
                tiny = d1 <= 1e-12
                if np.any(tiny):
                    d_fix, i_fix = tree.query(anchors[tiny], k=3, workers=-1)
                    d2_fix = d_fix[:, 2]  # 2-NN (since [:,0]=self, [:,1]=1NN)
                    d1[tiny] = d2_fix

            # Optional cosmetic clipping on base (OFF by default)
            if clip_quantiles is not None:
                qlo, qhi = clip_quantiles
                lo, hi = np.quantile(base, [qlo, qhi])
                base = np.clip(base, 0.5 * lo, hi)

            # Apply global radius scale
            base = base * float(radius)  # (K,1)

            # Symmetric half-1NN cap
            cap_each = d1.copy()  # (K,)
            # tighten i by any j for which i is their 1-NN
            np.minimum.at(cap_each, nn1, d1)

            med_d1 = float(np.median(d1[d1 > 0])) if np.any(d1 > 0) else 1.0
            eps = float(cap_eps_frac) * med_d1
            safe_cap = np.maximum(0.0, (0.5 - eps) * cap_each)[:, None]  # (K,1)

            anchor_radii = np.minimum(base, safe_cap).astype(float)

    else:
        anchor_radii = np.asarray(anchor_radii, float).reshape(-1, 1)
        if anchor_radii.shape[0] != K:
            raise ValueError("anchor_radii length mismatch")

    # ---------- vectorized mapping of points into their anchors ----------
    anchors_per_point = anchors[part]  # (N,d)
    points_mean_per_partition = cool_mean(points, part)  # (K,d)
    points_centered = points - points_mean_per_partition[part]
    anchor_radii_per_point = anchor_radii[part]  # (N,1)
    anchors_max_radius = cool_max_radius(points_centered, part)  # (K,)
    anchors_max_radius = np.where(anchors_max_radius == 0.0, 1.0, anchors_max_radius)

    if use_robust_scaling:
        # per-point norms
        pts_scaled = rubust_scale_per_parent(
            points_centered, part, sigma_mult=2.5, gamma=0.8
        )
    else:
        points_max_radius = anchors_max_radius[part][:, None]  # (N,1)
        pts_scaled = points_centered / points_max_radius

    # final move
    mapped = anchors_per_point + anchor_radii_per_point * pts_scaled

    return (
        mapped,  # (N,d)
        anchor_radii[:, 0],  # (K,)
        points_mean_per_partition,  # (K,d)
        anchors_max_radius,  # (K,)
    )


def move_projected_points_to_anchors_v1(
    points,
    anchors,
    partition,
    radius=0.9,
    anchor_radii=None,
    real_nn_threshold=30000,
    verbose=False,
):
    if anchor_radii is None:
        if anchors.shape[0] <= real_nn_threshold:
            distance_matrix = pairwise.pairwise_distances(
                anchors, anchors, metric="euclidean"
            )
            np.fill_diagonal(distance_matrix, 1e12)
            nearest_neighbor_idx = np.argmin(distance_matrix, axis=1).flatten()
        else:
            if verbose:
                print("Using ann to approximate 1-nns of the projected points...")
            knn_index = NNDescent(
                anchors, n_neighbors=2, metric="euclidean", verbose=verbose
            )
            nns, _ = knn_index.neighbor_graph
            nearest_neighbor_idx = nns[:, 1]

        anchor_distances_from_nns = np.linalg.norm(
            anchors - anchors[nearest_neighbor_idx], axis=1, keepdims=True
        )
        anchor_radii = anchor_distances_from_nns * radius

    anchors_per_point = anchors[partition]
    anchor_radii_per_point = anchor_radii[partition]

    points_mean_per_partition = cool_mean(points, partition)
    points_centered = points - points_mean_per_partition[partition]

    anchors_max_radius = cool_max_radius(points_centered, partition)
    anchors_max_radius = np.where(anchors_max_radius == 0.0, 1.0, anchors_max_radius)
    points_max_radius = np.expand_dims(anchors_max_radius[partition], axis=1)

    return (
        anchors_per_point
        + anchor_radii_per_point * points_centered / points_max_radius,
        anchor_radii[:, 0],
        points_mean_per_partition,
        anchors_max_radius,
    )


# ---------------------- Integrated multi_step_projection ---------------------- #


def multi_step_projection(
    data,
    partitions,
    partition_labels,
    radius,
    ann_threshold,
    dim=2,
    hnne_version: HNNEVersion = "version_2",
    partition_sizes=None,
    prefered_num_clust=None,
    requested_partition=None,
    preliminary_embedding="pca",
    random_state=None,
    verbose=False,
    # v2 knobs
    v2_size_threshold=100,  # (kept for API; ignored in auto mode)
    # start-at target (optional): "auto" or int
    start_cluster_view="auto",
):
    """
    h-NNE v2: integrate fast hierarchical packing at the top of the hierarchy
    and seed the classic h-NNE descent with those anchors/radii.

    :: hnne_version selects between the legacy v1 projection and the packing-aware v2 pipeline (default).

    Returns:
      curr_anchors, anchor_radii, moved_anchors, pca, scaler,
      points_means, points_max_radii, inflation_params_list,
      v2_layout
    """
    ver = _normalize_hnne_version(hnne_version)
    use_v2 = ver == "v2"

    # 0) Preliminary projection
    projected_points, pca, scaler = project_points(
        data,
        dim=dim,
        preliminary_embedding=preliminary_embedding,
        partition_sizes=partition_sizes,
        partitions=partitions,
        random_state=random_state,
        verbose=verbose,
    )

    if verbose and partition_sizes is not None:
        print(partition_sizes)

    N = int(data.shape[0])

    # Helper: projected anchors for a given partitions matrix
    def _projected_anchors_for(parts_mat):
        pa = get_finch_anchors(projected_points, partitions=parts_mat)
        return [projected_points] + pa  # index-aligned with parts_mat columns

    # Defaults (no v2 seeding)
    parts_for_loop = partitions
    projected_anchors_loop = _projected_anchors_for(parts_for_loop)
    reversed_partition_range = list(reversed(range(parts_for_loop.shape[1])))

    # v2 seeding holders
    seed_anchors = None
    seed_radii_col = None  # (K,1)
    moved_anchors = None
    anchor_radii = None
    v2_layout = []

    # 1) Decide which top levels to pack with v2 (if enabled)
    if use_v2:
        # Ensure partition_sizes present and consistent
        if partition_sizes is None:
            partition_sizes = [
                len(set(partitions[:, i])) for i in range(partitions.shape[1])
            ]
        ps = np.asarray(partition_sizes, dtype=int)

        # Respect user's preferred start if provided (and optional requested_partition)
        if prefered_num_clust is not None:
            if requested_partition is not None:
                if prefered_num_clust == len(np.unique(requested_partition)):
                    ind = [i for i, v in enumerate(ps) if v >= prefered_num_clust]
                    partitions, ps, partition_labels = partition_update(
                        partitions, ind[-1], requested_partition, ps, partition_labels
                    )
            start_cluster_view = int(prefered_num_clust)

        # If the user explicitly provided an integer start below top (coarsest) size, bump to top and warn
        if (isinstance(start_cluster_view, (int, np.integer))) and (
            start_cluster_view < ps[-1]
        ):
            print(
                f"[INFO]: The required start_cluster_view is smaller than the default top level "
                f"of the FINCH hierarchy (i.e. {ps[-1]} clusters). "
                f"Using {ps[-1]} instead. Set prefered_num_clust if you intend to override."
            )
            start_cluster_view = int(ps[-1])

        # Automatic block selection (fine→coarse indices)
        indices_v2 = choose_v2_level_block(ps, N, start_cluster_view=start_cluster_view)

        # (Optional legacy max v2 level: only apply if user *explicitly* set a small v2 level threshold)
        if (
            isinstance(v2_size_threshold, (int, np.integer))
            and v2_size_threshold > 0
            and start_cluster_view != "auto"
        ):
            idx_keep = np.where(ps <= int(v2_size_threshold))[0]
            # intersect while preserving contiguity if possible
            if idx_keep.size:
                lo, hi = indices_v2[0], indices_v2[-1]
                mask = (idx_keep >= lo) & (idx_keep <= hi)
                if np.any(mask):
                    indices_v2 = idx_keep[mask]

        if indices_v2.size > 0:
            # Split into v2 slice and the finer remainder for the classic loop
            idx0 = int(indices_v2[0])  # boundary (finest among v2)
            partitions_v2 = partitions[:, indices_v2]
            parts_for_loop = partitions[:, : idx0 + 1]

            if verbose:
                print(
                    f"[v2] packing levels (fine→coarse) indices {indices_v2.tolist()} "
                    f"with sizes {ps[indices_v2].tolist()}"
                )

            # v2 pack (ND-enabled)
            v2_layout = pack_hierarchy_iterative_k_nd(
                projected_points,
                partitions_v2,
                k=1,
                outer_sweeps=1,
                mass_mode_2d="area",
                mass_mode_nd="powerD",
            )
            anchors_by_level, radii_by_level = layout_to_level_arrays(v2_layout)

            if len(anchors_by_level) > 0:
                # boundary seed (level 0 in this sub-hierarchy)
                seed_anchors = anchors_by_level[0]
                seed_radii_col = radii_by_level[0].reshape(-1, 1)

                # Pre-populate outputs with ALL v2 levels from coarsest → boundary
                moved_anchors = [anchors_by_level[-1]]
                anchor_radii = [radii_by_level[-1]]
                for li in range(len(anchors_by_level) - 2, -1, -1):
                    moved_anchors.append(anchors_by_level[li])
                    anchor_radii.append(radii_by_level[li])

                # Prepare the finer stack for the classic loop
                projected_anchors_loop = _projected_anchors_for(parts_for_loop)
                reversed_partition_range = list(
                    reversed(range(parts_for_loop.shape[1]))
                )

            else:
                if verbose:
                    print(
                        "[v2] layout empty after packing; falling back to vanilla pipeline."
                    )

    # 2) Choose mapping kernel (v1 or optimized v2)
    if ver == "v1":
        mppta = move_projected_points_to_anchors_v1
    else:
        mppta = move_projected_points_to_anchors_v2
        radius = 0.8  # larger factor safe due to capping inside v2 kernel

    # 3) Initialize loop state
    curr_anchors = projected_anchors_loop[-1]  # coarsest projected anchors
    if seed_anchors is not None:
        curr_anchors = seed_anchors

    curr_anchor_radii = None
    if seed_radii_col is not None:
        curr_anchor_radii = seed_radii_col

    # If v2 wasn’t used, initialize outputs now like in v1
    if moved_anchors is None:
        moved_anchors = [curr_anchors]
    if anchor_radii is None:
        anchor_radii = []

    points_means = []
    points_max_radii = []
    inflation_params_list = []

    # 4) Descent through the remaining (finer) levels
    for cnt, i in enumerate(reversed_partition_range):
        partition_mapping = parts_for_loop[:, 0] if i == 0 else partition_labels[i - 1]
        current_points = projected_anchors_loop[i]

        # angle normalization for 2D/3D
        if dim == 2:
            thetas = np.linspace(0, np.pi / 2, 6)
            current_points, inflation_params = norm_angles(
                current_points, thetas, partition_mapping
            )
            inflation_params_list.append(inflation_params)
        elif dim == 3:
            alphas, beta, gammas = 3 * [np.linspace(0, np.pi / 2, 6)]
            current_points, inflation_params = norm_angles_3d(
                current_points, alphas, beta, gammas, partition_mapping
            )
            inflation_params_list.append(inflation_params)

        # first pass may use v2 radii; subsequent passes use None
        curr_anchors, radii, points_mean, points_max_radius = mppta(
            current_points,
            curr_anchors,
            partition_mapping,
            radius=radius,
            anchor_radii=curr_anchor_radii,
            real_nn_threshold=ann_threshold,
            verbose=verbose,
        )

        anchor_radii.append(radii)
        moved_anchors.append(curr_anchors)
        points_means.append(points_mean)
        points_max_radii.append(points_max_radius)

        # After the first pass, drop the v2 radii seed
        if curr_anchor_radii is not None:
            curr_anchor_radii = None

    return (
        curr_anchors,
        anchor_radii,
        moved_anchors,
        pca,
        scaler,
        points_means,
        points_max_radii,
        inflation_params_list,
        v2_layout,
    )
