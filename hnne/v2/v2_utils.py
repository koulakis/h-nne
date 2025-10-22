from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from hnne.cool_functions import cool_max, cool_mean

HNNEVersion = Literal["v1", "v2", "version_1", "version_2", "1", "2", "auto"]


def normalize_hnne_version(
    hnne_version: HNNEVersion = "version_2",
) -> Literal["v1", "v2"]:
    v = str(hnne_version).lower().strip()
    if v in {"v2", "version_2", "2", "auto"}:
        return "v2"
    if v in {"v1", "version_1", "1"}:
        return "v1"
    # Fallback: prefer v2
    return "v2"


# -----  v2 utilities  ------#
def rescale_layout(
    layout: Dict[Tuple[int, int], Tuple[float, ...]], target_top_median: float = 1.0
) -> Dict[Tuple[int, int], Tuple[float, ...]]:
    """Uniformly scale all coordinates/radii so median top radius = target."""
    levels = [lvl for (lvl, _) in layout.keys()]
    if not levels:
        return dict(layout)
    top = max(levels)
    top_r = [layout[(lvl, lab)][-1] for (lvl, lab) in layout.keys() if lvl == top]
    if not top_r:
        return dict(layout)
    med_r = float(np.median(top_r))
    if med_r <= 0:
        return dict(layout)
    s = target_top_median / med_r
    out = {}
    for k, vals in layout.items():
        coords = np.array(vals[:-1], float) * s
        r = float(vals[-1]) * s
        out[k] = tuple(coords.tolist() + [r])
    return out


def layout_to_level_arrays(
    layout: Dict[Tuple[int, int], Tuple[float, ...]],
    levels: Sequence[int] | None = None,
    sort_labels: bool = True,
    return_labels: bool = False,
) -> Union[
    Tuple[List[np.ndarray], List[np.ndarray]],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
]:
    """
    Convert a layout dict {(level, label): (*coords, r)} into per-level arrays.

    Parameters
    ----------
    layout : dict
        Keys are (level, label); values are (*coords, radius).
    levels : sequence[int] | None
        If given, only include these levels (in this order). If None, uses all
        levels present in `layout`, sorted ascending.
    sort_labels : bool
        If True, sort items within each level by label ascending for stable output.
    return_labels : bool
        If True, also return a list of 1D int arrays with labels per level.

    Returns
    -------
    anchors_by_level : list[np.ndarray]
        For each level, an array of shape (n_i, D) of centers.
    radii_by_level : list[np.ndarray]
        For each level, an array of shape (n_i,) of radii.
    labels_by_level : list[np.ndarray]  (only if return_labels=True)
        For each level, the labels (int) in the same row order.
    """
    if not layout:
        return ([], [], []) if return_labels else ([], [])

    # collect levels
    all_levels = (
        sorted({lvl for (lvl, _) in layout.keys()}) if levels is None else list(levels)
    )

    anchors_list: List[np.ndarray] = []
    radii_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for lvl in all_levels:
        # pull items for this level
        items = [
            (lab, layout[(lvl, lab)])
            for (L, lab) in layout.keys()
            if L == lvl and (lvl, lab) in layout
        ]
        if not items:
            # empty level (allowed if user passed explicit `levels`)
            if return_labels:
                anchors_list.append(np.zeros((0, 0), float))
                radii_list.append(np.zeros((0,), float))
                labels_list.append(np.zeros((0,), int))
            else:
                anchors_list.append(np.zeros((0, 0), float))
                radii_list.append(np.zeros((0,), float))
            continue

        if sort_labels:
            items.sort(key=lambda t: t[0])

        # infer dimensionality from first entry
        D = len(items[0][1]) - 1  # coords length
        m = len(items)

        A = np.empty((m, D), dtype=float)
        r = np.empty((m,), dtype=float)
        labs = np.empty((m,), dtype=int)

        for i, (lab, vals) in enumerate(items):
            A[i] = np.asarray(vals[:-1], float)
            r[i] = float(vals[-1])
            labs[i] = int(lab)

        anchors_list.append(A)
        radii_list.append(r)
        if return_labels:
            labels_list.append(labs)

    return (
        (anchors_list, radii_list, labels_list)
        if return_labels
        else (anchors_list, radii_list)
    )


# ---------------Robust scaling for move_projected_points-------#
# Robust per-parent scale instead of plain max


def rubust_scale_per_parent(
    points_centered: np.ndarray,  # (N,D),
    partition: np.ndarray,  # parent partition labels (N, 1)
    sigma_mult=2.5,  # per-parent scale = mean + sigma_mult * std
    gamma=0.9,  # radial power warp (<1 spreads mid-radii; 1.0 disables)
):
    """
    -   per-parent scale uses mean + λ·std of
        ||points - parent_mean|| instead of max, capped by the true max. This
        avoids a few outliers collapsing everyone else to the center.
      - Optional radial power warp `gamma<1` spreads typical points a bit to
        better fill large parents while keeping outliers bounded.
    """
    # print(f'runnig robust scaling')
    # per-point norms
    norms = np.linalg.norm(points_centered, axis=1)  # (n,)

    # per-parent mean and mean of squares -> std (fast with cool_mean)
    mu = cool_mean(norms[:, None], partition).ravel()  # (k,)
    mu2 = cool_mean((norms**2)[:, None], partition).ravel()  # (k,)
    var = np.maximum(mu2 - mu**2, 0.0)
    std = np.sqrt(var)

    # robust scale: mean + λ * std, then cap by true max so we never expand
    # beyond the farthest member in the parent
    r_max = cool_max(norms, partition)  # (k,)
    robust = mu + sigma_mult * std  # (k,)
    denom = np.minimum(robust, r_max)  # (k,)
    denom = np.where(denom <= 1e-12, 1.0, denom)  # avoid 0

    # per-point denominator
    denom_pp = denom[partition][:, None]  # (n,1)

    # radial power warp (gamma < 1 spreads mid-radii)
    rho = np.linalg.norm(points_centered, axis=1, keepdims=True) / denom_pp
    rho = np.clip(rho, 0.0, 1.0)  # stay in [0,1]
    unit = points_centered / np.maximum(
        np.linalg.norm(points_centered, axis=1, keepdims=True), 1e-12
    )
    pts_scaled = unit * (rho**gamma)  # (n,2)
    return pts_scaled


def partition_update(c, idx, req_c, num_clust, partition_clustering):
    if c.shape[-1] - 1 == idx:
        partition_clustering[idx] = np.zeros(len(np.unique(req_c)))
    else:
        _, ig = np.unique(c[:, idx + 1], return_inverse=True)
        u, ig_ = np.unique(req_c, return_index=True)
        partition_clustering[idx] = ig[ig_]

    _, ig = np.unique(req_c, return_inverse=True)
    u, ig_ = np.unique(c[:, idx - 1], return_index=True)
    partition_clustering[idx - 1] = ig[ig_]

    c[:, idx] = req_c
    num_clust[idx] = len(np.unique(req_c))
    return c, num_clust, partition_clustering


# -------------------#######--------------------------------------------------------------------------#
# #------------ Policy for v2 auto level selection ------------------------------------##


def _nearest_level_geq_coarsest(sizes: np.ndarray, target: int) -> int:
    """
    Return the *coarsest* index i (largest i) with sizes[i] >= target.
    If none exist (all < target), fall back to the index with the nearest size
    (ties broken toward the coarser side).
    """
    # geq = np.where(sizes >= target)[0]
    # if geq.size:
    #    return int(geq.max())
    # fallback: nearest by absolute difference; prefer coarser on ties
    diff = np.abs(sizes - target)
    best = int(np.argmin(diff))
    # tie-break toward coarser (larger index)
    ties = np.where(diff == diff[best])[0]
    return int(ties.max())


def _choose_policy_by_N(N: int) -> Tuple[str, int, int, Optional[int]]:
    """
    Map dataset size N to a level-selection policy.
    Returns a tuple:
      (mode, start_min_clusters, max_extra_levels, end_max_clusters)

    mode:
      - "top_to_threshold": start at *top* (coarsest) and include levels down to a cluster-cap
      - "start_and_descend": start near 'start_min_clusters' and include up to 'max_extra_levels'
                             finer levels, optionally capping by 'end_max_clusters'
    """
    if N <= 500_000:
        # Start from the very top; include levels down until ≤ 500 clusters
        return ("top_to_threshold", 0, 0, 500)     
    if N <= 1000_000:
        # Start from the very top; include levels down until ≤ 1k clusters
        return ("top_to_threshold", 10, 2, 500)     
    elif N <= 5000_000:
        # Start around ≥10 clusters; descend up to 2 levels, but don't exceed 1k clusters
        return ("start_and_descend", 50, 2, 2_000)
    elif N <= 10_000_000:
        # Start ≥1000; go 1 level
        return ("start_and_descend", 500, 1, 10_000)
    elif N <= 50_000_000:
        # Start ≥2000; go 1 level
        return ("start_and_descend", 5_000, 1, 50_000)
    else:
        # Very large: only one v2 level near ~50k, then pass to v1
        return ("start_and_descend", 10_000, 0, None)


def choose_v2_level_block(
    partition_sizes: Sequence[int],
    N: int,
    start_cluster_view: Union[str, int, None] = "auto",
) -> np.ndarray:
    """
    Decide which FINCH hierarchy levels (indices) to use for the v2 packer.

    Parameters
    ----------
    partition_sizes : Sequence[int]
        Number of clusters per level, ordered fine→coarse (monotonically decreasing).
        Example: [49316, 11032, 2543, 421, 41, 8]
    N : int
        Number of data points.
    start_cluster_view : {"auto", int, None}
        - "auto" / None: pick start/end levels automatically from N.
        - int: respect this as the desired starting cluster count (nearest level chosen),
               but choose how many child levels to include automatically per the N-policy.

    Returns
    -------
    np.ndarray
        Sorted indices (ascending, i.e., fine→coarse) forming a *contiguous* block
        [fine_start, ..., coarse_end], where the last index is the coarsest included level.
        Safe to use as: partitions_v2 = partitions[:, indices]
                        partitions_v1 = partitions[:, :indices[0] + 1]
    """
    sizes = np.asarray(partition_sizes, dtype=int)
    L = sizes.size
    if L == 0:
        return np.array([], dtype=int)

    # Sanity: sizes should be monotonically decreasing (fine→coarse)
    # We'll tolerate small deviations but selection assumes decreasing trend.

    mode, auto_start_min, max_extra, end_cap = _choose_policy_by_N(int(N))

    # --- Case A: small-N special (start at top; go down to a cluster cap) ---
    if (start_cluster_view in ("auto", None)) and (mode == "top_to_threshold"):
        # first index from fine-side that is ≤ end_cap (default 2000)
        # if none, include everything (s_idx = 0)
        s_idx = int(np.argmax(sizes <= end_cap)) if np.any(sizes <= end_cap) else 0
        e_idx = L - 1  # top (coarsest)
        return np.arange(s_idx, e_idx + 1, dtype=int)

    # --- Case B: start near a cluster target and descend a few levels ---
    # Determine the desired start cluster count
    if start_cluster_view in ("auto", None):
        start_target = auto_start_min
    else:
        start_target = int(start_cluster_view)

    # Choose the *coarsest* level with clusters ≥ start_target (or nearest if none)
    s = _nearest_level_geq_coarsest(sizes, start_target)

    # Decide how many finer levels to include beneath s
    steps = 0
    for step in range(1, max_extra + 1):
        cand = s - step
        if cand < 0:
            break
        if end_cap is not None and sizes[cand] > end_cap:
            break
        steps = step

    f = s - steps  # finest included index
    e = s  # coarsest included index (the chosen start)
    return np.arange(f, e + 1, dtype=int)
