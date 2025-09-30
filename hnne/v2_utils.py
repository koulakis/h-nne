from math import sqrt, pi, log2
import numpy as np
from hnne.cool_functions import cool_max_radius, cool_max, cool_mean
from typing import List, Dict, Tuple, Optional, Sequence, Union


#-----  v2 utilities  ------#
def rescale_layout(
    layout: Dict[Tuple[int,int], Tuple[float, ...]],
    target_top_median: float = 1.0
) -> Dict[Tuple[int,int], Tuple[float, ...]]:
    """Uniformly scale all coordinates/radii so median top radius = target."""
    levels = [lvl for (lvl, _) in layout.keys()]
    if not levels: return dict(layout)
    top = max(levels)
    top_r = [layout[(lvl, lab)][-1] for (lvl, lab) in layout.keys() if lvl == top]
    if not top_r: return dict(layout)
    med_r = float(np.median(top_r))
    if med_r <= 0: return dict(layout)
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
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
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
    all_levels = sorted({lvl for (lvl, _) in layout.keys()}) if levels is None else list(levels)

    anchors_list: List[np.ndarray] = []
    radii_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for lvl in all_levels:
        # pull items for this level
        items = [(lab, layout[(lvl, lab)]) for (L, lab) in layout.keys() if L == lvl and (lvl, lab) in layout]
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

    return (anchors_list, radii_list, labels_list) if return_labels else (anchors_list, radii_list)

#---------------Robust scaling for move_projected_points-------#
#Robust per-parent scale instead of plain max

def rubust_scale_per_parent(points_centered: np.ndarray,          # (N,D),
                            partition: np.ndarray,                # parent partition labels (N, 1)
                            sigma_mult=2.5,                       # per-parent scale = mean + sigma_mult * std
                            gamma=0.9                             # radial power warp (<1 spreads mid-radii; 1.0 disables)
                           ):

    """
    -   per-parent scale uses mean + λ·std of
        ||points - parent_mean|| instead of max, capped by the true max. This
        avoids a few outliers collapsing everyone else to the center.
      - Optional radial power warp `gamma<1` spreads typical points a bit to
        better fill large parents while keeping outliers bounded.
    """
    #print(f'runnig robust scaling')
    # per-point norms
    norms = np.linalg.norm(points_centered, axis=1)  # (n,)

    # per-parent mean and mean of squares -> std (fast with cool_mean)
    mu   = cool_mean(norms[:, None], partition).ravel()             # (k,)
    mu2  = cool_mean((norms**2)[:, None], partition).ravel()        # (k,)
    var  = np.maximum(mu2 - mu**2, 0.0)
    std  = np.sqrt(var)

    # robust scale: mean + λ * std, then cap by true max so we never expand
    # beyond the farthest member in the parent
    r_max = cool_max(norms, partition)                      # (k,)
    robust = mu + sigma_mult * std                          # (k,)
    denom = np.minimum(robust, r_max)                       # (k,)
    denom = np.where(denom <= 1e-12, 1.0, denom)            # avoid 0

    # per-point denominator
    denom_pp = denom[partition][:, None]                    # (n,1)

    # radial power warp (gamma < 1 spreads mid-radii)
    rho  = np.linalg.norm(points_centered, axis=1, keepdims=True) / denom_pp
    rho  = np.clip(rho, 0.0, 1.0)                           # stay in [0,1]
    unit = points_centered / np.maximum(
            np.linalg.norm(points_centered, axis=1, keepdims=True), 1e-12
        )
    pts_scaled = unit * (rho ** gamma)                      # (n,2)
    return pts_scaled



def partition_update(c, idx, req_c, num_clust, partition_clustering):
    if c.shape[-1] - 1 == idx:
        partition_clustering[idx] = np.zeros(len(np.unique(req_c)))
    else:
        _, ig = np.unique(c[:, idx + 1], return_inverse=True)
        u, ig_ = np.unique(req_c, return_index=True)
        partition_clustering[idx] = ig[ig_]

    _, ig = np.unique(req_c, return_inverse=True)
    u, ig_ = np.unique(c[:, idx -1], return_index=True)
    partition_clustering[idx - 1] = ig[ig_]  

    c[:, idx] = req_c
    num_clust[idx] = len(np.unique(req_c))
    return c, num_clust, partition_clustering
    
#-------------------#######--------------------------------------------------------------------------#
#-----  optional utilities  ------#


def generate_hierarchical_labels(N, levels, top_cluster_sizes, min_children=3, max_children=8, seed=None):
    """
    Generate a (N, levels) cluster label matrix with variable branching per cluster.

    Parameters:
    - N: number of data points
    - levels: number of hierarchy levels (e.g., 4)
    - top_cluster_sizes: list of proportions for top clusters (must sum to 1)
    - min_children, max_children: range of subclusters per parent
    - seed: random seed

    Returns:
    - c: (N, levels) integer array of hierarchical cluster labels
    """
    def get_partiton_labels(partitions):
        partition_labels=[]
        partition_sizes = []
        for partition in range(partitions.shape[1] - 1):
            _, ig = np.unique(partitions[:, partition + 1], return_inverse=True)
            u, ig_ = np.unique(partitions[:, partition], return_index=True)
            partition_labels.append(ig[ig_])
            partition_sizes.append(len(u))
        num_top_level = len(np.unique(partitions[:, -1]))
        return partition_labels + [np.zeros(num_top_level)], partition_sizes + [num_top_level]
    
    if seed is not None:
        np.random.seed(seed)

    c = np.full((N, levels), -1, dtype=int)
    next_label = 0  # To generate unique cluster IDs

    # Step 1: Top level cluster assignment
    top_cluster_counts = (np.array(top_cluster_sizes) * N).astype(int)
    top_cluster_counts[-1] = N - top_cluster_counts[:-1].sum()  # Adjust last to ensure sum = N
    top_labels = []
    offset = 0
    for cluster_id, count in enumerate(top_cluster_counts):
        c[offset:offset+count, -1] = cluster_id
        top_labels.append((cluster_id, np.arange(offset, offset+count)))
        offset += count

    # Step 2: For each level from top to bottom, recursively split each cluster
    label_counter = {level: {} for level in range(levels)}
    label_counter[levels - 1] = {cid: np.where(c[:, levels - 1] == cid)[0] for cid in range(len(top_cluster_sizes))}
    next_cluster_id = 0

    for level in reversed(range(levels - 1)):
        label_counter[level] = {}
        for parent_id, indices in label_counter[level + 1].items():
            num_children = np.random.randint(min_children, max_children + 1)
            child_sizes = np.random.multinomial(len(indices), np.random.dirichlet(np.ones(num_children)))
            start = 0
            for i, size in enumerate(child_sizes):
                if size == 0:
                    continue
                child_indices = indices[start:start+size]
                child_id = next_cluster_id
                c[child_indices, level] = child_id
                label_counter[level][child_id] = child_indices
                next_cluster_id += 1
                start += size
    partition_labels, partition_sizes = get_partiton_labels(c)
    # small fix for finch_like cluster labels starting from 0 onwards at each level
    partitions = c
    for col in range(c.shape[1]):
        _, ig = np.unique(c[:, col], return_inverse=True)
        partitions[:, col] = ig
    return partitions, partition_sizes, partition_labels

#-----------------------------
def format_time(seconds):
    """Format a duration in seconds into hr:min:sec or ms."""
    if seconds < 1:
        return f"{int(seconds * 1000)} ms"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"

def format_count(n):
    """Format large sample counts into readable form."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    else:
        return str(n)