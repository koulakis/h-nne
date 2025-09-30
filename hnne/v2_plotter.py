import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import defaultdict
from typing import Dict, Tuple, Sequence, List, Optional, Union


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

#----------------- single level 2D/3D plotters ------------------#
def plot_layout_level_2d(
    layout: Dict[Tuple[int,int], Tuple[float, ...]],
    level: int,
    title: str = "2D layout",
    draw_labels: bool = False,
    figsize=(8,8)
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    items = [(lab, vals) for (lvl, lab), vals in layout.items() if lvl == level]
    if not items:
        raise ValueError(f"No entries at level {level}")
    fig, ax = plt.subplots(figsize=figsize)
    xs, ys = [], []
    for lab, vals in items:
        x, y, r = vals[0], vals[1], vals[-1]
        ax.add_patch(Circle((x, y), r, fill=False, linewidth=1.2, edgecolor="k", alpha=0.9))
        if draw_labels:
            ax.text(x, y, f"{lab}", ha="center", va="center", fontsize=8)
        xs += [x-r, x+r]; ys += [y-r, y+r]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    padx = 0.06*(xmax-xmin) if xmax>xmin else 1.0
    pady = 0.06*(ymax-ymin) if ymax>ymin else 1.0
    ax.set_xlim(xmin-padx, xmax+padx); ax.set_ylim(ymin-pady, ymax+pady)
    ax.set_aspect("equal", "box"); ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title(title); ax.set_xlabel("X"); ax.set_ylabel("Y"); plt.tight_layout(); plt.show()


def plot_layout_level_3d(
    layout: Dict[Tuple[int,int], Tuple[float, ...]],
    level: int,
    anchors: Optional[Dict[int, Tuple[float, float, float]]] = None,
    title: str = "3D layout",
    figsize=(8,8),
    draw_labels: bool = False,
    sphere_wire: bool = False
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    items = [(lab, vals) for (lvl, lab), vals in layout.items() if lvl == level]
    if not items:
        raise ValueError(f"No entries at level {level}")
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = [], [], []
    # draw centers; optionally wire spheres (simple parametric)
    for lab, vals in items:
        x, y, z, r = vals[0], vals[1], vals[2], vals[-1]
        ax.scatter([x], [y], [z], s=20, c='k')
        if sphere_wire:
            u = np.linspace(0, 2*np.pi, 24)
            v = np.linspace(0, np.pi, 12)
            X = x + r*np.outer(np.cos(u), np.sin(v))
            Y = y + r*np.outer(np.sin(u), np.sin(v))
            Z = z + r*np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(X, Y, Z, color='gray', linewidth=0.5, alpha=0.4)
        if draw_labels:
            ax.text(x, y, z, f"{lab}", fontsize=7)
        xs += [x-r, x+r]; ys += [y-r, y+r]; zs += [z-r, z+r]
    if anchors:
        A = np.array([anchors[k] for k in anchors.keys()], float)
        ax.scatter(A[:,0], A[:,1], A[:,2], s=10, c='tab:red', marker='x')
        xs += A[:,0].tolist(); ys += A[:,1].tolist(); zs += A[:,2].tolist()
    # frame
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys); zmin, zmax = min(zs), max(zs)
    span = max(xmax-xmin, ymax-ymin, zmax-zmin)
    cx, cy, cz = 0.5*(xmax+xmin), 0.5*(ymax+ymin), 0.5*(zmax+zmin)
    pad = 0.06*span if span>0 else 1.0
    ax.set_xlim(cx-0.5*span-pad, cx+0.5*span+pad)
    ax.set_ylim(cy-0.5*span-pad, cy+0.5*span+pad)
    ax.set_zlim(cz-0.5*span-pad, cz+0.5*span+pad)
    ax.set_title(title); plt.tight_layout(); plt.show()

#------------------------------------------- Nested (parent - children) layout plotter----------------------------------------
def _strict_child_to_parent_map(
    partitions: np.ndarray,
    child_level: int,
    parent_level: int
) -> Dict[int, int]:
    """
    Build a strict child->parent label map from partitions[:, child_level] to partitions[:, parent_level].
    Raises if a child maps to multiple parents.
    """
    parts = np.asarray(partitions)
    child = parts[:, child_level]
    parent = parts[:, parent_level]
    c2p: Dict[int, int] = {}
    for c in np.unique(child):
        ps = np.unique(parent[child == c])
        if len(ps) != 1:
            raise ValueError(
                f"Inconsistent hierarchy at levels {child_level}->{parent_level}: "
                f"child {int(c)} -> parents {ps.tolist()}"
            )
        c2p[int(c)] = int(ps[0])
    return c2p


def plot_v2_nested_2d(
    layout: Dict[Tuple[int, int], Tuple[float, ...]],
    partitions: np.ndarray,
    *,
    parent_level: int,
    child_levels: Sequence[int],
    figsize: tuple = (9, 9),
    cmap: str = "tab20",
    parent_linewidth: float = 1.6,
    parent_linestyle: str = "--",
    parent_alpha: float = 0.9,
    child_linewidth: float = 0.9,
    child_alpha: float = 0.85,
    draw_parent_labels: bool = False,
    draw_child_labels: bool = False,
    child_label_limit: int = 150,   # avoid clutter
    parent_subset: Optional[Sequence[int]] = None,  # if we want to show only some parents
    time_elapsed: Optional[float] = None
):
    """
    Plot a chosen parent level as dotted circles and overlay one or more child levels
    (each drawn inside its respective parent, using the hierarchy from `partitions`).

    Parameters
    ----------
    layout : {(level,label): (x,y,r)}
        Output from pack_hierarchy_iterative_k_nd (2D levels expected).
    partitions : (N, L) int
        Original hierarchy labels; used to recover child->parent relations.
    parent_level : int
        Level index (as used in `layout` keys) to draw as parents (dotted).
    child_levels : list[int]
        One or more level indices (lower than parent_level) to draw as children.
    parent_subset : list[int] | None
        If provided, only these parent labels (at `parent_level`) are drawn; children filtered accordingly.
    """
    # --- collect parent items from layout ---
    parents_items = [(lab, layout[(parent_level, lab)])
                     for (lev, lab) in layout.keys()
                     if lev == parent_level and (parent_level, lab) in layout]
    if not parents_items:
        raise ValueError(f"No entries at parent level {parent_level} in layout.")

    # Order parents by label for consistent coloring
    parents_items.sort(key=lambda t: t[0])
    parent_labels_all = [int(lab) for lab, _ in parents_items]
    if parent_subset is not None:
        parent_subset = set(int(x) for x in parent_subset)
        parents_items = [(lab, vals) for (lab, vals) in parents_items if lab in parent_subset]
        if not parents_items:
            raise ValueError("parent_subset filtered out all parents.")

    parent_labels = [int(lab) for lab, _ in parents_items]
    P_par = np.array([vals[:2] for _, vals in parents_items], float)   # (P,2)
    R_par = np.array([vals[-1] for _, vals in parents_items], float)   # (P,)

    # --- build child->parent mapping per child level ---
    child_levels = list(child_levels)
    # sanity: ensure child levels are below parent_level
    for cl in child_levels:
        if cl >= parent_level:
            raise ValueError(f"Child level {cl} must be < parent_level {parent_level}.")

    c2p_by_level: Dict[int, Dict[int, int]] = {}
    for cl in child_levels:
        c2p_by_level[cl] = _strict_child_to_parent_map(partitions, cl, parent_level)

    # --- gather children from layout and group by parent for each child level ---
    child_items_by_parent: Dict[int, Dict[int, List[Tuple[int, Tuple[float, ...]]]]] = {
        p: {cl: [] for cl in child_levels} for p in parent_labels
    }
    # for each child level, collect available children in layout
    for cl in child_levels:
        # all items at this child level present in layout
        items = [(lab, layout[(cl, lab)]) for (lev, lab) in layout.keys()
                 if lev == cl and (cl, lab) in layout]
        for lab, vals in items:
            c = int(lab)
            p = c2p_by_level[cl].get(c, None)
            if p is None:
                continue
            if parent_subset is not None and (p not in parent_subset):
                continue
            if p in child_items_by_parent:
                child_items_by_parent[p][cl].append((c, vals))

    # --- choose colors per parent ---
    import matplotlib.cm as cm
    cmap_obj = cm.get_cmap(cmap, max(2, len(parent_labels)))
    color_of_parent = {p: cmap_obj(i) for i, p in enumerate(parent_labels)}

    # --- make plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # draw parents (dotted outline)
    for (lab, vals) in parents_items:
        x, y, r = float(vals[0]), float(vals[1]), float(vals[-1])
        color = color_of_parent[int(lab)]
        circ = Circle((x, y), r, fill=False,
                      linestyle=parent_linestyle,
                      linewidth=parent_linewidth,
                      edgecolor=color,
                      alpha=parent_alpha)
        ax.add_patch(circ)
        if draw_parent_labels:
            ax.text(x, y, f"P{lab}", ha="center", va="center", fontsize=9, color=color, alpha=0.9)

    # draw children (outlines, tinted by their parent color)
    # loop per parent to keep children visually grouped
    child_labels_drawn = 0
    for p in parent_labels:
        p_color = color_of_parent[p]
        for cl in child_levels:
            items = child_items_by_parent[p][cl]
            if not items:
                continue
            # light tint for this child level
            # (stacking multiple levels: make deeper levels a bit darker)
            level_idx = child_levels.index(cl)
            alpha_level = max(0.35, child_alpha - 0.08 * level_idx)

            for c_lab, vals in items:
                x, y, r = float(vals[0]), float(vals[1]), float(vals[-1])
                ax.add_patch(Circle((x, y), r, fill=False,
                                    linewidth=child_linewidth,
                                    edgecolor=p_color,
                                    alpha=alpha_level))
                if draw_child_labels and (child_labels_drawn < child_label_limit):
                    ax.text(x, y, f"{c_lab}", ha="center", va="center", fontsize=7,
                            color=p_color, alpha=0.85)
                    child_labels_drawn += 1


    # --- collect child items from layout for title display---
    num_child = []
    for i, c_lev in enumerate(child_levels):
        c_items = [(lab, layout[(c_lev, lab)])
                     for (lev, lab) in layout.keys()
                     if lev == c_lev and (c_lev, lab) in layout]
        
        num_child.append(str(f'L - {i + 1}: {len(c_items)} clusters'))
    if len(num_child) != 0:
        print_str = str(f"Start@Parent L: {len(parents_items)} clusters, Children@{', '.join(map(str,num_child))} ({time_elapsed})")
    else:
        print_str = str(f"Start@Parent L with {len(parents_items)} clusters ({time_elapsed})")
        
    # -----
    
    # autoscale to parents' extents
    xs = (P_par[:, 0] - R_par).tolist() + (P_par[:, 0] + R_par).tolist()
    ys = (P_par[:, 1] - R_par).tolist() + (P_par[:, 1] + R_par).tolist()
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_x = 0.06 * (xmax - xmin) if xmax > xmin else 1.0
    pad_y = 0.06 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_title(print_str)
    ax.set_xlabel("h-NNE 1")
    ax.set_ylabel("h-NNE 2")
    plt.tight_layout()
    #plt.show()
    return plt


#----------------------------------------- 4-subplot grid------------------------##
def plot_projection_grid(
    arrs: List= None,
    labels: np.ndarray | None = None,
    *,
    titles=["plot-1", "plot-2", "plot-3", "plot-4"],
    figsize=(12, 10),
    point_size=3,
    top_colors=("#e41a1c", "#377eb8", "#4daf4a", "#984ea3"),  # red, blue, green, purple
    other_color="#bdbdbd",
    other_alpha=0.12,
    top_alpha=0.7,
    hide_ticks=True,
    equal_aspect=False,
    show=True,
):
    """
    Plot a 2x2 grid of 2D projections.

    Parameters
    ----------
    pca_xy, lda_xy, hnne_pca_xy, hnne_lda_xy : (N, 2) arrays
        The four 2D projections (same number of points N, same order).
    labels : (N,) array-like or None
        Ground-truth labels. If provided, the 4 largest classes are colored
        (red, blue, green, purple) and shown in the legend of the first subplot.
        All other classes are rendered in light gray.
    titles : tuple[str, str, str, str]
        Titles for the four subplots (in row-major order).
    figsize : tuple
        Figure size in inches.
    point_size : float
        Marker size for scatter points.
    top_colors : tuple[str, str, str, str]
        Colors for the top-4 classes (largest to smallest among the top).
    other_color : str
        Color for all non-top classes.
    other_alpha : float
        Alpha for the non-top classes.
    top_alpha : float
        Alpha for the top classes.
    hide_ticks : bool
        If True, hide axis ticks for a cleaner look.
    equal_aspect : bool
        If True, set axes to equal aspect ratio.
    """
    # Basic shape checks
    for a in arrs:
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError("All projection arrays must have shape (N, 2).")
    n = arrs[0].shape[0]
    if any(a.shape[0] != n for a in arrs[1:]):
        raise ValueError("All projections must have the same number of rows (N).")

    labels = None if labels is None else np.asarray(labels)
    if labels is not None and len(labels) != n:
        raise ValueError("labels must have length N.")
    if len(arrs) == 4:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
    elif len(arrs) == 2:
        figsize = (12, 5)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes = axes.ravel()
    elif len(arrs) == 1:
        figsize = (5, 5)
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]
    
    

    if labels is None:
        # No labels: just plot all points in gray on each subplot
        for ax, xy, title in zip(axes, arrs, titles):
            ax.scatter(xy[:, 0], xy[:, 1],
                       s=point_size, c=other_color, alpha=0.5)
            ax.set_title(title)
            if equal_aspect:
                ax.set_aspect("equal", adjustable="box")
            if hide_ticks:
                ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.show()
        return fig, axes

    # With labels: determine top-4 classes by count
    uniq, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    uniq = uniq[order]
    counts = counts[order]
    top = uniq[:min(4, len(uniq))]
    top_counts = counts[:min(4, len(uniq))]
    # Precompute masks for speed
    top_masks = {lbl: (labels == lbl) for lbl in top}
    others_mask = ~np.isin(labels, top)

    # Color map for top classes
    color_map = {lbl: top_colors[i] for i, lbl in enumerate(top)}

    # Plot each subplot
    for ax, xy, title in zip(axes, arrs, titles):
        # Others first (so top classes sit on top visually)
        if np.any(others_mask):
            ax.scatter(xy[others_mask, 0], xy[others_mask, 1],
                       s=point_size, c=other_color, alpha=other_alpha, label=None)

        # Then overlay each top class in its assigned color
        handles = []
        labels_for_legend = []
        for i, lbl in enumerate(top):
            mask = top_masks[lbl]
            if not np.any(mask):
                continue
            sc = ax.scatter(xy[mask, 0], xy[mask, 1],
                            s=point_size, c=color_map[lbl], alpha=top_alpha, label=str(lbl))
            handles.append(sc)
            labels_for_legend.append(str(lbl)+': '+str(top_counts[i])+'-samples')

        ax.set_title(title) #fontsize=16, weight='bold'
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        if hide_ticks:
            ax.set_xticks([]); ax.set_yticks([])

        # Legend only on the first subplot
        if ax is axes[0] and handles:
            ax.legend(handles, labels_for_legend, title="Largest classes", frameon=True, loc="best", markerscale=6)

    plt.tight_layout()
    if show:
        plt.show()
    return plt
