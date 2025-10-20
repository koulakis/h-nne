import math
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def time_function_call(f, *args, **kwargs):
    start = timer()
    result = f(*args, **kwargs)
    end = timer()
    seconds = end - start
    return result, seconds  # timedelta(seconds=end-start)


def plot_proj_grid(
    all_projections,
    figsize,
    save_experiment,
    out_plotgrid_path,
    scale_mode="normalize",
    padding=0.05,
    draw_colorbar=False,
):
    """
    scale_mode:
        - "normalize": rescale each dataset to [-1, 1] (fills panel; shapes preserved, absolute scale not).
        - "global":    same global limits for all datasets (absolute scale comparable; might look smaller).
        - "per_axes":  each subplot uses its own limits (fills panel; scales differ across subplots).
    padding: margin fraction inside each panel (only affects "normalize" and "per_axes").
    """

    if not all_projections:
        return None

    # --- grid layout ---
    n = len(all_projections)
    n_cols = min(4, math.ceil(math.sqrt(n)))
    n_rows = math.ceil(n / n_cols)

    # --- shared color scale across subplots ---
    vmin = min(np.min(item["targ"]) for item in all_projections)
    vmax = max(np.max(item["targ"]) for item in all_projections)

    # --- figure size scales with grid ---
    grid_figsize = (figsize[0] * n_cols, figsize[1] * n_rows)

    # Decide axis handling
    share_axes = scale_mode in ("normalize", "global")

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=grid_figsize,
        squeeze=False,
        sharex=share_axes,
        sharey=share_axes,
    )

    # Precompute global limits if needed
    if scale_mode == "global":
        xs_all = np.concatenate([item["proj"][:, 0] for item in all_projections])
        ys_all = np.concatenate([item["proj"][:, 1] for item in all_projections])
        xmin, xmax = xs_all.min(), xs_all.max()
        ymin, ymax = ys_all.min(), ys_all.max()
        xspan, yspan = xmax - xmin, ymax - ymin
        span = max(xspan, yspan)
        if span == 0:
            span = 1.0
        # small padding
        span *= 1 + 2 * padding
        xmid, ymid = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        global_xlim = (xmid - span / 2.0, xmid + span / 2.0)
        global_ylim = (ymid - span / 2.0, ymid + span / 2.0)

    last_scatter = None
    for idx, item in enumerate(all_projections):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        proj = item["proj"]
        targ = item["targ"]

        if scale_mode == "normalize":
            # Center + scale each dataset to [-1, 1] with padding
            xmin_i, xmax_i = proj[:, 0].min(), proj[:, 0].max()
            ymin_i, ymax_i = proj[:, 1].min(), proj[:, 1].max()
            cx, cy = 0.5 * (xmin_i + xmax_i), 0.5 * (ymin_i + ymax_i)
            span_i = max(xmax_i - xmin_i, ymax_i - ymin_i)
            if span_i == 0:
                span_i = 1.0
            scale = (1.0 - 2 * padding) * 2.0 / span_i  # map max span to 2 ([-1,1])
            X = (proj[:, 0] - cx) * scale
            Y = (proj[:, 1] - cy) * scale
            sc = ax.scatter(X, Y, s=1, c=targ, cmap="Spectral", vmin=vmin, vmax=vmax)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)

        elif scale_mode == "global":
            sc = ax.scatter(
                proj[:, 0],
                proj[:, 1],
                s=1,
                c=targ,
                cmap="Spectral",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)

        elif scale_mode == "per_axes":
            # Each dataset fills its own square panel (scales differ across subplots)
            xmin_i, xmax_i = proj[:, 0].min(), proj[:, 0].max()
            ymin_i, ymax_i = proj[:, 1].min(), proj[:, 1].max()
            cx, cy = 0.5 * (xmin_i + xmax_i), 0.5 * (ymin_i + ymax_i)
            span_i = max(xmax_i - xmin_i, ymax_i - ymin_i)
            if span_i == 0:
                span_i = 1.0
            # add padding
            span_i *= 1 + 2 * padding
            sc = ax.scatter(
                proj[:, 0],
                proj[:, 1],
                s=1,
                c=targ,
                cmap="Spectral",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlim(cx - span_i / 2.0, cx + span_i / 2.0)
            ax.set_ylim(cy - span_i / 2.0, cy + span_i / 2.0)

        else:
            raise ValueError(f"Unknown scale_mode: {scale_mode}")

        # Square panels, centered visuals, no ticks
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)
        else:
            ax.set_aspect("equal", adjustable="box")

        ax.set(xticks=[], yticks=[])
        ax.tick_params(
            axis="both", which="both", length=0, labelbottom=False, labelleft=False
        )
        ax.set_title(item.get("title", f"Dataset {idx}"), fontsize=9)

        last_scatter = sc

    # hide any unused axes
    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    if draw_colorbar:
        # one colorbar for all
        if last_scatter is not None:
            fig.colorbar(last_scatter, ax=axes.ravel().tolist(), shrink=0.85)

    fig.tight_layout()

    if save_experiment:
        fig.savefig(out_plotgrid_path, dpi=300, bbox_inches="tight")

    return plt


# ###----------------- voerview df builder helpers ---------------------##
def build_scores_overview(
    scores_by_dataset,
    acc_col="KNN_ACC",
    trust_col="trustworthiness",
    proj_time_col="proj_time",
    finch_time_col="finch_time",
    knn_label_suffix="-nn",
    knn_group_name="KNN accuracy",
    other_group_name="",  # top-level group for non-KNN columns ("" keeps it blank)
):
    """
    Build an overview table:
      rows   = dataset names
      cols   = ('KNN accuracy', '1-nn'), ('KNN accuracy', '3-nn'), ...,
               ('', trustworthiness), ('', proj_time), ('', finch_time)

    Notes
    -----
    - Columns use a MultiIndex so Pandas will show a grouped header.
    - df.to_csv(...) works; it writes a 2-row header.
    - If you want a single header row in the CSV, use `flatten_multiindex_for_csv(df)` below.
    """
    # normalize input -> list[(name, df)]
    if isinstance(scores_by_dataset, dict):
        items = list(scores_by_dataset.items())
    else:
        items = list(scores_by_dataset)

    # union of k-values across datasets (sorted numerically if possible)
    all_knns = []
    for _, df in items:
        ks = list(df.index.values)
        for k in ks:
            try:
                all_knns.append(int(k))
            except Exception:
                all_knns.append(k)

    if all(isinstance(k, (int, np.integer)) for k in all_knns):
        knn_values = sorted(set(all_knns))
    else:
        seen, knn_values = set(), []
        for k in all_knns:
            if k not in seen:
                seen.add(k)
                knn_values.append(k)

    def knn_col_name(k):
        if isinstance(k, (int, np.integer)) or (isinstance(k, str) and k.isdigit()):
            return f"{int(k)}{knn_label_suffix}"
        return f"{k}{knn_label_suffix}"

    knn_leaf_cols = [knn_col_name(k) for k in knn_values]

    # Build MultiIndex columns
    knn_cols = [(knn_group_name, c) for c in knn_leaf_cols]
    other_cols = [
        (other_group_name, trust_col),
        (other_group_name, proj_time_col),
        (other_group_name, finch_time_col),
    ]
    multi_cols = pd.MultiIndex.from_tuples(knn_cols + other_cols)

    # helper: first non-empty scalar
    def first_nonempty(series: pd.Series):
        for v in series:
            if pd.notna(v) and (not isinstance(v, str) or v.strip() != ""):
                return v
        return None

    rows = {}
    for ds_name, df in items:
        row = pd.Series(index=multi_cols, dtype=object)

        # map per-k accuracy into ('KNN accuracy', '<k>-nn')
        acc_series = df[acc_col]
        for k, leaf_name in zip(knn_values, knn_leaf_cols):
            val = None
            if k in df.index:
                val = acc_series.loc[k]
            else:
                candidates = [str(k)]
                try:
                    candidates.append(int(k))
                except Exception:
                    pass
                for cand in candidates:
                    if cand in df.index:
                        val = acc_series.loc[cand]
                        break
                if val is None:
                    try:
                        idx_as_int = df.index.astype(int)
                        pos = np.where(idx_as_int == int(k))[0]
                        if len(pos) > 0:
                            val = acc_series.iloc[pos[0]]
                    except Exception:
                        pass

            row[(knn_group_name, leaf_name)] = val

        # single-value fields
        row[(other_group_name, trust_col)] = first_nonempty(df[trust_col])
        row[(other_group_name, proj_time_col)] = first_nonempty(df[proj_time_col])
        row[(other_group_name, finch_time_col)] = first_nonempty(df[finch_time_col])

        rows[ds_name] = row

    overview = pd.DataFrame.from_dict(rows, orient="index")
    overview.index.name = "dataset"

    # Ensure column order is as defined
    overview = overview.reindex(columns=multi_cols)
    return overview


def flatten_multiindex_for_csv(df: pd.DataFrame, sep=" | "):
    """
    Optional: flatten MultiIndex columns to a single row header for CSV export.
    Example: ('KNN accuracy','1-nn') -> 'KNN accuracy | 1-nn'
             ('','trustworthiness')  -> 'trustworthiness'
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    flat_cols = []
    for tpl in df.columns:
        parts = [str(x) for x in tpl if x not in (None, "", " ")]
        flat_cols.append(parts[0] if len(parts) == 1 else sep.join(parts))
    out = df.copy()
    out.columns = flat_cols
    return out


# --- helpers from before (unchanged) ---
def _safe_read_overview_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
            top = df.columns.get_level_values(0)
            top = top.map(
                lambda x: "" if (pd.isna(x) or str(x).startswith("Unnamed")) else x
            )
            df.columns = pd.MultiIndex.from_arrays(
                [top, df.columns.get_level_values(1)]
            )
            return df
    except Exception:
        pass
    return pd.read_csv(path, header=0, index_col=0)


def _get_col(df: pd.DataFrame, key_tuple, extra_fallbacks=()):
    top, leaf = key_tuple
    if isinstance(df.columns, pd.MultiIndex) and key_tuple in df.columns:
        return df[key_tuple]
    candidates = []
    if top not in (None, "", " "):
        candidates.append(f"{top} | {leaf}")
        candidates.append(f"{top}.{leaf}")
    candidates.append(leaf)
    candidates += list(extra_fallbacks)
    for name in candidates:
        if name in df.columns:
            return df[name]
    return pd.Series(index=df.index, dtype=object)


# --- main combiner (with clean CSV header) ---
def combine_method_overviews_from_csv(
    method_csvs,
    output_csv_path,
    knn_cols=("1-nn", "5-nn"),
    trust_col="trustworthiness",
    proj_time_col="proj_time",
    knn_group_name="KNN accuracy",
    suppress_header_names=True,  # <- remove 'metric', 'method', 'dataset' row
    write_index=True,  # keep dataset names as first column
):
    # normalize input
    if isinstance(method_csvs, dict):
        items = list(method_csvs.items())
    else:
        items = list(method_csvs)

    # read all
    overviews_by_method = {name: _safe_read_overview_csv(path) for name, path in items}

    # union of dataset rows
    all_idx = None
    for df in overviews_by_method.values():
        idx = df.index if isinstance(df.index, pd.Index) else pd.Index(df.index)
        all_idx = idx if all_idx is None else all_idx.union(idx)
    all_idx = all_idx.sort_values()

    # build blocks: columns are (metric, method)
    data_blocks = {}
    metric_order = []

    for k in knn_cols:
        metric_name = f"KNN acc ({k})"
        if metric_name not in metric_order:
            metric_order.append(metric_name)
        for method_name, df in overviews_by_method.items():
            s = _get_col(df, (knn_group_name, k)).reindex(all_idx)
            data_blocks[(metric_name, method_name)] = s

    metric_order.append(trust_col)
    for method_name, df in overviews_by_method.items():
        s = _get_col(df, ("", trust_col)).reindex(all_idx)
        data_blocks[(trust_col, method_name)] = s

    metric_order.append(proj_time_col)
    for method_name, df in overviews_by_method.items():
        s = _get_col(df, ("", proj_time_col)).reindex(all_idx)
        data_blocks[(proj_time_col, method_name)] = s

    combined = pd.DataFrame(data_blocks, index=all_idx)
    combined.columns = pd.MultiIndex.from_tuples(
        combined.columns, names=["metric", "method"]
    )

    # order metrics/methods
    metric_cat = pd.CategoricalIndex(
        combined.columns.get_level_values("metric"),
        categories=metric_order,
        ordered=True,
    )
    method_cat = pd.CategoricalIndex(
        combined.columns.get_level_values("method"),
        categories=[name for name, _ in items],
        ordered=True,
    )
    combined = combined.loc[:, (metric_cat, method_cat)]

    # --- remove level/index names so CSV has no extra header row ---
    if suppress_header_names:
        combined.columns = pd.MultiIndex.from_tuples(
            combined.columns, names=[None, None]
        )
        combined.index.name = None

    # --- write CSV ---
    index_label = "" if (suppress_header_names and write_index) else None
    combined.to_csv(output_csv_path, index=write_index, index_label=index_label)

    return combined
