# ============================================================
# Hierarchical h-NNE v2 circle/sphere packing (multi-level)
# - 2D fast path with vectorized overlap resolution (grid pairs)
# - ND fallback (D>2) with adaptive safe solver (no 3^d meshgrid)
# - Children inside parent: robust center fit + inflate-radius, normalise_span
# - Iterative top->bottom packer for any number of levels
# ============================================================

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------- basic helpers (dimension-agnostic) ----------------


def _cluster_means_nd(X: np.ndarray, labels: np.ndarray):
    labels = np.asarray(labels)
    uniq, inv = np.unique(labels, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))
    means = np.zeros((len(uniq), X.shape[1]), float)
    np.add.at(means, inv, X)
    means = means / counts[:, None]
    return uniq, counts, means


def _base_radii_from_counts(counts: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.maximum(1.0, counts.astype(float)))
    med = float(np.median(r)) if np.any(r > 0) else 1.0
    return r / (med if med > 0 else 1.0)


# ========================= 2D FAST PATH =============================
# --------- Broadphase grid ----------------------- #


def _pack_cell_keys_2d(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Pack (cx, cy) into a single int64 key (works for negatives too)."""
    return (cx.astype(np.int64) << 32) ^ (cy.astype(np.int64) & 0xFFFFFFFF)


def _grid_pairs_2d_bigaware(
    P: np.ndarray,
    r: np.ndarray,
    cell_size: float,
    big_frac: float = 0.05,  # top 5% radii treated as “big”
):
    """
    Build unique candidate pairs with a quantile-sized grid and a widened
    neighborhood only for the few biggest discs. Vastly fewer pairs for
    heavy-tailed radii, but still detects all overlaps.
    """
    n = len(P)
    if n <= 1:
        return np.empty(0, int), np.empty(0, int)

    # Integer cell coords + packed keys
    cells = np.floor(P / cell_size).astype(np.int64)
    cx = cells[:, 0]
    cy = cells[:, 1]
    keys = _pack_cell_keys_2d(cx, cy)

    order = np.argsort(keys)
    keys_sorted = keys[order]
    cx_sorted = cx[order]
    cy_sorted = cy[order]

    uniq_keys, grp_starts, grp_counts = np.unique(
        keys_sorted, return_index=True, return_counts=True
    )
    gx = cx_sorted[grp_starts]
    gy = cy_sorted[grp_starts]

    def group_indices(gidx: int) -> np.ndarray:
        start = grp_starts[gidx]
        cnt = grp_counts[gidx]
        return order[start : start + cnt]

    # quick map for neighbor lookup
    def find_groups_for_keys(qkeys: np.ndarray):
        qkeys = np.asarray(qkeys, dtype=np.int64)
        pos = np.searchsorted(uniq_keys, qkeys)
        ok = pos < len(uniq_keys)
        if np.any(ok):
            eq = np.zeros_like(ok, dtype=bool)
            pos_ok = pos[ok]
            eq_ok = uniq_keys[pos_ok] == qkeys[ok]
            eq[ok] = eq_ok
            ok = ok & eq
        return ok, pos  # caller must index pos[ok]

    I_list = []
    J_list = []

    # Within-cell (upper triangle)
    for g in range(len(uniq_keys)):
        cnt = grp_counts[g]
        if cnt >= 2:
            idx = group_indices(g)
            iu, ju = np.triu_indices(cnt, k=1)
            I_list.append(idx[iu])
            J_list.append(idx[ju])

    # Normal discs with fixed 5-offsets
    OFFS = [(1, 0), (0, 1), (1, 1), (-1, 1)]
    if len(uniq_keys):
        for ox, oy in OFFS:
            nk = _pack_cell_keys_2d(gx + ox, gy + oy)
            ok, pos = find_groups_for_keys(nk)
            if not np.any(ok):
                continue
            src = np.nonzero(ok)[0]
            dst = pos[ok]
            for g_src, g_dst in zip(src, dst):
                ia = group_indices(g_src)
                ib = group_indices(g_dst)
                I_list.append(np.repeat(ia, len(ib)))
                J_list.append(np.tile(ib, len(ia)))

    # Big discs with expanded offsets (only for cells that contain big discs)
    k_big = max(1, int(np.ceil(big_frac * n)))
    thr = np.partition(r, -k_big)[-k_big]  # radius threshold
    is_big = r >= thr
    if np.any(is_big) and len(uniq_keys):
        # cells that contain at least one big disc
        big_cells = _pack_cell_keys_2d(cx[is_big], cy[is_big])
        big_cells = np.unique(big_cells)
        ok_big, pos_big = find_groups_for_keys(big_cells)
        for gpos in pos_big[ok_big]:
            idxs = group_indices(gpos)
            r_big_local = np.max(r[idxs])
            r_q90 = float(np.percentile(r, 90))
            Rcells = int(np.ceil((r_big_local + r_q90) / cell_size)) + 1
            cx0 = gx[gpos]
            cy0 = gy[gpos]
            for dx in range(-Rcells, Rcells + 1):
                for dy in range(-Rcells, Rcells + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nk = _pack_cell_keys_2d(np.array([cx0 + dx]), np.array([cy0 + dy]))
                    ok, pos = find_groups_for_keys(nk)
                    if not np.any(ok):
                        continue
                    g_dst = pos[ok][0]
                    ia = group_indices(gpos)
                    ib = group_indices(g_dst)
                    ia_big = ia[is_big[ia]]
                    if ia_big.size == 0:
                        continue
                    I_list.append(np.repeat(ia_big, len(ib)))
                    J_list.append(np.tile(ib, len(ia_big)))

    if not I_list:
        return np.empty(0, int), np.empty(0, int)

    I = np.concatenate(I_list)
    J = np.concatenate(J_list)
    # keep i<j, drop duplicates
    swap = I > J
    if np.any(swap):
        I, J = I.copy(), J.copy()
        I[swap], J[swap] = J[swap], I[swap]
    pairs = np.stack([I, J], axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs[:, 0], pairs[:, 1]


def _resolve_overlaps_allpairs_cap_2d(
    P: np.ndarray,
    r: np.ndarray,
    *,
    gap: float,
    ur: float = 0.7,  # under-relaxation (0.6–0.8)
    node_cap_frac: float = 0.9,  # per-node step cap vs worst incident "need"
    eps: float = 1e-9,
    max_sweeps: int = 8,
    recenter: bool = True,
    mass_mode: str = "area",
) -> np.ndarray:
    """
    Final exact pass: resolves any remaining overlaps using *all* pairs,
    center-only moves, per-node step caps (avoid dilation), and early exit.
    """
    P = np.asarray(P, float).copy()
    r = np.asarray(r, float)
    n = len(r)
    if n <= 1:
        return P

    inv_m = 1.0 / ((r**2 if mass_mode == "area" else r) + 1e-12)
    iu, ju = np.triu_indices(n, 1)  # reuse same list each sweep

    for _ in range(max_sweeps):
        dvec = P[iu] - P[ju]  # (M,2)
        dist = np.linalg.norm(dvec, axis=1)  # (M,)
        need = (r[iu] + r[ju] + gap) - dist  # (M,)

        mask = need > eps
        if not np.any(mask):
            break

        im = iu[mask]
        jm = ju[mask]
        dvm = dvec[mask]
        distm = dist[mask]
        needm = need[mask]

        u = np.zeros_like(dvm)
        nz = distm >= 1e-12
        u[nz] = dvm[nz] / distm[nz][:, None]
        if np.any(~nz):  # coincident fallback
            u[~nz, 0] = 1.0

        move = ur * (needm + eps)
        wi, wj = inv_m[im], inv_m[jm]
        ws = wi + wj
        ti = wi / (ws + 1e-12)
        tj = wj / (ws + 1e-12)

        dPi = (ti * move)[:, None] * u
        dPj = -(tj * move)[:, None] * u

        # accumulate node increments
        add_x = np.bincount(im, weights=dPi[:, 0], minlength=n) + np.bincount(
            jm, weights=dPj[:, 0], minlength=n
        )
        add_y = np.bincount(im, weights=dPi[:, 1], minlength=n) + np.bincount(
            jm, weights=dPj[:, 1], minlength=n
        )

        # node-wise cap vs worst incident need
        worst_need = np.zeros(n, float)
        np.maximum.at(worst_need, im, needm)
        np.maximum.at(worst_need, jm, needm)
        cap = node_cap_frac * worst_need

        step_norm = np.hypot(add_x, add_y)
        scale = np.ones(n, float)
        ok = step_norm > (cap + 1e-12)
        if np.any(ok):
            scale[ok] = cap[ok] / (step_norm[ok] + 1e-12)

        P[:, 0] += add_x * scale
        P[:, 1] += add_y * scale

        if recenter:
            P -= P.mean(axis=0, keepdims=True)

    return P


#######-------------  Capped Jacobii Solver ------------#########
def separate_overlaps_minmove_grid_safe_2d(
    P0: np.ndarray,
    radii: Sequence[float],
    tol: float = 1e-9,
    mass_mode: str = "area",
    cell_size: Optional[float] = None,
    max_sweeps: int = 200,
    pair_gap: Optional[float] = 0.0,
    # speed/quality knobs (sensible defaults)
    ur: float = 0.7,  # under-relaxation
    node_cap_frac: float = 0.85,  # per-node step cap (fraction of worst "need")
    inner_iters: int = 3,  # reuse the same pairs this many times
    coarse_sweeps: int = 3,
    fine_sweeps: int = 8,
    recenter_each_sweep: bool = True,
) -> np.ndarray:
    """
    Grid broadphase + vectorized impulses with per-node step capping.
    Two-stage (coarse→fine) + inner micro-iterations to avoid rebuilding the grid
    every time. Compact like GS, fast like Jacobi.
    """
    P = np.asarray(P0, float).copy()
    r = np.asarray(radii, float)
    n = len(r)
    if n <= 1:
        return P

    gap = 0.0 if pair_gap is None else float(pair_gap)

    # masses
    inv_m = 1.0 / ((r**2 if mass_mode == "area" else r) + 1e-12)

    # relative tolerance so we don't chase ultra-tiny overlaps
    med_r = float(np.median(r[r > 0])) if np.any(r > 0) else 1.0
    eps = max(tol, 1e-6 * med_r)

    # base cell size (quantile-based keeps cells nice for the bulk)
    if cell_size is None:
        rp = r[r > 0]
        q90 = float(np.percentile(rp, 90)) if rp.size else 1.0
        med = float(np.median(rp)) if rp.size else 1.0
        base = max(2.0 * q90, 4.0 * med)
        cell0 = max(base, 1e-6)
    else:
        cell0 = max(float(cell_size), 1e-12)

    def _pairs_with_cell(P_, cs_):
        return _grid_pairs_2d_bigaware(P_, r, cs_)

    def _apply_batch(I, J, ur_local):
        dvec = P[I] - P[J]
        dist = np.linalg.norm(dvec, axis=1)
        need = (r[I] + r[J] + gap) - dist
        mask = need > eps
        if not np.any(mask):
            return 0, 0.0
        I2 = I[mask]
        J2 = J[mask]
        dvec = dvec[mask]
        dist = dist[mask]
        need = need[mask]

        u = np.zeros_like(dvec)
        nz = dist >= 1e-12
        u[nz] = dvec[nz] / dist[nz][:, None]
        if np.any(~nz):
            u[~nz, 0] = 1.0

        move = ur_local * (need + eps)
        wi, wj = inv_m[I2], inv_m[J2]
        ws = wi + wj
        ti = wi / (ws + 1e-12)
        tj = wj / (ws + 1e-12)

        dPi = (ti * move)[:, None] * u
        dPj = -(tj * move)[:, None] * u

        # accumulate node increments
        add_x = np.bincount(I2, weights=dPi[:, 0], minlength=n) + np.bincount(
            J2, weights=dPj[:, 0], minlength=n
        )
        add_y = np.bincount(I2, weights=dPi[:, 1], minlength=n) + np.bincount(
            J2, weights=dPj[:, 1], minlength=n
        )

        # node-wise cap
        worst_need = np.zeros(n, float)
        np.maximum.at(worst_need, I2, need)
        np.maximum.at(worst_need, J2, need)
        cap = node_cap_frac * worst_need
        step_norm = np.hypot(add_x, add_y)
        scale = np.ones(n, float)
        ok = step_norm > (cap + 1e-12)
        if np.any(ok):
            scale[ok] = cap[ok] / (step_norm[ok] + 1e-12)

        P[:, 0] += add_x * scale
        P[:, 1] += add_y * scale

        return np.count_nonzero(mask), float(np.max(need))

    # two stages: coarse (fewer pairs) then fine
    stages = [(1.6 * cell0, coarse_sweeps), (1.0 * cell0, fine_sweeps)]
    sweeps_done = 0

    for cs, sweeps in stages:
        for _ in range(int(sweeps)):
            sweeps_done += 1
            I, J = _pairs_with_cell(P, cs)
            M = I.size
            if M == 0:
                break

            total_over, max_need = 0, 0.0
            for _inner in range(max(1, int(inner_iters))):
                over_cnt, max_nd = _apply_batch(I, J, ur)
                total_over += over_cnt
                max_need = max(max_need, max_nd)
                if over_cnt == 0:
                    break

            if recenter_each_sweep:
                P -= P.mean(axis=0, keepdims=True)

            frac = total_over / max(M, 1)
            if (frac < 0.01) or (max_need < 3.0 * eps):
                break

            if sweeps_done >= max_sweeps:
                break
        if sweeps_done >= max_sweeps:
            break

    # final tiny polish (single all-pairs impulse) only if needed
    iu, ju = np.triu_indices(n, 1)
    d = np.linalg.norm(P[iu] - P[ju], axis=1)
    need_final = (r[iu] + r[ju] + gap) - d
    if np.any(need_final > eps):
        im = iu[need_final > eps]
        jm = ju[need_final > eps]
        dvec = P[im] - P[jm]
        dist = np.linalg.norm(dvec, axis=1)
        u = np.zeros_like(dvec)
        nz = dist >= 1e-12
        u[nz] = dvec[nz] / dist[nz][:, None]
        if np.any(~nz):
            u[~nz, 0] = 1.0
        wi, wj = inv_m[im], inv_m[jm]
        ws = wi + wj
        ti, tj = wi / (ws + 1e-12), wj / (ws + 1e-12)
        step = need_final[need_final > eps] + eps
        dPi = (ti * step)[:, None] * u
        dPj = -(tj * step)[:, None] * u
        np.add.at(P, im, dPi)
        np.add.at(P, jm, dPj)
        if recenter_each_sweep:
            P -= P.mean(axis=0, keepdims=True)

    # ----- final exact polish (all-pairs, center-only, capped) -----
    med_r = float(np.median(r[r > 0])) if np.any(r > 0) else 0.0
    gap_final = max(float(pair_gap or 0.0), 0.95 * med_r)
    eps_final = max(eps, 1e-6 * (med_r if med_r > 0 else 1.0))

    iu, ju = np.triu_indices(n, 1)
    d = np.linalg.norm(P[iu] - P[ju], axis=1)
    need = (r[iu] + r[ju] + gap_final) - d
    if np.any(need > eps_final):
        P = _resolve_overlaps_allpairs_cap_2d(
            P,
            r,
            gap=gap_final,
            ur=0.7,
            node_cap_frac=0.85,
            eps=eps_final,
            max_sweeps=80,
            recenter=True,
            mass_mode=mass_mode,
        )

    return P


# -- 2D kNN edges / scaling / alignment utilities --


def _knn_edges_2d(A: np.ndarray, k: int):
    n = len(A)
    if n <= 1:
        return np.empty((0, 2), int), np.empty((0,), float)
    k = int(np.clip(k, 1, n - 1))
    diff = A[:, None, :] - A[None, :, :]
    D2 = np.einsum("...i,...i->...", diff, diff)
    neigh = []
    for i in range(n):
        idx = np.argpartition(D2[i], kth=min(k, n - 1))[: k + 1]
        idx = idx[idx != i]
        if len(idx) > k:
            idx = idx[np.argsort(D2[i, idx])[:k]]
        neigh.append(set(idx.tolist()))
    edges = set()
    for i in range(n):
        for j in neigh[i]:
            if i in neigh[j]:
                a, b = (i, j) if i < j else (j, i)
                edges.add((a, b))
    if not edges:
        return np.empty((0, 2), int), np.empty((0,), float)
    E = np.array(sorted(edges), int)
    d = np.sqrt(np.maximum(0.0, D2[E[:, 0], E[:, 1]]))
    return E, d


def _scale_anchors_touch_2d(A: np.ndarray, r: np.ndarray):
    n = len(r)
    if n < 2:
        return A.copy()
    i, j = np.triu_indices(n, 1)
    touch_med = float(np.median(r[i] + r[j]))
    ai, aj = np.triu_indices(len(A), 1)
    medA = float(np.median(np.linalg.norm(A[ai] - A[aj], axis=1)))
    return A if medA <= 0 else A * (touch_med / medA)


def _auto_edge_targets(A: np.ndarray, r: np.ndarray, k: int):
    """
    Build kNN edges from anchors and return relaxed/capped targets L that
    avoid huge layouts automatically (no user params).
    """
    E, a_len = _knn_edges_2d(A, k=max(1, min(k, len(A) - 1)))
    if E.size == 0:
        return E, a_len  # trivial

    touch = r[E[:, 0]] + r[E[:, 1]]
    ratio = a_len / np.maximum(touch, 1e-12)

    med_ratio = float(np.median(ratio))
    p90_ratio = float(np.percentile(ratio, 90))

    cap_mult = 1.50
    if med_ratio > 2.0:
        cap_mult = 1.35
    if med_ratio > 4.0 or p90_ratio > 6.0:
        cap_mult = 1.25
    if med_ratio > 8.0 or p90_ratio > 12.0:
        cap_mult = 1.18

    edge_relax = 0.30 if med_ratio <= 2.0 else 0.18
    if med_ratio > 4.0 or p90_ratio > 6.0:
        edge_relax = 0.10
    if med_ratio > 8.0 or p90_ratio > 12.0:
        edge_relax = 0.05

    L_target = (1.0 - edge_relax) * touch + edge_relax * a_len
    L = np.minimum(L_target, cap_mult * touch)
    L = np.maximum(L, touch)
    return E, L


def _first_auto_scale(
    P: np.ndarray,
    r: np.ndarray,
    k: int,
    max_sweeps: int = 120,
    pair_gap: Optional = None,
) -> np.ndarray:
    if len(P) <= 1:
        return P
    E, dpos = _knn_edges_2d(P, k=max(1, min(k, len(P) - 1)))
    if E.size == 0:
        return P
    touch = r[E[:, 0]] + r[E[:, 1]]
    med_pos = float(np.median(dpos))
    med_touch = float(np.median(touch))
    if med_touch <= 1e-12:
        return P
    target = 1.2 * med_touch
    if med_pos <= target:
        return P
    s = med_pos / target
    c = P.mean(axis=0, keepdims=True)
    P = (P - c) / s + c
    # P = P / s
    if pair_gap is None:
        pair_gap = 0.01 * float(np.median(r)) if np.any(r > 0) else 0.0

    P = separate_overlaps_minmove_grid_safe_2d(
        P,
        r,
        tol=1e-9,
        mass_mode="area",
        cell_size=None,
        pair_gap=pair_gap,
        max_sweeps=max_sweeps,
    )
    return P


def _align_to_anchors_2d(P: np.ndarray, A: np.ndarray) -> np.ndarray:
    P0 = P - P.mean(axis=0)
    A0 = A - A.mean(axis=0)
    H = P0.T @ A0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return (P0 @ R) + A.mean(axis=0)


# ----------------------------------------------------------------#


def _separate_centers_allpairs_2d(
    P: np.ndarray,
    r: np.ndarray,
    pair_gap: float = 0.0,
    tol: float = 1e-9,
    max_sweeps: int = 60,
    mass_mode: str = "area",
) -> np.ndarray:
    """
    Move centers only (no radii change) until all pairs satisfy
        ||p_i - p_j|| >= r_i + r_j + pair_gap.
    Robust all-pairs pass (top level).
    """
    P = np.asarray(P, float).copy()
    r = np.asarray(r, float)
    n = len(r)
    if n <= 1:
        return P

    inv_m = 1.0 / ((r**2 if mass_mode == "area" else r) + 1e-12)
    iu, ju = np.triu_indices(n, k=1)

    for _ in range(max_sweeps):
        dvec = P[iu] - P[ju]
        dist = np.linalg.norm(dvec, axis=1)
        need = (r[iu] + r[ju] + pair_gap) - dist
        mask = need > tol
        if not np.any(mask):
            break

        im, jm = iu[mask], ju[mask]
        dvm, distm, needm = dvec[mask], dist[mask], need[mask]

        u = np.zeros_like(dvm)
        nz = distm >= 1e-12
        u[nz] = dvm[nz] / distm[nz][:, None]
        if np.any(~nz):
            u[~nz, 0] = 1.0

        wi, wj = inv_m[im], inv_m[jm]
        ws = wi + wj
        ti = wi / (ws + 1e-12)
        tj = wj / (ws + 1e-12)

        step = needm + tol
        dPi = (ti * step)[:, None] * u
        dPj = -(tj * step)[:, None] * u
        np.add.at(P, im, dPi)
        np.add.at(P, jm, dPj)

    return P


def _normalize_top_level_span_2d(
    P: np.ndarray,
    r: np.ndarray,
    k: int,
    max_sweeps: int = 120,
    beta: float = 1.25,
    gap_frac: float = 0.01,
    mode: str = "auto",  # "auto", "allpairs", "grid"
    n_threshold_allpairs: int = 1000,
) -> np.ndarray:
    """
    Uniformly scale centers (about the centroid) so that
    median kNN edge length ≈ beta * median touching distance.
    Radii are not changed. Uses all-pairs separation before/after.
    """
    gap = gap_frac * float(np.median(r)) if np.any(r > 0) else 0.0

    P = np.asarray(P, float)
    r = np.asarray(r, float)
    n = len(r)
    if n <= 1:
        return P
    if n < n_threshold_allpairs:
        P = _separate_centers_allpairs_2d(P, r, pair_gap=gap, max_sweeps=10)

    E, dpos = _knn_edges_2d(P, k=max(1, min(k, n - 1)))
    if E.size == 0:
        return P
    touch = r[E[:, 0]] + r[E[:, 1]]
    med_pos = float(np.median(dpos))
    med_touch = float(np.median(touch))
    if med_touch <= 1e-12:
        return P

    target = beta * med_touch
    s = med_pos / target
    if s <= 1.02:
        return P

    c = P.mean(axis=0, keepdims=True)
    P = (P - c) / s + c

    if mode == "auto":
        mode = "allpairs" if n <= n_threshold_allpairs else "grid"
    if mode == "allpairs":
        return _separate_centers_allpairs_2d(P, r, pair_gap=gap, max_sweeps=50)
    elif mode == "grid":
        return separate_overlaps_minmove_grid_safe_2d(
            P, r, tol=1e-9, cell_size=None, pair_gap=gap, max_sweeps=80
        )
    else:
        return P


# -----------Main 2D packing--------------#


def pack_circles_from_anchors_2d_spacefill_k(
    anchors: np.ndarray,
    radii: Sequence[float],
    k: int,
    tol: float = 1e-9,
    mass_mode: str = "area",
    outer_sweeps: int = 1,
    pair_gap: Optional[float] = None,
    grid_cell_size: Optional[float] = None,
    auto_scale: bool = True,
    post_normalize: bool = False,
) -> np.ndarray:
    """
    Fast 2D packer: preserves local relations via capped/relaxed kNN targets
    and auto-compacts when anchors are very stretched (no user knobs needed).
    """
    A = np.asarray(anchors, float)
    r = np.asarray(radii, float)
    n = len(r)

    # --- automatic, size-aware anchor scaling (keeps deep levels dense) ---
    if auto_scale:
        A_sc = A
        P = _first_auto_scale(A_sc, r, k=8, pair_gap=pair_gap, max_sweeps=160)
    else:
        A_sc = _scale_anchors_touch_2d(A, r)
        P = separate_overlaps_minmove_grid_safe_2d(
            A_sc,
            r,
            tol=tol,
            mass_mode=mass_mode,
            cell_size=grid_cell_size,
            pair_gap=pair_gap,
            max_sweeps=160,
        )

    # Auto edge targets (relax/cap) from anchors
    E, L = _auto_edge_targets(A_sc, r, k=k if n > 1 else 1)
    if E.size == 0:
        return P

    # Project along kNN edges toward those targets
    for _ in range(max(1, outer_sweeps)):
        i = E[:, 0]
        j = E[:, 1]
        dij = P[i] - P[j]
        dist = np.linalg.norm(dij, axis=1)
        err = dist - L

        mask = np.abs(err) > tol
        if np.any(mask):
            i_m, j_m = i[mask], j[mask]
            dij_m, dist_m, err_m = dij[mask], dist[mask], err[mask]
            u = np.zeros_like(dij_m)
            z = dist_m < 1e-12
            u[z] = np.array([1.0, 0.0])
            u[~z] = dij_m[~z] / dist_m[~z, None]

            inv_m = 1.0 / ((r**2 if mass_mode == "area" else r) + 1e-12)
            wi, wj = inv_m[i_m], inv_m[j_m]
            ws = wi + wj
            ti = wi / (ws + 1e-12)
            tj = wj / (ws + 1e-12)

            dPi = -ti[:, None] * err_m[:, None] * u
            dPj = tj[:, None] * err_m[:, None] * u
            np.add.at(P, i_m, dPi)
            np.add.at(P, j_m, dPj)

        # clean overlaps each pass
        P = separate_overlaps_minmove_grid_safe_2d(
            P,
            r,
            tol=tol,
            mass_mode=mass_mode,
            cell_size=grid_cell_size,
            pair_gap=pair_gap,
            max_sweeps=160,
        )
        # rigid align (no scaling)
        P = _align_to_anchors_2d(P, A_sc)

        # Normalize top-level span (compact, tiny gap)
        # start = time.time()
        if post_normalize:
            P = _normalize_top_level_span_2d(
                P, r, k=8, beta=1.25, gap_frac=0.01, n_threshold_allpairs=1000
            )
    return P


# ---------- max radii given fixed centers (fast, dense, overlap-free) ----------


def _pack_children_inside_parent_dense_2d(
    A_local: np.ndarray,
    r_local: np.ndarray,
    parent_center: np.ndarray,
    parent_radius: float,
    k_local: int,
    outer_sweeps: int,
    mass_mode: str,
):
    m = len(r_local)
    if m == 0:
        return np.empty((0, 2), float), np.empty((0,), float)
    if m == 1:
        return parent_center[None, :], np.array([0.99 * parent_radius], float)

    # ---------#
    # 1) local kNN pack (preserve relations)
    P_local = pack_circles_from_anchors_2d_spacefill_k(
        A_local,
        r_local,
        k=min(k_local, max(1, m - 1)),
        tol=1e-9,
        mass_mode=mass_mode,
        outer_sweeps=outer_sweeps,
        pair_gap=None,
        post_normalize=False,
    )

    if P_local.ndim == 1:
        P_local = P_local.reshape(-1, 2)
    if P_local.shape[0] != m:
        P_local = np.resize(P_local, (m, 2))

    # 2) robust fit of centers only (no radius scaling)
    P_fit, r_local = _fit_children_inside_parent_nd(
        P_local, r_local, parent_center, parent_radius, fill_frac=0.99
    )

    P_rel = P_fit - parent_center

    P_rel, r_local = _inflate_radii_balloon(
        P_rel, r_local, parent_radius, rate=0.25, steps=5
    )  # pair_gap=0.01*np.median(r_local)
    P_rel = _normalize_top_level_span_2d(
        P_rel, r_local, k=5, beta=1.25, gap_frac=0.01, n_threshold_allpairs=1000
    )

    if P_rel.ndim == 1:
        P_rel = P_rel.reshape(-1, 2)
    if P_rel.shape[0] != m:
        P_rel = np.resize(P_rel, (m, 2))

    P_final = P_rel + parent_center
    return P_final, r_local


# ---------------------------------------------------------------

# ========================= ND (D>2) PATH ============================

# ---------------------------------------------------------------


def separate_overlaps_minmove_grid_safe_nd(
    P0: np.ndarray,
    radii: Sequence[float],
    tol: float = 1e-9,
    mass_mode: str = "powerD",
    cell_size: Optional[float] = None,
    max_sweeps: int = 600,
    pair_gap: Optional[float] = None,
) -> np.ndarray:
    """
    Resolve overlaps among balls in R^d.
    - d <= 6: grid broadphase + 3^d neighborhood (generated lazily)
    - d >  6: vectorized all-pairs (no meshgrid; O(n^2) per sweep)
    """
    P = np.asarray(P0, float).copy()
    r = np.asarray(radii, float)
    n, d = P.shape
    if n <= 1:
        return P

    # masses
    if mass_mode == "powerD":
        inv_m = 1.0 / (np.power(r, d) + 1e-12)
    elif mass_mode == "radius":
        inv_m = 1.0 / (r + 1e-12)
    else:
        inv_m = np.ones_like(r)

    if pair_gap is None:
        mr = float(np.median(r)) if np.any(r > 0) else 0.0
        pair_gap = 0.01 * mr

    if d > 6:
        iu, ju = np.triu_indices(n, 1)
        for _ in range(max_sweeps):
            dvec = P[iu] - P[ju]
            dist = np.linalg.norm(dvec, axis=1)
            need = (r[iu] + r[ju] + pair_gap) - dist
            mask = need > tol
            if not np.any(mask):
                break
            im = iu[mask]
            jm = ju[mask]
            dvecm = dvec[mask]
            distm = dist[mask]
            needm = need[mask]
            u = np.zeros_like(dvecm)
            nz = distm >= 1e-12
            u[nz] = dvecm[nz] / distm[nz][:, None]
            if np.any(~nz):
                u[~nz, 0] = 1.0
            move = needm + tol
            wi, wj = inv_m[im], inv_m[jm]
            ws = wi + wj
            ti, tj = wi / (ws + 1e-12), wj / (ws + 1e-12)
            dPi = ti[:, None] * move[:, None] * u
            dPj = -tj[:, None] * move[:, None] * u
            np.add.at(P, im, dPi)
            np.add.at(P, jm, dPj)
        return P

    # low/medium dims: grid broadphase with small 3^d neighborhood
    from itertools import product

    if cell_size is None:
        rp = r[r > 0]
        base = max(
            2.0 * (float(np.max(rp)) if rp.size else 1.0),
            4.0 * (float(np.median(rp)) if rp.size else 0.5),
        )
        cell_size = max(base, 1e-6)
    else:
        cell_size = max(float(cell_size), 1e-12)

    offsets = list(product((-1, 0, 1), repeat=d))  # small for d<=6

    for _ in range(max_sweeps):
        max_gap = 0.0
        grid: Dict[Tuple[int, ...], List[int]] = {}
        cells = np.floor(P / cell_size).astype(np.int64)

        for i in range(n):
            key = tuple(int(x) for x in cells[i])
            grid.setdefault(key, []).append(i)

        for i in range(n):
            base = cells[i]
            for off in offsets:
                key = tuple(int(base[k] + off[k]) for k in range(d))
                bucket = grid.get(key)
                if not bucket:
                    continue
                for j in bucket:
                    if j <= i:
                        continue
                    dvec = P[i] - P[j]
                    dist = float(np.linalg.norm(dvec))
                    need = (r[i] + r[j] + pair_gap) - dist
                    if need <= tol:
                        continue
                    max_gap = max(max_gap, need)
                    if dist < 1e-12:
                        u = np.zeros(d)
                        u[0] = 1.0
                    else:
                        u = dvec / dist
                    move = need + tol
                    wi, wj = inv_m[i], inv_m[j]
                    ws = wi + wj
                    ti, tj = wi / (ws + 1e-12), wj / (ws + 1e-12)
                    P[i] += ti * move * u
                    P[j] -= tj * move * u

        if max_gap <= tol:
            break

    return P


def pack_balls_from_anchors_nd_spacefill_k(
    anchors: np.ndarray,
    radii: Sequence[float],
    k: int,
    tol: float = 1e-9,
    mass_mode: str = "powerD",
    outer_sweeps: int = 1,
) -> np.ndarray:
    """
    Light ND packer (used only when D>2): resolve overlaps; no heavy edge projections.
    """
    A = np.asarray(anchors, float)
    r = np.asarray(radii, float)
    P = separate_overlaps_minmove_grid_safe_nd(
        A, r, tol=tol, mass_mode=mass_mode, max_sweeps=240
    )
    for _ in range(max(1, outer_sweeps)):
        P = separate_overlaps_minmove_grid_safe_nd(
            P, r, tol=tol, mass_mode=mass_mode, max_sweeps=240
        )
    return P


def _fit_children_inside_parent_nd(
    P_child: np.ndarray,
    r_child: np.ndarray,
    parent_center: np.ndarray,
    parent_radius: float,
    fill_frac: float = 0.98,
):

    if len(r_child) == 0 or parent_radius <= 0:
        return np.empty((0, P_child.shape[1]), float), np.empty((0,), float)
    c = P_child.mean(axis=0)
    ext = float(np.max(np.linalg.norm(P_child - c, axis=1) + r_child))
    s = (
        (fill_frac * parent_radius / ext)
        if ext > 1e-12
        else (fill_frac * parent_radius)
    )
    return parent_center + s * (P_child - c), s * r_child


def _pack_children_inside_parent_dense_nd(
    A_local: np.ndarray,
    r_local: np.ndarray,
    parent_center: np.ndarray,
    parent_radius: float,
    k_local: int,
    outer_sweeps: int,
    mass_mode: str,
):
    """
    ND fallback (used only when D>2): local light pack -> fit -> one light separation.
    """
    P_local = pack_balls_from_anchors_nd_spacefill_k(
        A_local,
        r_local,
        k=k_local,
        tol=1e-9,
        mass_mode=mass_mode,
        outer_sweeps=outer_sweeps,
    )
    P_fit, r_fit = _fit_children_inside_parent_nd(
        P_local, r_local, parent_center, parent_radius, fill_frac=0.98
    )
    P_rel = P_fit - parent_center
    P_rel = separate_overlaps_minmove_grid_safe_nd(
        P_rel, r_fit, tol=1e-9, mass_mode=mass_mode, max_sweeps=200, pair_gap=0.0
    )
    return parent_center + P_rel, r_fit


# ------------------------------------


def _inflate_radii_balloon(
    P_rel: np.ndarray,  # centers relative to parent center, (m,2)
    r: np.ndarray,  # radii, (m,)
    R: float,  # parent radius
    steps: int = 24,
    rate: float = 0.55,  # growth fraction of available slack per step (0..1)
    pair_gap: float = 0.0,  # keep 0.0 for max fill
    wall_gap: float = 0.0,  # keep 0.0 for max fill
    tol_stop: float = 1e-6,  # stop if total growth is tiny
) -> tuple[np.ndarray, np.ndarray]:
    """
    Non-uniform 'balloon' inflation: grows radii toward the maximum feasible values
    while keeping centers inside the parent (radius R) and resolving overlaps after
    each growth step. Returns (P_rel_new, r_new).
    """
    P = np.asarray(P_rel, float).copy()
    r = np.asarray(r, float).copy()
    m = len(r)
    if m == 0 or R <= 0:
        return P, r

    for _ in range(steps):
        # --- compute slack limits ---
        rho = np.linalg.norm(P, axis=1)  # distance to parent center
        wall_slack = (
            (R - wall_gap) - rho - r
        )  # <= how much each can grow w.r.t. boundary
        wall_slack = np.maximum(0.0, wall_slack)

        if m >= 2:
            diff = P[:, None, :] - P[None, :, :]
            D = np.linalg.norm(diff, axis=2)  # pairwise center distances
            # for i, max extra before touching j: D_ij - pair_gap - (r_i + r_j)
            pair_slack_mat = D - pair_gap - (r[:, None] + r[None, :])
            np.fill_diagonal(pair_slack_mat, np.inf)  # ignore self
            pair_slack = np.min(pair_slack_mat, axis=1)
            pair_slack = np.maximum(0.0, pair_slack)
        else:
            pair_slack = wall_slack.copy()

        # choose the tightest constraint per circle
        slack = np.minimum(wall_slack, pair_slack)

        # growth (favor bigger circles slightly by squaring slack weight)
        grow = rate * slack
        total_growth = float(np.sum(grow))
        if total_growth <= tol_stop:
            break
        r += grow

        # --- resolve overlaps after growth ---
        P = separate_overlaps_minmove_grid_safe_2d(
            P,
            r,
            tol=1e-9,
            mass_mode="area",
            cell_size=None,
            pair_gap=0.0,
            max_sweeps=160,
        )

        # --- clamp to parent boundary (if moved out by separation) ---
        rho = np.linalg.norm(P, axis=1)
        max_rho = np.maximum(0.0, (R - wall_gap) - r)
        # scale any offenders back onto the valid disk
        mask = rho > (max_rho + 1e-12)
        if np.any(mask):
            P[mask] *= (max_rho[mask] / rho[mask])[:, None]

        # recentre to avoid drift
        P -= P.mean(axis=0)

    # final tight clean
    P = separate_overlaps_minmove_grid_safe_2d(
        P, r, tol=1e-9, mass_mode="area", cell_size=None, pair_gap=0.0, max_sweeps=100
    )
    rho = np.linalg.norm(P, axis=1)
    max_rho = np.maximum(0.0, (R - wall_gap) - r)
    mask = rho > (max_rho + 1e-12)
    if np.any(mask):
        P[mask] *= (max_rho[mask] / rho[mask])[:, None]
    P -= P.mean(axis=0)

    return P, r


# =================== ITERATIVE MULTI-LEVEL PACKER ====================


def pack_hierarchy_iterative_k_nd(
    X_nd: np.ndarray,  # (N,D)
    partitions: np.ndarray,  # (N,L), top at partitions[:, -1]
    k: int = 1,
    outer_sweeps: int = 1,
    mass_mode_2d: str = "area",  # used when D=2
    mass_mode_nd: str = "powerD",  # used when D>2
) -> Dict[Tuple[int, int], Tuple[float, ...]]:
    """
    Multi-level iterative packer (top -> bottom).
    - D=2 -> fast 2D path (vectorized overlaps; dense inside-parent)
    - D>2 -> ND fallback (lighter)
    Returns layout with keys (level_index_in_partitions, label_at_that_level)
    and values (...coords..., radius).
    """
    X = np.asarray(X_nd, float)
    parts = np.asarray(partitions)
    assert X.ndim == 2 and parts.ndim == 2 and X.shape[0] == parts.shape[0]
    N, D = X.shape
    L = parts.shape[1]

    layout: Dict[Tuple[int, int], Tuple[float, ...]] = {}

    # ---- Top level ----
    lev = L - 1
    lab_top, cnt_top, anc_top = _cluster_means_nd(X, parts[:, lev])
    rad_top = _base_radii_from_counts(cnt_top)

    if D == 2:
        # Pack with sibling-only edges (no parent confinement)
        P_top = pack_circles_from_anchors_2d_spacefill_k(
            anc_top,
            rad_top,
            k=k,
            tol=1e-9,
            mass_mode=mass_mode_2d,
            outer_sweeps=outer_sweeps,
            pair_gap=None,
            auto_scale=True,
            post_normalize=True,
        )
    else:
        P_top = pack_balls_from_anchors_nd_spacefill_k(
            anc_top,
            rad_top,
            k=k,
            tol=1e-9,
            mass_mode=mass_mode_nd,
            outer_sweeps=outer_sweeps,
        )

    top_pos_map = {int(l): P_top[i] for i, l in enumerate(lab_top)}
    top_rad_map = {int(l): float(rad_top[i]) for i, l in enumerate(lab_top)}
    for i, l in enumerate(lab_top):
        layout[(lev, int(l))] = tuple(P_top[i].tolist() + [float(rad_top[i])])

    # ---- Descend levels ----
    for lev in range(L - 2, -1, -1):
        labels_child = parts[:, lev]
        labels_parent = parts[:, lev + 1]

        lab_child, cnt_child, anc_child = _cluster_means_nd(X, labels_child)
        rad_child_base = _base_radii_from_counts(cnt_child)

        # strict child->parent map
        c2p = {}
        child = labels_child
        parent = labels_parent
        for c in np.unique(child):
            ps = np.unique(parent[child == c])
            if len(ps) != 1:
                raise ValueError(
                    f"Inconsistent hierarchy at levels {lev}->{lev+1}: child {int(c)} -> {ps.tolist()}"
                )
            c2p[int(c)] = int(ps[0])

        # children grouped by parent (indices in lab_child space)
        parent_to_children_idx: Dict[int, List[int]] = {}
        idx_map = {int(l): i for i, l in enumerate(lab_child)}
        for c_lab in lab_child:
            parent_to_children_idx.setdefault(c2p[int(c_lab)], []).append(
                idx_map[int(c_lab)]
            )

        pos_child_global = np.zeros((len(lab_child), D), float)
        rad_child_global = np.zeros((len(lab_child),), float)

        for p_lab, child_idx_list in parent_to_children_idx.items():
            child_idx = np.array(child_idx_list, dtype=int)
            p_center = np.asarray(top_pos_map[int(p_lab)], float)
            p_radius = float(top_rad_map[int(p_lab)])

            A_local = anc_child[child_idx]
            r_local = rad_child_base[child_idx]

            if len(child_idx) == 1:
                pos_child_global[child_idx] = p_center[None, :]
                rad_child_global[child_idx] = np.array([0.99 * p_radius], float)
            else:
                if D == 2:
                    P_dense, r_grown = _pack_children_inside_parent_dense_2d(
                        A_local,
                        r_local,
                        p_center,
                        p_radius,
                        k_local=min(k, max(1, len(child_idx) - 1)),
                        outer_sweeps=outer_sweeps,
                        mass_mode=mass_mode_2d,
                    )
                else:
                    P_dense, r_grown = _pack_children_inside_parent_dense_nd(
                        A_local,
                        r_local,
                        p_center,
                        p_radius,
                        k_local=min(k, max(1, len(child_idx) - 1)),
                        outer_sweeps=outer_sweeps,
                        mass_mode=mass_mode_nd,
                    )
                pos_child_global[child_idx] = P_dense
                rad_child_global[child_idx] = r_grown

        # store this level and use it as the parent for the next iteration
        top_pos_map, top_rad_map = {}, {}
        for i, c_lab in enumerate(lab_child):
            coords = tuple(pos_child_global[i].tolist() + [float(rad_child_global[i])])
            layout[(lev, int(c_lab))] = coords
            top_pos_map[int(c_lab)] = pos_child_global[i]
            top_rad_map[int(c_lab)] = float(rad_child_global[i])

    return layout


# -------------------#######
