"""
Evaluation metrics for segmentation quality assessment.
Used by segment.py evaluate command when ground-truth masks are available.
"""

import numpy as np


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute pixel-level accuracy between predicted and ground-truth masks.

    Because segment IDs may differ, we relabel both masks so each unique
    connected component is matched by majority vote (Hungarian-free greedy).

    For correctness checking under the competition rules, the right metric is
    whether adjacency constraints are satisfied — see check_constraints().
    """
    # Flatten and compute agreement via contingency table matching
    pred_flat = pred.ravel()
    gt_flat   = gt.ravel()

    pred_ids = np.unique(pred_flat)
    gt_ids   = np.unique(gt_flat)

    # Build contingency matrix (pred_id vs gt_id → pixel overlap count)
    contingency = {}
    for p, g in zip(pred_flat, gt_flat):
        contingency[(p, g)] = contingency.get((p, g), 0) + 1

    # Greedy matching: assign each pred_id to the gt_id it overlaps most
    matched = 0
    used_gt = set()
    # Sort by overlap descending
    pairs = sorted(contingency.items(), key=lambda x: -x[1])
    pred_matched = set()
    for (p, g), cnt in pairs:
        if p not in pred_matched and g not in used_gt:
            matched += cnt
            pred_matched.add(p)
            used_gt.add(g)

    return matched / len(pred_flat)


def adjusted_rand_index(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index between two flat label arrays.
    ARI = 1.0 means perfect agreement; 0.0 = random; can be negative.

    This is ID-agnostic — only pairwise co-membership matters.
    """
    n = len(labels_pred)
    if n == 0:
        return 0.0

    # Build contingency table
    classes_pred = np.unique(labels_pred)
    classes_true = np.unique(labels_true)

    contingency = np.zeros((len(classes_pred), len(classes_true)), dtype=np.int64)
    pred_idx = {c: i for i, c in enumerate(classes_pred)}
    true_idx = {c: i for i, c in enumerate(classes_true)}

    for p, t in zip(labels_pred, labels_true):
        contingency[pred_idx[p], true_idx[t]] += 1

    # ARI formula
    sum_comb_c = np.sum([_comb2(n_ij) for n_ij in contingency.ravel()])
    sum_comb_a = sum(_comb2(a) for a in contingency.sum(axis=1))
    sum_comb_b = sum(_comb2(b) for b in contingency.sum(axis=0))
    comb_n     = _comb2(n)

    expected = (sum_comb_a * sum_comb_b) / comb_n if comb_n > 0 else 0
    maximum  = 0.5 * (sum_comb_a + sum_comb_b)

    if maximum - expected == 0:
        return 1.0

    return (sum_comb_c - expected) / (maximum - expected)


def _comb2(n: int) -> int:
    """C(n, 2) = n*(n-1)/2"""
    return n * (n - 1) // 2


def check_constraints(image_array: np.ndarray, labels: np.ndarray,
                       threshold: int = 15) -> dict:
    """
    Verify that the segmentation satisfies the problem constraints:
      1. Every pixel has exactly one label (trivially true from BFS).
      2. Adjacent pixels in the SAME segment have |intensity diff| <= threshold.
      3. Every pixel is assigned (no -1 labels remain).

    Returns a dict with violation counts (should all be 0 for a correct solution).
    """
    H, W = image_array.shape
    violations = 0
    unlabeled  = int(np.sum(labels == -1))

    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(H):
        for c in range(W):
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    same_seg = labels[r, c] == labels[nr, nc]
                    diff = abs(int(image_array[r, c]) - int(image_array[nr, nc]))
                    if same_seg and diff > threshold:
                        violations += 1

    # Each edge counted twice (r→nr and nr→r), halve it
    violations //= 2

    return {
        "unlabeled_pixels": unlabeled,
        "same_segment_threshold_violations": violations,
        "passed": unlabeled == 0 and violations == 0,
    }