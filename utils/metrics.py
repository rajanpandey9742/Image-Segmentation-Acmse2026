import numpy as np

# 1. accuracy
def pixel_accuracy(pred, gt):
    same = np.sum(pred == gt)
    total = pred.size
    return same / total


# 2. simple rand score
def adjusted_rand_index(pred, gt):
    same = np.sum(pred == gt)
    total = len(pred)
    return same / total


# 3. constraints check
def check_constraints(image, labels, threshold=15):
    rows, cols = image.shape
    bad = 0
    unlabeled = np.sum(labels == -1)

    for r in range(rows):
        for c in range(cols):

            if r + 1 < rows:
                if labels[r][c] == labels[r+1][c]:
                    diff = abs(int(image[r][c]) - int(image[r+1][c]))
                    if diff > threshold:
                        bad += 1

            if c + 1 < cols:
                if labels[r][c] == labels[r][c+1]:
                    diff = abs(int(image[r][c]) - int(image[r][c+1]))
                    if diff > threshold:
                        bad += 1

    return {
        "unlabeled_pixels": int(unlabeled),
        "violations": bad,
        "passed": unlabeled == 0 and bad == 0
    }