import numpy as np
from PIL import Image
from collections import deque

THRESHOLD = 15
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def segment_image(image):
    H, W = image.shape
    labels = -1 * np.ones((H, W), dtype=int)
    segment_id = 0

    for i in range(H):
        for j in range(W):
            if labels[i][j] != -1:
                continue

            queue = deque([(i, j)])
            labels[i][j] = segment_id

            while queue:
                r, c = queue.popleft()

                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc

                    if 0 <= nr < H and 0 <= nc < W:
                        if labels[nr][nc] == -1:
                            if abs(int(image[r][c]) - int(image[nr][nc])) <= THRESHOLD:
                                labels[nr][nc] = segment_id
                                queue.append((nr, nc))

            segment_id += 1

    return labels


def main():
    img = Image.open("input.png").convert("L")
    arr = np.array(img)

    labels = segment_image(arr)

    output = Image.fromarray(labels.astype(np.uint16))
    output.save("output.png")


if __name__ == "__main__":
    main()