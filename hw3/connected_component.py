import cv2
import numpy as np
import os
from dsu import DSU
from random import randint


def get_label_color(label):
    if label == 0:
        return (0, 0, 0)
    np.random.seed(label)
    return tuple(np.random.randint(0, 255, 3))


"""
Binary transfer
"""


def to_binary(img):
    ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # use otsu's method to find the best threshold
    thresh, ret = cv2.threshold(ret, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret = cv2.bitwise_not(ret)
    print(f"Threshold: {thresh}")
    print(f"Binary image shape: {ret.shape}")
    return ret


"""
Two-pass algorithm
"""


def two_pass(img, connectivity):
    m, n = img.shape
    dsu = DSU(m * n)
    ret = np.zeros((m, n), dtype=np.uint32)
    for i in range(m):
        for j in range(n):
            if img[i][j] == 0:
                continue
            if connectivity == 4:
                if i > 0 and img[i - 1][j] == 255:
                    dsu.union(i * n + j, (i - 1) * n + j)
                if j > 0 and img[i][j - 1] == 255:
                    dsu.union(i * n + j, i * n + j - 1)
            elif connectivity == 8:
                if i > 0 and j > 0 and img[i - 1][j - 1] == 255:
                    dsu.union(i * n + j, (i - 1) * n + j - 1)
                if i > 0 and img[i - 1][j] == 255:
                    dsu.union(i * n + j, (i - 1) * n + j)
                if i > 0 and j < n - 1 and img[i - 1][j + 1] == 255:
                    dsu.union(i * n + j, (i - 1) * n + j + 1)
                if j > 0 and img[i][j - 1] == 255:
                    dsu.union(i * n + j, i * n + j - 1)
            else:
                raise ValueError("Connectivity should be 4 or 8")

    for i in range(m):
        for j in range(n):
            if img[i][j] == 0:
                continue
            root = dsu.find(i * n + j)
            assert root != 0
            ret[i][j] = root
    return ret


"""
TODO Seed filling algorithm
"""


def seed_filling(img, connectivity):
    raise NotImplementedError


"""
Bonus
"""


def other_cca_algorithm():
    raise NotImplementedError


"""
Color mapping
"""


def color_mapping(label):
    m, n = label.shape
    ret = np.zeros((m, n, 3), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            ret[i][j] = get_label_color(label[i][j])
    return ret


"""
Main function
"""


def main():
    out_dir = "result/connected_component"
    out_dir_two_pass = os.path.join(out_dir, "two_pass")
    out_dir_seed_filling = os.path.join(out_dir, "seed_filling")
    os.makedirs(out_dir_two_pass, exist_ok=True)
    os.makedirs(out_dir_seed_filling, exist_ok=True)
    connectivity_type = [4, 8]

    for i in range(2):
        print(f"Reading image {i+1}")
        img = cv2.imread(f"data/connected_component/input{i+1}.png")

        # TODO Part1: Transfer to binary image
        binary_img = to_binary(img)
        cv2.imwrite(os.path.join(out_dir, f"input{i+1}_binary.png"), binary_img)

        for connectivity in connectivity_type:
            print(f"Connectivity: {connectivity}")

            # TODO Part2: CCA algorithm
            two_pass_label = two_pass(binary_img, connectivity)
            # seed_filling_label = seed_filling(binary_img, connectivity)

            # TODO Part3: Color mapping
            two_pass_color = color_mapping(two_pass_label)
            # seed_filling_color = color_mapping(seed_filling_label)

            cv2.imwrite(
                os.path.join(out_dir_two_pass, f"input{i+1}_c{connectivity}.png"),
                two_pass_color,
            )
            # cv2.imwrite(
            #     os.path.join(out_dir_seed_filling, f"input{i+1}_c{connectivity}.png"),
            #     seed_filling_color,
            # )


if __name__ == "__main__":
    main()
