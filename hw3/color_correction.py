import cv2
import numpy as np
import os


"""
White patch algorithm
"""


def white_patch_algorithm(img):
    img = img.astype(np.float32)
    img = img / 255
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    R_max, G_max, B_max = np.max(R), np.max(G), np.max(B)
    img[:, :, 0] = R / R_max
    img[:, :, 1] = G / G_max
    img[:, :, 2] = B / B_max
    return (img * 255).astype(np.uint8)


"""
Gray-world algorithm
"""


def gray_world_algorithm(img):
    img = img.astype(np.float32)
    img = img / 255
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    R_avg, G_avg, B_avg = np.mean(R), np.mean(G), np.mean(B)
    avg = (R_avg + G_avg + B_avg) / 3
    img[:, :, 0] = R * avg / R_avg
    img[:, :, 1] = G * avg / G_avg
    img[:, :, 2] = B * avg / B_avg
    return (img * 255).astype(np.uint8)


"""
Bonus 
"""


def other_white_balance_algorithm():
    raise NotImplementedError


"""
Main function
"""


def main():

    os.makedirs("result/color_correction", exist_ok=True)
    for i in range(2):
        img = cv2.imread(f"data/color_correction/input{i+1}.bmp")

        # White-balance algorithm
        white_patch_img = white_patch_algorithm(img)
        gray_world_img = gray_world_algorithm(img)

        cv2.imwrite(
            f"result/color_correction/white_patch_input{i+1}.bmp", white_patch_img
        )
        cv2.imwrite(
            f"result/color_correction/gray_world_input{i+1}.bmp", gray_world_img
        )


if __name__ == "__main__":
    main()
