import cv2
import numpy as np
import os
from copy import deepcopy


"""
Part 1: Gamma correction
"""


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gamma correction is used to correct the brightness of an image.
    The formula is:
        I_out = I_in ^ gamma
    where I_out is the output image, I_in is the input image, and gamma is the gamma value.

    Args:
        img (np.ndarray): The input image.
        gamma (float): The gamma value.

    Returns:
        np.ndarray: The output image.
    """
    # Normalize the image to [0, 1]
    img = img / 255.0

    # Apply gamma correction
    img = img**gamma

    # Denormalize the image to [0, 255]
    img = img * 255.0

    # Convert the image to uint8
    img = img.astype(np.uint8)

    return img


"""
Part 2: Histogram equalization
"""


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Histogram equalization is used to enhance the contrast of an image.
    The formula is:
        I_out = (I_in - I_min) / (I_max - I_min) * 255
    where I_out is the output image, I_in is the input image, I_min is the minimum intensity of the input image,
    and I_max is the maximum intensity of the input image.

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The output image.
    """

    img = img.astype(np.uint8)

    # Compute the histogram of the input image
    hist, _ = np.histogram(img.flatten(), 256, (0, 256))

    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Normalize the CDF to [0, 255]
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-10)
    cdf = np.clip(cdf, 0, 255).astype(np.uint8)

    # Apply the histogram equalization
    img = cdf[img]

    return img


"""
Bonus
"""


def histogram_equalization_color(img: np.ndarray) -> np.ndarray:
    ret = deepcopy(img)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = histogram_equalization(img_yuv[:, :, 0])
    ret = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return ret


def adaptive_histogram_equalization(img: np.ndarray, block_size: int) -> np.ndarray:
    """
    Adaptive histogram equalization is used to enhance the contrast of an image locally.
    The image is divided into blocks of size block_size x block_size, and histogram equalization
    is applied to each block separately.

    Args:
        img (np.ndarray): The input image.
        block_size (int): The size of the blocks for local histogram equalization.

    Returns:
        np.ndarray: The output image.
    """
    ret = np.zeros_like(img)

    # Apply adaptive histogram equalization
    for c in range(3):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(block_size, block_size))
        ret[:, :, c] = clahe.apply(img[:, :, c])

    # Convert the image to uint8
    img = img.astype(np.uint8)

    return ret


"""
Main function
"""


def main():
    img = cv2.imread("data/image_enhancement/input.bmp")

    # Modify the hyperparameter
    gamma_list = [1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4]

    merged = [deepcopy(img)]
    # Part 1: Gamma correction
    for gamma in gamma_list:
        gamma_correction_img = gamma_correction(img, gamma)
        cv2.putText(
            gamma_correction_img,
            f"Gamma = {gamma:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(
            f"output/image_enhancement/gamma_correction_{gamma:.2f}.png",
            gamma_correction_img,
        )
        merged.append(gamma_correction_img)

    cv2.putText(
        merged[0],
        "Original",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(
        "output/image_enhancement/gamma_correction.png",
        np.hstack(merged),
    )

    # Part 2: Image enhancement using the better balanced image as input
    histogram_equalization_img = histogram_equalization(img)

    cv2.imwrite(
        "output/image_enhancement/histogram_equalization.png",
        np.hstack([img, histogram_equalization_img]),
    )
    cv2.waitKey(0)

    # Bonus1: Color image enhancement
    histogram_equalization_color_img = histogram_equalization_color(img)
    cv2.imwrite(
        "output/image_enhancement/histogram_equalization_color.png",
        np.hstack([img, histogram_equalization_color_img]),
    )

    # Bonus2: Adaptive histogram equalization
    block_size = 16
    adaptive_histogram_equalization_img = adaptive_histogram_equalization(
        img, block_size
    )
    cv2.imwrite(
        "output/image_enhancement/adaptive_histogram_equalization.png",
        np.hstack([img, adaptive_histogram_equalization_img]),
    )


if __name__ == "__main__":
    os.makedirs("output/image_enhancement", exist_ok=True)
    main()
