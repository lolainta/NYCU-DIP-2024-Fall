import cv2
import numpy as np
import os


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
    # Compute the histogram of the input image
    hist, _ = np.histogram(img.flatten(), 256, (0, 256))

    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Normalize the CDF to [0, 255]
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    # Apply the histogram equalization
    img = cdf[img]

    return img


"""
Bonus
"""


def other_enhancement_algorithm():
    raise NotImplementedError


"""
Main function
"""


def main():
    img = cv2.imread("data/image_enhancement/input.bmp")

    # Modify the hyperparameter
    gamma_list = [1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4]

    merged = [img]
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

    cv2.imwrite(
        "output/image_enhancement/gamma_correction.png",
        np.vstack(merged),
    )

    # Part 2: Image enhancement using the better balanced image as input
    histogram_equalization_img = histogram_equalization(img)

    cv2.imwrite(
        "output/image_enhancement/histogram_equalization.png",
        histogram_equalization_img,
    )
    # cv2.imshow("Histogram equalization", np.vstack([img, histogram_equalization_img]))
    cv2.waitKey(0)


if __name__ == "__main__":
    os.makedirs("output/image_enhancement", exist_ok=True)
    main()
