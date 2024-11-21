import cv2
import numpy as np
import os
from copy import deepcopy

"""
Part 1: Motion blur PSF generation
"""


def generate_motion_blur_psf(shape: tuple, length: int, angle: int) -> np.ndarray:

    # Initialize the PSF
    psf = np.zeros(shape)

    # Compute the center of the PSF
    center = (shape[0] // 2, shape[1] // 2)

    # Compute the angle in radians
    angle = np.radians(angle)

    # Compute the direction of the motion blur
    direction = (np.cos(angle), np.sin(angle))

    # Compute the length of the motion blur
    length = int(length)

    # Compute the start point of the motion blur
    start = (center[0] - direction[0] * length, center[1] - direction[1] * length)

    # Compute the end point of the motion blur
    end = (center[0] + direction[0] * length, center[1] + direction[1] * length)

    # Draw the motion blur
    cv2.line(
        psf,
        (int(start[1]), int(start[0])),
        (int(end[1]), int(end[0])),
        (255, 255, 255),
        1,
    )

    print("PSF shape:", psf.shape)
    cv2.imwrite("output/image_restoration/psf.png", psf)

    # Normalize the PSF
    psf = psf / np.sum(psf)

    return psf


"""
Part 2: Wiener filtering
"""


def wiener_filtering(img: np.ndarray, psf: np.ndarray, snr: float) -> np.ndarray:
    filtered_image = np.zeros(img.shape)
    for channel in range(img.shape[2]):
        # Compute the Fourier transform of the input image
        img_fft = np.fft.fft2(img[:, :, channel])
        img_fft = np.fft.fftshift(img_fft)

        # Compute the Fourier transform of the PSF
        psf_fft = np.fft.fft2(psf, img.shape[:2])
        psf_fft = np.fft.fftshift(psf_fft)

        # Compute the Wiener filter
        wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + snr + 1e-10)

        # Compute the Fourier transform of the restored image
        restored_img_fft = img_fft * wiener_filter

        # Compute the restored image
        restored_img = np.fft.ifft2(restored_img_fft)
        restored_img = np.fft.ifftshift(restored_img)

        # Normalize the restored image
        restored_img = np.real(restored_img)

        # Clip the restored image to [0, 255]
        restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)

        filtered_image[:, :, channel] = restored_img

    return filtered_image


"""
Part 3: Constrained least squares filtering
"""


def constrained_least_square_filtering(
    img: np.ndarray, psf: np.ndarray, lambd: float = 1e-3
) -> np.ndarray:
    """
    Constrained least squares filtering is used to restore an image from a blurred image.
    The formula is:
        I_restored = argmin ||I_blurred - I_restored * PSF||^2 + lambda * ||Laplacian(I_restored)||^2
    where I_restored is the restored image, I_blurred is the blurred image, PSF is the point spread function,
    and lambda is the regularization parameter.

    Args:
        img (np.ndarray): The blurred image.
        psf (np.ndarray): The point spread function.

    Returns:
        np.ndarray: The restored image.
    """
    restored_image = np.zeros(img.shape)
    for channel in range(3):
        img_fft = np.fft.fft2(img[:, :, channel])
        img_fft = np.fft.fftshift(img_fft)

        psf_fft = np.fft.fft2(psf, img.shape[:2])
        psf_fft = np.fft.fftshift(psf_fft)

        # Compute the Fourier transform of the Laplacian operator
        laplacian_fft = np.fft.fft2(
            np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
            img.shape[:2],
        )
        # laplacian_fft = np.fft.fftshift(laplacian_fft)

        # Compute the denominator of the CLS filter
        denominator = np.abs(psf_fft) ** 2 + lambd * np.abs(laplacian_fft)

        # Compute the CLS filter
        cls_filter = np.conj(psf_fft) / denominator

        # Compute the Fourier transform of the restored image
        restored_img_fft = img_fft * cls_filter

        # Compute the restored image
        restored_img = np.fft.ifft2(restored_img_fft)
        restored_img = np.fft.ifftshift(restored_img)

        # Normalize the restored image
        restored_img = np.real(restored_img)

        # Clip the restored image to [0, 255]
        restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)

        restored_image[:, :, channel] = restored_img
    return restored_image


"""
Bouns
"""


def other_restoration_algorithm():
    raise NotImplementedError


def compute_PSNR(image_original, image_restored):
    # PSNR = 10 * log10(max_pixel ^ 2 / MSE)
    psnr = 10 * np.log10(
        255**2
        / np.mean(
            (image_original.astype(np.float64) - image_restored.astype(np.float64)) ** 2
        )
    )
    return psnr


"""
Main function
"""


def main():

    for i in range(1, 3):
        print(f"\n---------- Testcase {i} ----------".format(i))
        os.makedirs(f"output/image_restoration/testcase{i}", exist_ok=True)

        img_original = cv2.imread(
            f"data/image_restoration/testcase{i}/input_original.png"
        )
        img_blurred = cv2.imread(
            f"data/image_restoration/testcase{i}/input_blurred.png"
        )

        assert img_blurred.shape == img_original.shape
        shape = img_original.shape

        # Part 1: Motion blur PSF generation
        length = 20
        angle = 135
        psf = generate_motion_blur_psf(shape[:2], length, angle)

        # Part 2: Wiener filtering
        wieners = [deepcopy(img_blurred)]
        cv2.putText(
            wieners[0],
            "Original",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for snr in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
            print(f"Method: Wiener filtering with SNR = {snr}")
            wiener_img = wiener_filtering(img_blurred, psf, snr=snr)
            cv2.putText(
                wiener_img,
                f"SNR = {snr}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            wieners.append(wiener_img)
            cv2.imwrite(
                f"output/image_restoration/testcase{i}/wiener_{snr}.png", wiener_img
            )
            print(f"PSNR = {compute_PSNR(img_original, wiener_img)}\n")

        cv2.imwrite(
            f"output/image_restoration/testcase{i}/wiener_img.png", np.hstack(wieners)
        )

        # Part 3: Constrained least squares filtering
        clses = [deepcopy(img_blurred)]
        cv2.putText(
            clses[0],
            "Original",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for lambd in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
            print(f"Method: Constrained least squares filtering with lambda = {lambd}")
            cls_img = constrained_least_square_filtering(img_blurred, psf, lambd=lambd)
            cv2.putText(
                cls_img,
                f"Lambda = {lambd}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            clses.append(cls_img)
            print(f"PSNR = {compute_PSNR(img_original, cls_img)}\n")
            cv2.imwrite(
                f"output/image_restoration/testcase{i}/cls_{lambd}.png", cls_img
            )
        cv2.imwrite(
            f"output/image_restoration/testcase{i}/cls_img.png", np.hstack(clses)
        )

        cv2.waitKey(0)


if __name__ == "__main__":
    main()
