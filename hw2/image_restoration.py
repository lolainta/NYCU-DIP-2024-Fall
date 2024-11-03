import cv2
import numpy as np
import os

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
TODO Part 3: Constrained least squares filtering
"""


def constrained_least_square_filtering():
    raise NotImplementedError


"""
Bouns
"""


def other_restoration_algorithm():
    raise NotImplementedError


def compute_PSNR(image_original, image_restored):
    # PSNR = 10 * log10(max_pixel^2 / MSE)
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
        wiener_img = wiener_filtering(img_blurred, psf, snr=0.01)

        # TODO Part 3: Constrained least squares filtering
        # constrained_least_square_img = constrained_least_square_filtering()
        constrained_least_square_img = wiener_img

        print(f"\n---------- Testcase {i} ----------".format(i))
        print("Method: Wiener filtering")
        print(f"PSNR = {compute_PSNR(img_original, wiener_img)}\n")

        print("Method: Constrained least squares filtering")
        print(f"PSNR = {compute_PSNR(img_original, constrained_least_square_img)}\n")

        os.makedirs(f"output/image_restoration/testcase{i}", exist_ok=True)
        cv2.imwrite(f"output/image_restoration/testcase{i}/wiener_img.png", wiener_img)
        # cv2.imshow(
        #     "window", np.hstack([img_blurred, wiener_img, constrained_least_square_img])
        # )
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
