from tqdm import trange
import multiprocessing
import os
from main import *


# Gaussian filter
def test_gaussian():
    plist = []
    input_img = cv2.imread("input_part1.jpg")
    for i in range(3, 10, 2):
        for j in range(1, 4):

            def wrapper(input_img, i, j):
                output_img = gaussian_filter(input_img, i, j)
                cv2.putText(
                    output_img,
                    f"kernel_size={i}, sigma={j}",
                    (20, 60),
                    0,
                    2,
                    (0, 0, 255),
                    4,
                )
                cv2.imwrite(f"output/gaussian_{i}_{j}.jpg", output_img)

            plist.append(
                multiprocessing.Process(target=wrapper, args=(input_img, i, j))
            )
            plist[-1].start()
    for p in plist:
        p.join()

    # Combine all the images
    gaussian_fnames = [
        f"output/gaussian_{i}_{j}.jpg" for i in range(3, 10, 2) for j in range(1, 4)
    ]
    gaussian_imgs = [cv2.imread(fname) for fname in gaussian_fnames]
    gaussian_imgs = [cv2.resize(img, (1328, 750)) for img in gaussian_imgs]
    # Make a 4x3 grid
    gaussian_imgs = np.array(gaussian_imgs).reshape(4, 3, 750, 1328, 3)
    gaussian_imgs = gaussian_imgs.transpose(0, 2, 1, 3, 4).reshape(4 * 750, 3 * 1328, 3)

    print(gaussian_imgs.shape)
    cv2.imwrite("output/gaussian.jpg", gaussian_imgs)


# Median filter
def test_median():
    plist = []
    input_img = cv2.imread("input_part1.jpg")
    for i in range(3, 10, 2):

        def wrapper(input_img, i):
            output_img = median_filter(input_img, i)
            cv2.putText(
                output_img,
                f"kernel_size={i}",
                (20, 60),
                0,
                2,
                (0, 0, 255),
                4,
            )
            cv2.imwrite(f"output/median_{i}.jpg", output_img)

        plist.append(multiprocessing.Process(target=wrapper, args=(input_img, i)))
        plist[-1].start()
    for p in plist:
        p.join()

    # Combine all the images
    median_fnames = [f"output/median_{i}.jpg" for i in range(3, 10, 2)]
    median_imgs = [cv2.imread(fname) for fname in median_fnames]
    # Horizontally stack the images
    median_imgs = np.hstack(median_imgs)
    cv2.imwrite("output/median.jpg", median_imgs)


def test_lapalaican():
    plist = []
    input_img = cv2.imread("input_part2.jpg")
    for i in [1, 2]:

        def wrapper(input_img, i):
            output_img = laplacian_sharpening(input_img, i)
            cv2.putText(
                output_img,
                f"filter_type={i}",
                (20, 60),
                0,
                2,
                (0, 0, 255),
                4,
            )
            cv2.imwrite(f"output/laplacian_{i}.jpg", output_img)

        plist.append(multiprocessing.Process(target=wrapper, args=(input_img, i)))
        plist[-1].start()
    for p in plist:
        p.join()

    # Combine all the images
    laplacian_fnames = [f"output/laplacian_{i}.jpg" for i in [1, 2]]
    laplacian_imgs = [cv2.imread(fname) for fname in laplacian_fnames]
    # Horizontally stack the images
    laplacian_imgs = np.hstack(laplacian_imgs)
    cv2.imwrite("output/laplacian.jpg", laplacian_imgs)


def main():
    os.makedirs("output", exist_ok=True)
    test_gaussian()
    test_median()
    test_lapalaican()


if __name__ == "__main__":
    main()
