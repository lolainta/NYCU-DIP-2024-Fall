import numpy as np
import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gaussian", action="store_true")
    parser.add_argument("-m", "--median", action="store_true")
    parser.add_argument("-l", "--laplacian", action="store_true")
    args = parser.parse_args()
    return args


def padding(input_img, kernel_size):
    ############### YOUR CODE STARTS HERE ###############
    pad_size = kernel_size // 2
    output_img = np.zeros(
        (input_img.shape[0] + 2 * pad_size, input_img.shape[1] + 2 * pad_size, 3)
    )
    output_img[pad_size:-pad_size, pad_size:-pad_size] = input_img

    # padding the top
    output_img[:pad_size, pad_size:-pad_size] = input_img[0]
    # padding the bottom
    output_img[-pad_size:, pad_size:-pad_size] = input_img[-1]
    # padding the left
    output_img[pad_size:-pad_size, :pad_size] = input_img[:, 0][:, None]
    # padding the right
    output_img[pad_size:-pad_size, -pad_size:] = input_img[:, -1][:, None]

    # padding the top-left
    output_img[:pad_size, :pad_size] = input_img[0, 0]
    # padding the top-right
    output_img[:pad_size, -pad_size:] = input_img[0, -1]
    # padding the bottom-left
    output_img[-pad_size:, :pad_size] = input_img[-1, 0]
    # padding the bottom-right
    output_img[-pad_size:, -pad_size:] = input_img[-1, -1]

    # print(output_img)
    # print(output_img.shape)
    ############### YOUR CODE ENDS HERE #################
    return output_img


def convolution(input_img, kernel):
    ############### YOUR CODE STARTS HERE ###############
    output_img = np.zeros_like(input_img, dtype=np.float32)
    kernel_size = kernel.shape[0]
    assert kernel_size == kernel.shape[1], "Kernel should be square"
    assert kernel_size % 2 == 1, "Kernel size should be odd"
    input_img = padding(input_img, kernel_size)
    for i in range(input_img.shape[0] - kernel_size + 1):
        for j in range(input_img.shape[1] - kernel_size + 1):
            for k in range(3):
                output_img[i, j, k] = np.sum(
                    input_img[i : i + kernel_size, j : j + kernel_size, k] * kernel
                )
    ############### YOUR CODE ENDS HERE #################
    return output_img


def gaussian_filter(input_img, kernel_size=5, sigma=1):
    ############### YOUR CODE STARTS HERE ###############
    kernel = np.array(
        [
            [
                np.exp(
                    -((i - kernel_size // 2) ** 2 + (j - kernel_size // 2) ** 2)
                    / (2 * sigma**2)
                )
                for j in range(kernel_size)
            ]
            for i in range(kernel_size)
        ]
    )
    kernel = kernel / np.sum(kernel)
    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, kernel)


def median_filter(input_img, kernel_size=5):
    ############### YOUR CODE STARTS HERE ###############
    output_img = np.zeros_like(input_img)
    input_img = padding(input_img, kernel_size)
    for i in range(input_img.shape[0] - kernel_size + 1):
        for j in range(input_img.shape[1] - kernel_size + 1):
            for k in range(3):
                output_img[i, j, k] = np.median(
                    input_img[i : i + kernel_size, j : j + kernel_size, k]
                )
    ############### YOUR CODE ENDS HERE #################
    return output_img


def laplacian_sharpening(input_img, filter_type=1):
    ############### YOUR CODE STARTS HERE ###############
    kernel = (
        np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ]
        )
        if filter_type == 1
        else np.array(
            [
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1],
            ]
        )
    )
    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, kernel)


def main():
    args = parse_args()
    print(args)
    if args.gaussian:
        input_img = cv2.imread("input_part1.jpg")
        output_img = gaussian_filter(input_img)
    elif args.median:
        input_img = cv2.imread("input_part1.jpg")
        output_img = median_filter(input_img)
    elif args.laplacian:
        input_img = cv2.imread("input_part2.jpg")
        output_img = laplacian_sharpening(input_img)
    else:
        raise ValueError("Please specify the operation to perform")
    cv2.imwrite("output.jpg", output_img)


if __name__ == "__main__":
    main()
    # test()
