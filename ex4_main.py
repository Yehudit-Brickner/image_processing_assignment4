import os
from ex4_utils import *
import cv2
import time


def displayDepthImage(l_img, r_img, disparity_range=(0, 5), method=disparitySSD):
    p_size = 5
    st = time.time()


    d_ssd = method(l_img, r_img, disparity_range, p_size)
    print("Time: {:.3f} sec".format(time.time() - st))
    plt.matshow(d_ssd)
    # plt.imshow(d_ssd)
    plt.colorbar()
    plt.show()


def main():
    # Print your ID number
    print("ID:", 328601018)

    # Read images
    i = 0
    L = cv2.imread(os.path.join('input', 'pair%d-L.png' % i), 0) / 255.0
    R = cv2.imread(os.path.join('input', 'pair%d-R.png' % i), 0) / 255.0

    # L=cv2.imread('input/pair1-L.png',0)/255.0
    # R=cv2.imread('input/pair1-R.png',0)/255.0
    # # Display depth SSD
    displayDepthImage(L, R, (0, 4), method=disparitySSD)

    # Display depth NC
    # displayDepthImage(L, R, (0, 4), method=disparityNC)

    # src = np.array([[279, 552],
    #                 [372, 559],
    #                 [362, 472],
    #                 [277, 469]])
    # dst = np.array([[24, 566],
    #                 [114, 552],
    #                 [106, 474],
    #                 [19, 481]])
    # h, error = computeHomography(src, dst)
    #
    # print(h, error)
    #
    # dst = cv2.imread(os.path.join('input', 'billBoard.jpg'))[:, :, [2, 1, 0]] / 255.0
    # src = cv2.imread(os.path.join('input', 'car.jpg'))[:, :, [2, 1, 0]] / 255.0
    #
    # warpImag(src, dst)


if __name__ == '__main__':
    main()
