import cv2
import numpy as np
import matplotlib.pyplot as plt


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """


    # create a blank image
    new = np.zeros((img_l.shape[0], img_l.shape[1]))
    # go over all the value in the left image
    # for r in range(img_l.shape[0]):
    #     for c in range(img_l.shape[1]):
    #         # go over the "window" around r,c
    #         # min = float("inf")
    #         # best_d = -1
    #         sum = 0
    #         for i in range(k_size * 2 + 1):
    #             for j in range(k_size * 2 + 1):
    #                 # check if the values are in the other image
    #                 if 0 <= (r + i - (k_size * 2 + 1) // 2) < img_r.shape[0] and 0 <= (c + j - (k_size * 2 + 1) // 2) < img_r.shape[1]:
    #                     # add to sum this number from the equation
    #                     sum += (img_l[r][c] - img_r[r + i - (k_size * 2 + 1) // 2][c + j - (k_size * 2 + 1) // 2]) ** 2
    #         # put the sum into the new image
    #         new[r][c] = sum
    # offset_adjust = 255 / disp_range[1]
    # for r in range(k_size, img_l.shape[0]-k_size):
    #     for c in range(k_size, img_l.shape[1]-k_size):
    for r in range(img_l.shape[0]):
        for c in range(img_l.shape[1]):
            best_offset = 0
            prev_ssd = float("inf")
            for offset in range(disp_range[0], disp_range[1]):
                ssd = 0

                for v in range(-k_size, k_size+1):
                    for u in range(-k_size, k_size+1):
                        if 0<=r+v-offset<img_r.shape[0] and 0<=c+u-offset<img_r.shape[1] :
                            ssd += ((img_l[r,c]) - (img_r[r + v-offset,c + u - offset]))**2


                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset
            new[r][c]=best_offset
    print(new)

    return new


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

    # create a blank image
    new = np.zeros((img_l.shape[0], img_l.shape[1]))
    # go over all the value in the left image
    for r in range(img_l.shape[0]):
        for c in range(img_l.shape[1]):
            # go over the "window" around r,c
            top = 0
            bottom1 = 0
            bottom2 = 0
            for i in range(k_size * 2 + 1):
                for j in range(k_size * 2 + 1):
                    # check if the values are in the other image
                    if 0 <= (r + i - disp_range[0] // 2) < img_r.shape[0] and 0 <= (c + j - disp_range[1] // 2) < \
                            img_r.shape[1]:
                        # add to top this number from the equation
                        top += (img_l[r][c] * img_r[r + i - disp_range[0] // 2][c + j - disp_range[1] // 2]) ** 2
                        bottom1 += (img_r[r][c] * img_r[r + i - disp_range[0] // 2][c + j - disp_range[1] // 2]) ** 2
                        bottom2 += (img_l[r][c] * img_l[r + i - disp_range[0] // 2][c + j - disp_range[1] // 2]) ** 2
                    # if 0 <= (disp_range[0]//2+i-disp_range[0]//2) < img_l.shape[0] and 0 <= (disp_range[1]//2+j-disp_range[1]//2) < img_l.shape[1] :
                    #     # and 0 <= disp_range[0] // 2 < img_l.shape[0] and 0 <= disp_range[1] // 2 < img_l.shape[1]
                    #     bottom2 += (img_l[disp_range[0] // 2][disp_range[1] // 2] *img_l[disp_range[0] // 2 + i - disp_range[0] // 2][disp_range[1] // 2 + j - disp_range[1] // 2]) ** 2
                    # if 0 <= (r + i - (k_size * 2 + 1) // 2) < img_r.shape[0] and 0 <= (c + j - (k_size * 2 + 1) // 2) < img_r.shape[1]:
                    #     # add to top this number from the equation
                    #     top += (img_l[r][c] * img_r[r + i - (k_size * 2 + 1)// 2][c + j - (k_size * 2 + 1) // 2]) ** 2
                    #     bottom1 += (img_r[r][c] * img_r[r + i - (k_size * 2 + 1) // 2][c + j - (k_size * 2 + 1)// 2]) ** 2
                    #     bottom2 += (img_l[r][c] * img_l[r+ i -(k_size * 2 + 1) // 2][c+ j -(k_size * 2 + 1) // 2]) ** 2
            # put the sum into the new image
            if (bottom1 * bottom2) != 0:
                new[r][c] = top / np.sqrt(bottom1 * bottom2)
            else:
                new[r][c] = top

    return new


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    # create vector A
    A = np.zeros((src_pnt.shape[0] * 2, 9))
    # A=np.array([])
    for x in range(src_pnt.shape[0]):
        A[2 * x] = np.array(
            [src_pnt[x][0], src_pnt[x][1], 1, 0, 0, 0, -dst_pnt[x][0] * src_pnt[x][0], -dst_pnt[x][0] * src_pnt[x][1],
             -dst_pnt[x][0]])
        A[2 * x + 1] = np.array(
            [0, 0, 0, src_pnt[x][0], src_pnt[x][1], 1, -dst_pnt[x][1] * src_pnt[x][0], -dst_pnt[x][1] * src_pnt[x][1],
             -dst_pnt[x][1]])

    # get the svd of A and use it to get the h vector
    u, s, vh = np.linalg.svd(A)
    # reshape and divide by the number in the third row third col
    hom = vh[-1].reshape(3, 3)
    hom = hom / hom[2][2]
    # print(hom)
    # hom2, e2 = cv2.findHomography(src_pnt,dst_pnt)
    # print(hom2)

    # get the error
    diff = src_pnt[0:3, :] - dst_pnt[0:3, :]
    ddd = hom.dot(diff)
    error = np.sqrt((np.sum(ddd) ** 2))

    return hom, error


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """
    # pick 4 points in the destination image
    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######

    # find 2 smallest x take x with smallest y = top left corner
    # other is bottem left
    # other 2 points smaller y is top right
    # last is bottom right

    # we will find the  small x values, in dst_p
    minx1 = float("inf")
    minx2 = float("inf")
    minxrow1 = -1
    minxrow2 = -1
    for row in range(len(dst_p)):
        if dst_p[row][0] <= minx1:
            minx1 = dst_p[row][0]
            minxrow1 = row
    for row in range(len(dst_p)):
        if minx1 <= dst_p[row][0] <= minx2 and row != minxrow1:
            minx2 = dst_p[row][0]
            minxrow2 = row
    # we will the smaller y value
    # and get the topleft and bottomleft corners
    if (dst_p[minxrow1][1] < dst_p[minxrow2][1]):
        tl = minxrow1
        bl = minxrow2
    else:
        tl = minxrow2
        bl = minxrow1

    # we will find the 2 point that arent topleft and bottomleft
    # and find the bigger y value
    # and get the topright and bottomright
    lst = [0, 1, 2, 3]
    lst.remove(tl)
    lst.remove(bl)
    if dst_p[lst[0]][1] > dst_p[lst[1]][1]:
        tr = lst[1]
        br = lst[0]
    else:
        tr = lst[0]
        br = lst[1]

    # print("read clockwise\n", "topleft", tl, dst_p[tl, :], "topright", tr, dst_p[tr, :], "bottomright", br,
    #       dst_p[br, :], "bpttomleft", bl, dst_p[bl, :])

    # corners of the src_img
    tl_src = np.array([0, 0])
    tr_src = np.array([0, src_img.shape[1]])
    br_src = np.array([src_img.shape[0], src_img.shape[1]])
    bl_src = np.array([src_img.shape[0], 0])
    # create the src_p array so that the corners match
    src_p = np.zeros((4, 2))
    src_p[tl, :] = bl_src
    src_p[tr, :] = br_src
    src_p[br, :] = tr_src
    src_p[bl, :] = tl_src

    # make the mask of the image
    mask = np.zeros((dst_img.shape[0], dst_img.shape[1], 3))
    for j in range(dst_img.shape[0]):
        for i in range(dst_img.shape[1]):
            if i > dst_p[tl][0] and i > dst_p[bl][0] and i < dst_p[br][0] and i < dst_p[tr][0] \
                    and j > dst_p[tl][1] and j > dst_p[tr][1] and j < dst_p[br][1] and j < dst_p[bl][1]:
                mask[j][i][0] = 1
                mask[j][i][1] = 1
                mask[j][i][2] = 1
    # plt.imshow(mask)
    # plt.show()

    # get the homgraphy of teh images
    hom, e = computeHomography(src_p, dst_p)
    theta= 1.5708
    turn=np.array([[np.cos(theta), -np.sin(theta),dst_img.shape[0]*3//4],
             [np.sin(theta), np.cos(theta), 0],
             [0,0,1]],dtype=np.float)
    hom=hom@turn
    print(hom)
    # hom1 , e1= cv2.findHomography(src_p.astype(float), dst_p.astype(float))
    # print(hom1)
    # warp the image
    src_out = cv2.warpPerspective(src_img, hom, (dst_img.shape[1], dst_img.shape[0]))
    plt.imshow(src_out)
    plt.show()

    # connect the images
    out = dst_img * (1 - mask) + src_out * (mask)
    plt.imshow(out)
    plt.show()
