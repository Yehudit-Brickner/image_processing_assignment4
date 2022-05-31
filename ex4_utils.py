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
    #make a padded image
    img_l_pad=cv2.copyMakeBorder(img_l,k_size+disp_range[1],k_size+disp_range[1],k_size+disp_range[1],k_size+disp_range[1],cv2.BORDER_REFLECT_101)
    img_r_pad=cv2.copyMakeBorder(img_r,k_size+disp_range[1],k_size+disp_range[1],k_size+disp_range[1],k_size+disp_range[1],cv2.BORDER_REFLECT_101)


    # go over the image
    for r in range(img_l.shape[0]):
        for c in range(img_l.shape[1]):
            r1=r+k_size+disp_range[1]
            c1=c+k_size+disp_range[1]
            best_offset =0
            prev_ssd = float("inf")
            # go over the offset
            for offset in range(disp_range[0], disp_range[1]):
                ssd = 0
                if r1 - k_size >= 0 and r1 + k_size + 1 < img_l_pad.shape[0] and c1 - k_size >= 0 and c1 + k_size + 1 < img_l_pad.shape[1]:  # check that we are in range in the left image
                    if r1 - k_size >= 0 and r1 + k_size + 1 < img_r_pad.shape[0] and c1 - k_size - offset >= 0 and c1 + k_size - offset + 1 < img_r_pad.shape[1]:  # check that we are in range in the right image
                        ssd = np.sum((img_l_pad[r1 - k_size:r1 + k_size + 1, c1 - k_size:c1 + k_size + 1] - img_r_pad[r1 - k_size:r1 + k_size + 1,c1 - k_size - offset:c1 + k_size + 1 - offset]) ** 2)
                # check if smaller then prev
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset
            # put in the value in the new image
            new[r][c] = best_offset
            # if (r+1 < img_l.shape[0]):
            #     new[r+1][c] = best_offset
            # if c+1<=img_l.shape[1]:
            #     new[r ][c+1] = best_offset
            # if r+1 <img_l.shape[0] and c+1 <img_l.shape[1]:
            #     new[r+1][c + 1] = best_offset
    return new


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int,int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    # create a blank image
    new = np.zeros((img_l.shape[0], img_l.shape[1]))
    # make a padded image
    img_l_pad = cv2.copyMakeBorder(img_l, k_size + disp_range[1], k_size + disp_range[1], k_size + disp_range[1],k_size + disp_range[1], cv2.BORDER_REFLECT_101)
    img_r_pad = cv2.copyMakeBorder(img_r, k_size + disp_range[1], k_size + disp_range[1], k_size + disp_range[1],k_size + disp_range[1], cv2.BORDER_REFLECT_101)

    # go over the image
    for r in range(0,img_l.shape[0],2):
        for c in range(0,img_l.shape[1],2):
            r1 = r + k_size + disp_range[1]
            c1 = c + k_size + disp_range[1]
            best_offset = 0
            prev_ssd = 0
            # go over the offset
            for offset in range(disp_range[0], disp_range[1]):
                ssd = 0
                if r1 - k_size >= 0 and r1 + k_size < img_l_pad.shape[0] and c1 - k_size >= 0 and c1 + k_size < img_l_pad.shape[1]:  # check that we are in range
                    if r1 - k_size >= 0 and r1 + k_size < img_r_pad.shape[0] and c1 - k_size - offset >= 0 and c1 + k_size - offset < img_r_pad.shape[1]: # check that we are in range
                        top = np.sum((img_l_pad[r1 - k_size:r1 + k_size + 1, c1 - k_size:c1 + k_size + 1]* img_r_pad[r1 - k_size:r1 + k_size + 1, c1 - k_size - offset:c1 + k_size - offset + 1]))
                        bottom1 = np.sum((img_l_pad[r1 - k_size:r1 + k_size + 1, c1 - k_size:c1 + k_size + 1] * img_l_pad[r1 - k_size:r1 + k_size + 1, c1 - k_size :c1 + k_size + 1]))
                        bottom2 = np.sum((img_r_pad[r1 - k_size:r1 + k_size + 1, c1 - k_size- offset:c1 + k_size + 1- offset] * img_r_pad[r1 - k_size:r1 + k_size + 1, c1 - k_size - offset:c1 + k_size - offset + 1]))
                        if bottom1*bottom2!=0: # make sure im not dividing by zero
                            ssd=top/np.sqrt(bottom1*bottom2)
                        else:
                            print("zero")
                            # ssd=top
                            ssd=0
                # check if smaller then prev
                if ssd >= prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset
            # put in the value in the new image
            new[r][c] = best_offset
            if (r+1 < img_l.shape[0]):
                new[r+1][c] = best_offset
            if c+1<=img_l.shape[1]:
                new[r ][c+1] = best_offset
            if r+1 <img_l.shape[0] and c+1 <img_l.shape[1]:
                new[r+1][c + 1] = best_offset
    # print(new)

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
        A[2 * x] = np.array( [src_pnt[x][0], src_pnt[x][1], 1, 0, 0, 0, -dst_pnt[x][0] * src_pnt[x][0], -dst_pnt[x][0] * src_pnt[x][1],-dst_pnt[x][0]])
        A[2 * x + 1] = np.array([0, 0, 0, src_pnt[x][0], src_pnt[x][1], 1, -dst_pnt[x][1] * src_pnt[x][0], -dst_pnt[x][1] * src_pnt[x][1],-dst_pnt[x][1]])

    # get the svd of A and use it to get the h vector
    u, s, vh = np.linalg.svd(A)
    # reshape and divide by the number in the third row third col
    hom = vh[-1].reshape(3, 3)
    hom = hom / hom[2][2]


    # get the error
    # create a one array to add to connect with src and dst points
    one=np.ones(src_pnt.shape[0]).reshape(src_pnt.shape[0],1)
    # add the one and switch direction
    src_points = np.concatenate((src_pnt,one), axis=1).T
    dst_points = np.concatenate((dst_pnt,one), axis=1).T
    # warp the points
    diff=hom.dot(src_points)
    # normalize the points
    diff=diff/diff[2,:]
    # subtract the dest points
    diff=diff-dst_points
    # get the error
    error=np.sqrt(np.sum(diff**2))

    return hom, error


def getEquation(p1, p2):
    top = int(p1[1]) - int(p2[1])  # y1-y2
    bottom = int(p1[0]) - int(p2[0])  # x1-x2
    if bottom != 0:
        #slope
        slope = top / bottom
        # b
        b = -slope * int(p1[0]) + int(p1[1])
        # print("y=", slope, "x ", b)
        return slope, b
    else:
        #line is parallel to y
        # print("line is parallel to y")
        return 0, 0


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

    # MATCH THE STABBED POINTS TO TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT

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
    # we will find the smaller y value
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
    # match up the corners- looks off but this is because hoe cv2 plots the image
    src_p[tl, :] = bl_src
    src_p[tr, :] = br_src
    src_p[br, :] = tr_src
    src_p[bl, :] = tl_src




    # to make the mask we need to map out the shape that we stabbed
    # we will make 4 equations: TL-TR, BL-BR, TL-BL ,TR-BR
    # using these equation we will check per pixel if it is part of the mask or not
    mask1 = np.zeros((dst_img.shape[0], dst_img.shape[1], 3))
    maxX = max([dst_p[br][0], dst_p[tr][0]])
    minX = min([dst_p[bl][0], dst_p[tl][0]])
    maxY = max([dst_p[br][1], dst_p[bl][1]])
    minY = min([dst_p[tr][1], dst_p[tl][1]])
   
    TL_TR_slope, TL_TR_b = getEquation((dst_p[tl][0], dst_p[tl][1]), (dst_p[tr][0], dst_p[tr][1]))
    BL_BR_slope, BL_BR_b = getEquation((dst_p[bl][0], dst_p[bl][1]), (dst_p[br][0], dst_p[br][1]))
    TL_BL_slope, TL_BL_b = getEquation((dst_p[tl][0], dst_p[tl][1]), (dst_p[bl][0], dst_p[bl][1]))
    TR_BR_slope, TR_BR_b = getEquation((dst_p[br][0], dst_p[br][1]), (dst_p[tr][0], dst_p[tr][1]))

    # go over image
    for x in range(dst_img.shape[0]):
        for y in range(dst_img.shape[1]):
            if minX <= y <= maxX and minY <= x <= maxY:  # (opposite because of how plt plots)
                if TL_TR_slope * y + TL_TR_b <= x <= BL_BR_slope * y + BL_BR_b:  # check if the x is in the okay range (opposite because of how plt plots- should be checking the col)
                    if TL_BL_slope != 0 and TR_BR_slope != 0:  # make sure the left and right arent parallel to the y axis
                        if (x - TL_BL_b) / TL_BL_slope <= y <= (x - TR_BR_b) / TR_BR_slope:  # check if the y is in the okay range (opposite because of how plt plots- should be checking the row)
                            mask1[x][y][0] = 1
                            mask1[x][y][1] = 1
                            mask1[x][y][2] = 1
                    elif TL_BL_slope != 0 and TR_BR_slope == 0:  # if right is parallel to the y axis
                        if x <= dst_p[br][1] and (x - TL_BL_b) / TL_BL_slope <= y:
                            mask1[x][y][0] = 1
                            mask1[x][y][1] = 1
                            mask1[x][y][2] = 1

                    elif TL_BL_slope == 0 and TR_BR_slope != 0:  # if left is parallel to the y axis
                        if (x - TR_BR_b) / TR_BR_slope >= y and x >= dst_p[tl][0] :
                            mask1[x][y][0] = 1
                            mask1[x][y][1] = 1
                            mask1[x][y][2] = 1

                    else:  # both are parallel to y axis
                        if dst_p[tl][0] <= x <= dst_p[tr][0]:
                            mask1[x][y][0] = 1
                            mask1[x][y][1] = 1
                            mask1[x][y][2] = 1

    # get the homgraphy of the images
    hom, e = computeHomography(src_p, dst_p)
    theta = 1.5708
    # rotate the homagraph and shift so that the image is in the correct spot
    turn = np.array([[np.cos(theta), -np.sin(theta), (dst_img.shape[1]//2)-35],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]], dtype=np.float)
    hom = hom @ turn
    # warp the image
    src_out = cv2.warpPerspective(src_img, hom, (dst_img.shape[1], dst_img.shape[0]))

    # plt.imshow(src_out)
    # plt.show()
    # plt.imshow(mask1 -src_out)
    # plt.show()

    # connect the images
    out = dst_img * (1 - mask1) + src_out * (mask1)
    plt.imshow(out)
    plt.show()
